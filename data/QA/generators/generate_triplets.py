#!/usr/bin/env python3
"""
Generate training triplets for fine-tuning a cross-encoder model.
Each triplet consists of:
- question: A question about a rule
- pos: The correct rule text (augmented)
- neg: An incorrect rule text (augmented)
"""

import json
import os
import sys
import time
import random
import argparse
import asyncio
import aiohttp
from typing import Dict, List, Set, Optional, Tuple, Callable, Any
from pathlib import Path
from tqdm import tqdm
import importlib.util
import inspect

# Deepseek API integration
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
MAX_CONCURRENT_REQUESTS = 20  # Limit concurrent API calls
TEMP_DIR = "/tmp/dnd_triplet_generator"

# Token tracking
class TokenTracker:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.total_requests = 0
        self.last_report = 0
    
    def update(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens = self.prompt_tokens + self.completion_tokens
        self.total_requests += 1
        
        # Report every 100 requests
        if self.total_requests - self.last_report >= 100:
            print(f"\nToken usage: {self}")
            self.last_report = self.total_requests
    
    def __str__(self):
        return f"Requests: {self.total_requests}, Prompt: {self.prompt_tokens}, Completion: {self.completion_tokens}, Total: {self.total_tokens}"

# Global token tracker
token_tracker = TokenTracker()

def load_augmentation_functions(module_path: str) -> Dict[str, Callable]:
    """Load augmentation functions from a module."""
    spec = importlib.util.spec_from_file_location("augmentation_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Get all functions from the module
    functions = {}
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and name.startswith('augment_'):
            functions[name] = obj
    
    return functions

async def generate_question(section: Dict, api_key: str, session: aiohttp.ClientSession) -> Optional[str]:
    """Generate a question for a rule section using the Deepseek API."""
    assert section['name'] and section['content'], "Invalid rule section provided"
    
    # Use the section name for the temp file
    safe_name = section['name'].replace('/', '_').replace('\\', '_')
    temp_file = Path(TEMP_DIR) / f"{safe_name}_question.txt"
    
    # Check if we already have a result for this rule
    if temp_file.exists():
        try:
            with open(temp_file, 'r') as f:
                return f.read().strip()
        except IOError:
            # If the file is corrupted, we'll regenerate
            pass
    
    prompt = f"""Generate ONE specific question about the D&D 5e rules section provided below.
The question should be answerable using ONLY the content from this specific section.
Return ONLY the question text, with no additional formatting or explanation.

Rules Section Name: {section['name']}
Section Content:
{section['content']}
"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that generates questions about D&D 5th edition rules."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 200
    }

    try:
        async with session.post(DEEPSEEK_API_URL, headers=headers, json=payload) as response:
            response.raise_for_status()
            response_data = await response.json()
            
            # Track token usage
            if 'usage' in response_data:
                token_tracker.update(
                    response_data['usage'].get('prompt_tokens', 0),
                    response_data['usage'].get('completion_tokens', 0)
                )
            
            question = response_data['choices'][0]['message']['content'].strip()
            
            # Save to temp file
            os.makedirs(TEMP_DIR, exist_ok=True)
            with open(temp_file, 'w') as f:
                f.write(question)
            
            return question
    except (KeyError, aiohttp.ClientError) as e:
        print(f"\nError generating question for {section['name']}: {str(e)}")
    except Exception as e:
        print(f"\nUnexpected error for {section['name']}: {str(e)}")
    
    return None

async def process_rule_batch(rules_batch: List[Dict], api_key: str, 
                            augmentation_func: Callable, 
                            all_rules: List[Dict],
                            max_triplets: int,
                            semaphore: asyncio.Semaphore,
                            pbar: tqdm) -> List[Dict]:
    """Process a batch of rules with rate limiting."""
    results = []
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for rule in rules_batch:
            if len(tasks) >= max_triplets:
                break
                
            task = process_single_rule(rule, api_key, session, augmentation_func, all_rules, semaphore, pbar)
            tasks.append(task)
        
        for task_result in asyncio.as_completed(tasks):
            triplet = await task_result
            if triplet:
                results.append(triplet)
                pbar.update(1)
            
            if len(results) >= max_triplets:
                break
    
    return results

async def process_single_rule(rule: Dict, api_key: str, session: aiohttp.ClientSession,
                             augmentation_func: Callable, all_rules: List[Dict],
                             semaphore: asyncio.Semaphore, pbar: tqdm) -> Optional[Dict]:
    """Process a single rule with rate limiting."""
    async with semaphore:
        # Generate a question for this rule
        question = await generate_question(rule, api_key, session)
        
        if not question:
            return None
        
        # Create the positive example using the augmentation function
        section_data = {
            'title': rule['name'].split(' > ')[-1],
            'path': rule['name'].split(' > ')[:-1],
            'text': rule['content'],
            'references': [],  # Placeholder, could be populated if available
            'scope': 'rule'    # Placeholder, could be populated if available
        }
        
        pos_text = augmentation_func(section_data)
        
        # Select a random different rule for the negative example
        other_rules = [r for r in all_rules if r['name'] != rule['name']]
        if not other_rules:
            return None
            
        neg_rule = random.choice(other_rules)
        neg_section_data = {
            'title': neg_rule['name'].split(' > ')[-1],
            'path': neg_rule['name'].split(' > ')[:-1],
            'text': neg_rule['content'],
            'references': [],  # Placeholder
            'scope': 'rule'    # Placeholder
        }
        
        neg_text = augmentation_func(neg_section_data)
        
        # Create the triplet
        triplet = {
            'question': question,
            'pos': pos_text,
            'neg': neg_text,
            'pos_rule': rule['name'],
            'neg_rule': neg_rule['name']
        }
        
        return triplet

async def main():
    parser = argparse.ArgumentParser(description="Generate training triplets for cross-encoder fine-tuning")
    parser.add_argument("rules_file", help="Path to the JSON file containing rules")
    parser.add_argument("output_file", help="Path to save the generated triplets")
    parser.add_argument("--augmentation-module", default="src/augmentation_functions.py", 
                        help="Path to the Python module with augmentation functions")
    parser.add_argument("--augmentation-func", default="augment_with_title", 
                        help="Name of the augmentation function to use")
    parser.add_argument("--max-triplets", type=int, default=1000, 
                        help="Maximum number of triplets to generate")
    parser.add_argument("--concurrent", type=int, default=MAX_CONCURRENT_REQUESTS,
                        help=f"Maximum concurrent API requests (default: {MAX_CONCURRENT_REQUESTS})")
    
    args = parser.parse_args()
    
    # Ensure temp directory exists
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable not set")
    
    # Load rules
    with open(args.rules_file, 'r', encoding='utf-8') as f:
        rules_data = json.load(f)
        print(f'Loaded rules from file: {len(rules_data["rules"])} rules found')
    
    # Load augmentation functions
    augmentation_functions = load_augmentation_functions(args.augmentation_module)
    if args.augmentation_func not in augmentation_functions:
        available_funcs = ", ".join(augmentation_functions.keys())
        raise ValueError(f"Augmentation function '{args.augmentation_func}' not found. Available functions: {available_funcs}")
    
    augmentation_func = augmentation_functions[args.augmentation_func]
    print(f"Using augmentation function: {args.augmentation_func}")
    
    # Prepare rules to process
    rules = []
    for rule in rules_data['rules']:
        rule_query = {
            'name': ' > '.join(rule['path'] + [rule['title']]),
            'content': rule['text']
        }
        rules.append(rule_query)
    
    # Load existing triplets if output file exists
    existing_triplets = []
    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, 'r') as f:
                existing_triplets = json.load(f)
                print(f"Loaded {len(existing_triplets)} existing triplets")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading existing triplets: {e}")
    
    # Calculate how many more triplets we need to generate
    num_to_generate = min(args.max_triplets - len(existing_triplets), len(rules))
    if num_to_generate <= 0:
        print(f"Already have {len(existing_triplets)} triplets, which meets or exceeds the requested {args.max_triplets}")
        return
    
    print(f"Generating {num_to_generate} new triplets")
    
    # Process rules in parallel with rate limiting
    semaphore = asyncio.Semaphore(args.concurrent)
    
    # Use tqdm for progress tracking
    with tqdm(total=num_to_generate, desc="Generating triplets") as pbar:
        # Shuffle rules to get a good variety
        random.shuffle(rules)
        
        # Process rules
        new_triplets = await process_rule_batch(
            rules[:num_to_generate], 
            api_key, 
            augmentation_func, 
            rules,
            num_to_generate,
            semaphore,
            pbar
        )
    
    # Combine with existing triplets
    all_triplets = existing_triplets + new_triplets
    
    # Write final results
    with open(args.output_file, "w") as f:
        json.dump(all_triplets, f, indent=2)
    
    print(f"\nCompleted processing. Total triplets: {len(all_triplets)}")
    print(f"Final token usage: {token_tracker}")

if __name__ == "__main__":
    asyncio.run(main())
