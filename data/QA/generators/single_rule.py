import json
import os
import sys
import time
import hashlib
import aiohttp
import asyncio
from typing import Dict, List, Set, Optional
from pathlib import Path

# Deepseek API integration
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
MAX_CONCURRENT_REQUESTS = 50  # Limit concurrent API calls
TEMP_DIR = "/tmp/dnd_qa_generator"

async def generate_qa_pair(section: Dict, api_key: str, session: aiohttp.ClientSession) -> Optional[Dict]:
    """Generate a QA pair for a rule section using the Deepseek API."""
    assert section['name'] and section['content'], "Invalid rule section provided"
    
    # Create a unique ID for this rule to use in temp files
    rule_id = hashlib.md5(section['name'].encode()).hexdigest()
    temp_file = Path(TEMP_DIR) / f"{rule_id}.json"
    
    # Check if we already have a result for this rule
    if temp_file.exists():
        try:
            with open(temp_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # If the file is corrupted, we'll regenerate
            pass
    
    prompt = f"""Generate exactly ONE question/answer pair about the D&D 5e rules section provided below.
The question should be answerable using ONLY the content from this specific section.
The answer should be a direct response to the question. Any explanatory text should be kept solely in the rules explanation field.
Use this exact JSON format:
[{{"question": "question text", "answer": "answer text", "rules_explanation":"An explanation of the rules in the section"}}]

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
            {"role": "system", "content": "You are a helpful assistant that generates question/answer pairs about D&D 5th edition rules."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 500
    }

    try:
        async with session.post(DEEPSEEK_API_URL, headers=headers, json=payload) as response:
            response.raise_for_status()
            response_data = await response.json()
            
            result = json.loads(response_data['choices'][0]['message']['content'])
            if isinstance(result, list) and len(result) > 0:
                qa_pair = result[0]
                
                # Save to temp file
                os.makedirs(TEMP_DIR, exist_ok=True)
                with open(temp_file, 'w') as f:
                    json.dump(qa_pair, f)
                
                return qa_pair
    except (KeyError, json.JSONDecodeError, aiohttp.ClientError) as e:
        print(f"Error generating QA for {section['name']}: {str(e)}")
    except Exception as e:
        print(f"Unexpected error for {section['name']}: {str(e)}")
    
    return None

async def process_rule_batch(rules_batch: List[Dict], api_key: str, semaphore: asyncio.Semaphore) -> List[Dict]:
    """Process a batch of rules with rate limiting."""
    results = []
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for rule_query in rules_batch:
            task = process_single_rule(rule_query, api_key, session, semaphore)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
    
    return [r for r in results if r is not None]

async def process_single_rule(rule_query: Dict, api_key: str, session: aiohttp.ClientSession, 
                             semaphore: asyncio.Semaphore) -> Optional[Dict]:
    """Process a single rule with rate limiting."""
    async with semaphore:
        print(f"Processing: {rule_query['name']}")
        qa_pair = await generate_qa_pair(rule_query, api_key, session)
        
        if qa_pair:
            qa_pair['rules'] = rule_query['name']
            print(f"✓ Generated QA for: {rule_query['name']}")
            return qa_pair
        else:
            print(f"✗ Failed to generate QA for: {rule_query['name']}")
            return None

async def main(filepath: str, output_path: str):
    # Ensure temp directory exists
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable not set")
    
    # Load rules
    with open(filepath, 'r', encoding='utf-8') as f:
        rules_data = json.load(f)
        print('Loaded rules from file')
        print(f'Number of rules: {len(rules_data["rules"])}')
    
    # Load existing QA pairs if output file exists and is not empty
    qa_pairs = []
    finished_rules = set()
    
    try:
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            with open(output_path, 'r') as f:
                qa_pairs = json.load(f)
                finished_rules = {' > '.join(qa_pair['rules']) for qa_pair in qa_pairs}
                print(f"Loaded {len(qa_pairs)} existing QA pairs")
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading existing QA pairs: {e}")
        # Continue with empty qa_pairs
    
    # Prepare rules to process
    rules = {' > '.join(rule['path'] + [rule['title']]): rule for rule in rules_data['rules']}
    rules_to_process = []
    
    for rule_name, rule in rules.items():
        if rule_name not in finished_rules:
            rule_query = {
                'name': rule_name,
                'content': rule['text']
            }
            rules_to_process.append(rule_query)
    
    print(f"Found {len(rules_to_process)} rules to process")
    
    # Process rules in parallel with rate limiting
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    new_qa_pairs = await process_rule_batch(rules_to_process, api_key, semaphore)
    
    # Combine with existing QA pairs
    qa_pairs.extend(new_qa_pairs)
    
    # Write final results
    with open(output_path, "w") as f:
        json.dump(qa_pairs, f, indent=4)
    
    print(f"Completed processing. Total QA pairs: {len(qa_pairs)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: python single_rule.py <rules_json_file> <qa output file>")
    
    asyncio.run(main(sys.argv[1], sys.argv[2]))
