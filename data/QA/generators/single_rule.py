import json
import os
import sys
from typing import Dict
import requests
import asyncio

# Deepseek API integration
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

async def generate_qa_pair(section: Dict, api_key: str) -> Dict:
    assert section['name'] and section['content'], "Invalid rule section provided"
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

    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    
    try:
        result = json.loads(response.json()['choices'][0]['message']['content'])
        if isinstance(result, list) and len(result) > 0:
            return result[0]  # Return first QA pair
    except (KeyError, json.JSONDecodeError):
        pass
    
    return None

async def main(filepath: str, output_path: str):
    rules = []
    with open(filepath,
              'r',
              encoding='utf-8') as f:
        rules = json.load(f)
        print('Loaded rules from file')
        print(f'Number of rules: {len(rules)}')

    qa = []
    finished_rules = []

    if open(output_path, 'r').read():
        old_qa = json.load(open(output_path, 'r'))
        finished_rules = [qa_pair['rules'] for qa_pair in old_qa]
        qa = old_qa

    qa_pairs = {}
     
    rules = {' > '.join(rule['path'] + [rule['title']]): rule for rule in rules['rules']}
    for rule in [r for (k,r) in rules.items() if k not in finished_rules]:
        rule_query = {}
        print(f"Generating QA for: {rule['title']}")
        print(f"rule from json is {rule}")
        print(f"rule['path'] is {rule['path']} - thing to join is {rule['path'] + [rule['title']]}")
        rule_query['name'] = ' > '.join(rule['path'] + [rule['title']])
        rule_query['content'] = rule['text']
        qa_task = generate_qa_pair(rule_query, os.getenv("DEEPSEEK_API_KEY"))
        if qa_task:

            qa_pairs[rule_query['name']] = qa_task
            #print(f"qa_pair is {qa_pair}")
            #qa = json.loads(qa_pair)
            #qa['rules'] = rule_query['name']
            #qa.append((rule_query['name'], qa_pair))
        else:
            print(f"Failed to generate QA for: {rule['name']}")
        #qa.append(generate_qa_pair(rule_query, os.getenv("DEEPSEEK_API_KEY")))

    print(f"Generated {len(qa)} QA pairs")
    asyncio.gather(qa_pairs.values())
    with open(output_path, "w") as f:
        json.dump(qa, f, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: python single_rule_qa_generator.py <rules_json_file> <qa output file>")
    main(sys.argv[1], sys.argv[2])