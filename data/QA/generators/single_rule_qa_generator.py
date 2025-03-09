import os
import json
import mistune
from mistune.renderers.markdown import BaseRenderer
from typing import List, Dict
import requests
import time

# Custom Markdown parser
class SectionParser(BaseRenderer):
    def __init__(self):
        super().__init__()
        self.sections = []
        self.current_section = {'name': '', 'content': []}
        self.current_level = 0

    def heading(self, text, level, **attrs):
        if self.current_section['name']:
            self.sections.append({
                'name': self.current_section['name'],
                'content': '\n'.join(self.current_section['content']).strip()
            })
        
        self.current_section = {
            'name': text.strip(),
            'content': []
        }
        self.current_level = level
        return ''

    def paragraph(self, text):
        if self.current_section['name']:
            self.current_section['content'].append(text.strip())
        return ''

    def finalize(self):
        if self.current_section['name']:
            self.sections.append({
                'name': self.current_section['name'],
                'content': '\n'.join(self.current_section['content']).strip()
            })

def parse_markdown(md_content: str) -> List[Dict]:
    parser = SectionParser()
    markdown = mistune.Markdown(renderer=parser)
    markdown(md_content)
    parser.finalize()
    return [section for section in parser.sections if section['content']]

# Deepseek API integration
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

def generate_qa_pair(section: Dict, api_key: str) -> Dict:
    prompt = f"""Generate exactly ONE question/answer pair about the D&D 5e rules section provided below.
The question should be answerable using ONLY the content from this specific section.
Use this exact JSON format:
[{{"question": "question text", "answer": "answer text", "rule section": "section name"}}]

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

def main(markdown_path: str, output_path: str):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Deepseek API key not found in environment variables")

    with open(markdown_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    sections = parse_markdown(md_content)
    qa_pairs = []

    for section in sections:
        try:
            qa = generate_qa_pair(section, api_key)
            if qa:
                qa['rule section'] = section['name']
                qa_pairs.append(qa)
                print(f"Generated QA for: {section['name']}")
            time.sleep(1)  # Rate limiting
        except Exception as e:
            print(f"Error processing {section['name']}: {str(e)}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2)

    print(f"Generated {len(qa_pairs)} QA pairs. Saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate D&D 5e QA pairs from markdown rules')
    parser.add_argument('input_file', help='Path to input markdown file')
    parser.add_argument('output_file', help='Path to output JSON file')
    args = parser.parse_args()

    main(args.input_file, args.output_file)