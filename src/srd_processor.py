import re
import json
import markdown
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
import spacy

# Make sure to download required NLTK data
# nltk.download('punkt')

class SRDProcessor:
    def __init__(self, srd_path):
        """Initialize the SRD processor with the path to the markdown file."""
        self.srd_path = srd_path
        self.nlp = spacy.load("en_core_web_sm")
        self.rules_data = []
        self.terms_dict = {}
        self.rule_relationships = []
        
    def load_and_parse_markdown(self):
        """Load and convert markdown to HTML for structured parsing."""
        with open(self.srd_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert markdown to HTML
        html_content = markdown.markdown(md_content)
        
        # Parse HTML with BeautifulSoup
        self.soup = BeautifulSoup(html_content, 'html.parser')
        
        return self.soup
    
    def extract_headings_hierarchy(self):
        """Extract hierarchical structure from markdown headings."""
        headings = {}
        current_h1 = None
        current_h2 = None
        current_h3 = None
        
        for tag in self.soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            if tag.name == 'h1':
                current_h1 = tag.text.strip()
                current_h2 = None
                current_h3 = None
                headings[current_h1] = {}
            elif tag.name == 'h2' and current_h1:
                current_h2 = tag.text.strip()
                current_h3 = None
                headings[current_h1][current_h2] = {}
            elif tag.name == 'h3' and current_h2:
                current_h3 = tag.text.strip()
                headings[current_h1][current_h2][current_h3] = {}
            # Add more levels as needed
        
        self.headings_hierarchy = headings
        return headings
    
    def extract_sections(self):
        """Extract content sections based on heading hierarchy."""
        sections = []
        heading_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
        
        # Process each heading and its content
        for i, tag in enumerate(self.soup.find_all(heading_tags + ['p', 'ul', 'ol', 'table'])):
            if tag.name in heading_tags:
                # If it's a heading, start a new section
                heading_level = int(tag.name[1])
                section = {
                    'id': f"section_{len(sections)}",
                    'heading': tag.text.strip(),
                    'level': heading_level,
                    'content': [],
                    'path': [],  # Will store the path in the hierarchy
                }
                
                # Find the path in the hierarchy
                current_path = []
                for prev_section in reversed(sections):
                    if prev_section['level'] < heading_level:
                        current_path = prev_section['path'] + [prev_section['heading']]
                        break
                
                section['path'] = current_path
                sections.append(section)
            else:
                # If it's content and we have at least one section
                if sections:
                    # Add content to the most recent section
                    text_content = tag.text.strip()
                    if text_content:  # Only add non-empty content
                        sections[-1]['content'].append({
                            'type': tag.name,
                            'text': text_content
                        })
        
        self.sections = sections
        return sections
    
    def identify_rule_blocks(self):
        """Identify blocks of text that likely represent rules."""
        rule_blocks = []
        
        for section in self.sections:
            # Initialize a new rule based on the section
            rule = {
                'id': section['id'],
                'title': section['heading'],
                'level': section['level'],
                'path': section['path'],
                'text': [],
                'type': 'UNKNOWN',  # Will be classified later
                'references': [],
                'terms': []
            }
            
            # Combine all content into text
            for content in section['content']:
                rule['text'].append(content['text'])
            
            rule['text'] = '\n'.join(rule['text'])
            
            # Only add rules that have content
            if rule['text'].strip():
                rule_blocks.append(rule)
        
        self.rule_blocks = rule_blocks
        return rule_blocks
    
    def extract_terms(self):
        """Extract game-specific terms from the rule text."""
        game_terms = set()
        print('')
        # First pass: collect capitalized terms that might be game concepts
        for num, rule in enumerate(self.rule_blocks):
            print(f"\033[AExtracting terms for rule {num + 1}/{len(self.rule_blocks)}")
            doc = self.nlp(rule['text'])
            
            # Look for capitalized multi-word terms or specific patterns
            for sent in doc.sents:
                # Find potential capitalized terms (2+ words)
                matches = re.finditer(r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', sent.text)
                for match in matches:
                    term = match.group(0)
                    game_terms.add(term)
                
                # Find terms in quotes that might be game concepts
                matches = re.finditer(r'"([^"]+)"', sent.text)
                for match in matches:
                    term = match.group(1)
                    game_terms.add(term)
            
            # Also look for terms that might be in a defined format
            # For example, D&D often uses Title Case for special abilities
            title_case_terms = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', rule['text'])
            for term in title_case_terms:
                if len(term.split()) > 1:  # Only multi-word terms
                    game_terms.add(term)
        
        # Second pass: find where these terms are defined
        print('')
        term_definitions = {}
        for num, rule in enumerate(self.rule_blocks):

            print(f"\033[AProcessing definition locations for rule {num + 1}/{len(self.rule_blocks)}")
            doc = self.nlp(rule['text'])
            
            for term in game_terms:
                # Simple definition pattern: "Term: definition"
                pattern = f"{term}:\\s+([^.]+)"
                matches = re.search(pattern, rule['text'])
                if matches and matches.group(1) is not None:
                    #print(f"matches.group(1) = {matches.group(1)}")
                    definition = matches.group(1).strip()
                    term_definitions[term] = {
                        'definition': definition,
                        'source_rule_id': rule['id']
                    }
                
                # Alternative pattern: "Term. Definition starting sentence"
                pattern = f"{term}\\.\\s+([A-Z][^.]+)"
                matches = re.search(pattern, rule['text'])
                if matches and matches.group(1) is not None:
                    definition = matches.group(1).strip()
                    term_definitions[term] = {
                        'definition': definition,
                        'source_rule_id': rule['id']
                    }
        
        self.terms_dict = term_definitions
        
        # Now add found terms to each rule
        for rule in self.rule_blocks:
            for term in game_terms:
                if term.lower() in rule['text'].lower():
                    rule['terms'].append(term)
        
        return term_definitions
    
    def identify_cross_references(self):
        """Identify references between rules."""
        #return self.rule_relationships
        # Common reference patterns in RPG rules
        reference_patterns = [
            r'see\s+(?:the\s+)?([^.,]+)',
            r'as\s+described\s+in\s+(?:the\s+)?([^.,]+)',
            r'refer\s+to\s+(?:the\s+)?([^.,]+)',
            r'according\s+to\s+(?:the\s+)?([^.,]+)',
            r'in\s+the\s+([^.,]+)\s+section',
        ]
        
        for rule in self.rule_blocks:
            for pattern in reference_patterns:
                matches = re.finditer(pattern, rule['text'], re.IGNORECASE)
                for match in matches:
                    reference = match.group(1).strip()
                    # Don't add duplicates
                    if reference not in rule['references']:
                        rule['references'].append(reference)
            
            # Also check for heading references
            for section in self.sections:
                if section['heading'] != rule['title'] and section['heading'] in rule['text'] and section['heading'] not in rule['references']:
                    rule['references'].append(section['heading'])
        
        # Build relationship graph
        for rule in self.rule_blocks:
            for reference in rule['references']:
                # Try to find the target rule
                target_rules = [r for r in self.rule_blocks if reference.lower() in r['title'].lower()]
                for target in target_rules:
                    self.rule_relationships.append({
                        'source': rule['id'],
                        'target': target['id'],
                        'type': 'REFERENCES'
                    })
        
        return self.rule_relationships
    
    def classify_rule_types(self):
        """Classify rules into different types based on content analysis."""
        # Keywords and patterns for different rule types
        core_patterns = [
            r'basic\s+rule',
            r'core\s+mechanic',
            r'fundamental',
            r'always\s+applies',
        ]
        
        exception_patterns = [
            r'exception',
            r'however',
            r'but\s+if',
            r'unless',
            r'except\s+when',
            r'normally.+but',
            r'does\s+not\s+apply',
        ]
        
        definition_patterns = [
            r'^[A-Z][^.]+:',  # Starts with capitalized term followed by colon
            r'is\s+defined\s+as',
            r'refers\s+to',
            r'means',
        ]
        
        example_patterns = [
            r'example',
            r'for\s+instance',
            r'for\s+example',
            r'to\s+illustrate',
        ]
        
        # Check each rule against patterns
        for rule in self.rule_blocks:
            text = rule['text'].lower()
            
            # Check if it contains a table
            if '<table>' in text or '|' in text and '-+-' in text:
                rule['type'] = 'TABLE'
                continue
                
            # Check if it's an example
            for pattern in example_patterns:
                if re.search(pattern, text):
                    rule['type'] = 'EXAMPLE'
                    break
            
            # If not already classified, check if it's a definition
            if rule['type'] == 'UNKNOWN':
                for pattern in definition_patterns:
                    if re.search(pattern, rule['text']):
                        rule['type'] = 'DEFINITION'
                        break
            
            # If not already classified, check if it's an exception
            if rule['type'] == 'UNKNOWN':
                for pattern in exception_patterns:
                    if re.search(pattern, text):
                        rule['type'] = 'EXCEPTION'
                        break
            
            # If not already classified, check if it's a core rule
            if rule['type'] == 'UNKNOWN':
                for pattern in core_patterns:
                    if re.search(pattern, text):
                        rule['type'] = 'CORE_RULE'
                        break
            
            # Default to derived rule if not otherwise classified
            if rule['type'] == 'UNKNOWN':
                rule['type'] = 'DERIVED_RULE'
        
        return self.rule_blocks
    
    def detect_rule_scope(self):
        """Determine the scope of each rule (combat, character creation, etc.)."""
        # Keywords for different scopes
        scope_keywords = {
            'COMBAT': ['attack', 'damage', 'hit', 'critical', 'initiative', 'armor class', 
                       'hit points', 'round', 'turn', 'action', 'bonus action'],
            'CHARACTER_CREATION': ['character', 'race', 'class', 'background', 'ability score', 
                                  'skill', 'proficiency', 'feat', 'level up'],
            'SPELLCASTING': ['spell', 'cast', 'magic', 'scroll', 'wizard', 'sorcerer', 
                            'warlock', 'spell slot', 'concentration'],
            'EQUIPMENT': ['weapon', 'armor', 'shield', 'tool', 'item', 'gear', 'gold', 'silver'],
            'EXPLORATION': ['travel', 'journey', 'rest', 'exhaustion', 'environment', 
                           'light', 'vision', 'movement'],
            'SOCIAL': ['charisma', 'persuasion', 'deception', 'intimidation', 'interaction']
        }
        
        for rule in self.rule_blocks:
            text = rule['text'].lower()
            scope_scores = {}
            
            # Count keywords for each scope
            for scope, keywords in scope_keywords.items():
                score = 0
                for keyword in keywords:
                    score += text.count(keyword.lower())
                scope_scores[scope] = score
            
            # Assign the scope with the highest score if above threshold
            if max(scope_scores.values(), default=0) > 0:
                rule['scope'] = max(scope_scores.items(), key=lambda x: x[1])[0]
            else:
                rule['scope'] = 'GENERAL'
        
        return self.rule_blocks
    
    def extract_metadata(self):
        """Extract additional metadata for each rule."""
        complexity_indicators = {
            'SIMPLE': ['simple', 'basic', 'easy', 'straightforward'],
            'MEDIUM': ['additional', 'also', 'furthermore', 'however'],
            'COMPLEX': ['complex', 'complicated', 'exception', 'unless', 'except when']
        }
        
        for rule in self.rule_blocks:
            text = rule['text'].lower()
            
            # Determine complexity
            complexity_scores = {}
            for complexity, indicators in complexity_indicators.items():
                score = sum(text.count(indicator) for indicator in indicators)
                complexity_scores[complexity] = score
            
            # Default to MEDIUM if no clear indicator
            if max(complexity_scores.values(), default=0) == 0:
                rule['complexity'] = 'MEDIUM'
            else:
                rule['complexity'] = max(complexity_scores.items(), key=lambda x: x[1])[0]
            
            # Determine applicability
            if any(class_name.lower() in text for class_name in 
                  ['barbarian', 'bard', 'cleric', 'druid', 'fighter', 'monk', 
                   'paladin', 'ranger', 'rogue', 'sorcerer', 'warlock', 'wizard']):
                rule['applicability'] = 'CLASS_SPECIFIC'
            elif any(race_name.lower() in text for race_name in 
                    ['dwarf', 'elf', 'halfling', 'human', 'dragonborn', 
                     'gnome', 'half-elf', 'half-orc', 'tiefling']):
                rule['applicability'] = 'RACE_SPECIFIC'
            else:
                rule['applicability'] = 'UNIVERSAL'
            
            # Estimate frequency
            if any(term in text for term in ['rare', 'uncommon', 'special case', 'specific circumstance']):
                rule['frequency'] = 'RARE'
            else:
                rule['frequency'] = 'COMMON'
        
        return self.rule_blocks
    
    def build_rule_graph(self):
        """Build a complete graph representation of rules."""
        graph = {
            'nodes': [],
            'edges': []
        }
        
        # Add all rules as nodes
        for rule in self.rule_blocks:
            node = {
                'id': rule['id'],
                'label': rule['title'],
                'type': rule['type'],
                'scope': rule.get('scope', 'UNKNOWN'),
                'complexity': rule.get('complexity', 'MEDIUM')
            }
            graph['nodes'].append(node)
        
        # Add all relationships as edges
        for relation in self.rule_relationships:
            edge = {
                'source': relation['source'],
                'target': relation['target'],
                'type': relation['type']
            }
            graph['edges'].append(edge)
        
        # Add term definition relationships
        for term, term_info in self.terms_dict.items():
            # Find rules that use this term
            for rule in self.rule_blocks:
                if term in rule['terms'] and rule['id'] != term_info['source_rule_id']:
                    edge = {
                        'source': rule['id'],
                        'target': term_info['source_rule_id'],
                        'type': 'USES_TERM',
                        'term': term
                    }
                    graph['edges'].append(edge)
        
        self.graph = graph
        return graph
    
    def process(self):
        """Run the full processing pipeline."""
        print("Loading and parsing SRD markdown...")
        self.load_and_parse_markdown()
        
        print("Extracting heading hierarchy...")
        self.extract_headings_hierarchy()
        
        print("Extracting content sections...")
        self.extract_sections()
        
        print("Identifying rule blocks...")
        self.identify_rule_blocks()
        
        print("Extracting game terms...")
        self.extract_terms()
        
        print("Identifying cross-references...")
        self.identify_cross_references()
        
        print("Classifying rule types...")
        self.classify_rule_types()
        
        print("Detecting rule scope...")
        self.detect_rule_scope()
        
        print("Extracting metadata...")
        self.extract_metadata()
        
        print("Building rule graph...")
        self.build_rule_graph()
        
        print("Processing complete!")
        return {
            'rules': self.rule_blocks,
            'terms': self.terms_dict,
            'relationships': self.rule_relationships,
            'graph': self.graph
        }
    
    def save_output(self, output_path):
        """Save processed data to JSON file."""
        output = {
            'rules': self.rule_blocks,
            'terms': self.terms_dict,
            'relationships': self.rule_relationships,
            'graph': self.graph
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        
        print(f"Output saved to {output_path}")
        
        # Also save individual components for easier analysis
        with open(output_path.replace('.json', '_rules.json'), 'w', encoding='utf-8') as f:
            json.dump(self.rule_blocks, f, indent=2)
        
        with open(output_path.replace('.json', '_terms.json'), 'w', encoding='utf-8') as f:
            json.dump(self.terms_dict, f, indent=2)
        
        with open(output_path.replace('.json', '_graph.json'), 'w', encoding='utf-8') as f:
            json.dump(self.graph, f, indent=2)


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python srd_processor.py <path_to_srd.md> [output_path.json]")
        sys.exit(1)
    
    srd_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "srd_processed.json"
    
    processor = SRDProcessor(srd_path)
    processor.process()
    processor.save_output(output_path)
