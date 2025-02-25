import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import markdown
from bs4 import BeautifulSoup
import logging
from pathlib import Path

@dataclass
class RuleSection:
    """Represents a section of rules with hierarchical structure."""
    title: str
    level: int  # Header level (1-6)
    content: str
    parent: Optional['RuleSection'] = None
    children: List['RuleSection'] = field(default_factory=list)
    
    def get_full_path(self) -> str:
        """Get the full hierarchical path of this section."""
        if self.parent:
            return f"{self.parent.get_full_path()}/{self.title}"
        return self.title

@dataclass
class Rule:
    """Represents a single rule with its context and metadata."""
    content: str
    section: str  # Full hierarchical path
    section_title: str
    references: List[str] = field(default_factory=list)
    systems: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    properties: Dict[str, str] = field(default_factory=dict)

class SRDProcessor:
    """Processes SRD markdown into structured Rule objects."""
    
    def __init__(self):
        # Common game systems that rules might reference
        self.game_systems = {
            'combat': ['attack', 'damage', 'weapon', 'armor', 'shield', 'initiative'],
            'spellcasting': ['spell', 'cast', 'magic', 'scroll', 'component'],
            'skills': ['skill', 'check', 'ability', 'proficiency'],
            'equipment': ['item', 'gear', 'tool', 'weapon', 'armor'],
            'character': ['class', 'race', 'background', 'level', 'ability score'],
            'adventuring': ['rest', 'travel', 'exploration', 'light', 'vision']
        }
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Patterns for identifying rule properties
        self.property_patterns = [
            (r'\*\*([^:]+):\*\*\s*([^*]+)', 'general'),  # **Property:** Value
            (r'Prerequisites?:(.*?)(?:\n|$)', 'prerequisite'),
            (r'See also:(.*?)(?:\n|$)', 'reference')
        ]
    
    def _convert_markdown_to_html(self, markdown_text: str) -> str:
        """Convert markdown to HTML for easier processing."""
        return markdown.markdown(markdown_text, extensions=['extra'])
    
    def _extract_header_level(self, tag: BeautifulSoup) -> int:
        """Extract header level from HTML tag."""
        if tag.name and tag.name.startswith('h'):
            return int(tag.name[1])
        return 0
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove markdown emphasis that wasn't converted
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        return text
    
    def _extract_properties(self, text: str) -> Tuple[str, Dict[str, str], List[str]]:
        """Extract properties, prerequisites, and references from text."""
        properties = {}
        prerequisites = []
        references = []
        
        # Process each pattern
        for pattern, ptype in self.property_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if ptype == 'general':
                    key, value = match.groups()
                    properties[key.strip()] = self._clean_text(value)
                elif ptype == 'prerequisite':
                    prereqs = match.group(1).strip()
                    prerequisites.extend(self._clean_text(p) for p in prereqs.split(','))
                elif ptype == 'reference':
                    refs = match.group(1).strip()
                    references.extend(self._clean_text(r) for r in refs.split(','))
                
                # Remove the matched text to avoid double-processing
                text = text.replace(match.group(0), '')
        
        return text.strip(), properties, prerequisites, references
    
    def _identify_systems(self, text: str, properties: Dict[str, str]) -> List[str]:
        """Identify game systems referenced in the rule."""
        systems = set()
        combined_text = text.lower() + ' ' + ' '.join(properties.values()).lower()
        
        for system, keywords in self.game_systems.items():
            if any(keyword in combined_text for keyword in keywords):
                systems.add(system)
        
        return list(systems)
    
    def _build_section_tree(self, html_content: str) -> List[RuleSection]:
        """Build a hierarchical tree of sections from HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')
        sections = []
        current_section = None
        section_stack = []
        
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']):
            if tag.name.startswith('h'):
                level = self._extract_header_level(tag)
                title = self._clean_text(tag.text)
                
                # Pop stack until we find the parent section
                while section_stack and section_stack[-1].level >= level:
                    section_stack.pop()
                
                # Create new section
                current_section = RuleSection(
                    title=title,
                    level=level,
                    content='',
                    parent=section_stack[-1] if section_stack else None
                )
                
                if section_stack:
                    section_stack[-1].children.append(current_section)
                else:
                    sections.append(current_section)
                
                section_stack.append(current_section)
            
            elif tag.name == 'p' and current_section:
                current_section.content += '\n' + self._clean_text(tag.text)
        
        return sections
    
    def _section_to_rules(self, section: RuleSection) -> List[Rule]:
        """Convert a section and its children into Rule objects."""
        rules = []
        
        # Process current section
        if section.content.strip():
            # Extract properties and clean content
            clean_content, properties, prerequisites, references = self._extract_properties(
                section.content
            )
            
            # Create rule if there's content
            if clean_content or properties:
                systems = self._identify_systems(clean_content, properties)
                
                rules.append(Rule(
                    content=clean_content,
                    section=section.get_full_path(),
                    section_title=section.title,
                    references=references,
                    systems=systems,
                    prerequisites=prerequisites,
                    properties=properties
                ))
        
        # Process child sections
        for child in section.children:
            rules.extend(self._section_to_rules(child))
        
        return rules
    
    def process_file(self, filepath: str) -> List[Rule]:
        """Process an SRD markdown file into Rule objects."""
        self.logger.info(f"Processing file: {filepath}")
        
        # Read markdown file
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                markdown_text = f.read()
        except Exception as e:
            self.logger.error(f"Error reading file: {e}")
            return []
        
        # Convert to HTML and build section tree
        html_content = self._convert_markdown_to_html(markdown_text)
        sections = self._build_section_tree(html_content)
        
        # Convert sections to rules
        rules = []
        for section in sections:
            rules.extend(self._section_to_rules(section))
        
        self.logger.info(f"Processed {len(rules)} rules")
        return rules

def main():
    # Example usage
    processor = SRDProcessor()
    
    # Example markdown content
    example_content = """### **The Bard**

**Weapons:** Simple weapons, hand crossbows, longswords, rapiers, shortswords
**Tools:** Three musical instruments of your choice
**Saving Throws:** Dexterity, Charisma
**Skills:** Choose any three

A bard weaves magic through words and music to inspire allies, demoralize foes, manipulate minds, create illusions, and heal wounds.

#### **Spellcasting**
You have learned to untangle and reshape the fabric of reality in harmony with your wishes and music.
See also: Magic chapter, Bard spell list

**Spells Known:** 4 1st-level spells of your choice from the bard spell list"""

    # Write example to temporary file
    with open('example_srd.md', 'w') as f:
        f.write(example_content)
    
    # Process the file
    rules = processor.process_file('example_srd.md')
    
    # Print results
    for rule in rules:
        print(f"\nSection: {rule.section}")
        print(f"Title: {rule.section_title}")
        print(f"Content: {rule.content}")
        print(f"Properties: {rule.properties}")
        print(f"Systems: {rule.systems}")
        print(f"References: {rule.references}")
        print("-" * 80)

if __name__ == "__main__":
    main()