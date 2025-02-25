from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

@dataclass
class Rule:
    """Represents a single rule with its context."""
    content: str
    section: str
    references: List[str]  # References to other rules
    systems: List[str]     # Affected game systems

@dataclass
class RulePair:
    """Represents a pair of related rules."""
    rule1: Rule
    rule2: Rule
    relationship_type: str  # e.g., "prerequisite", "modification", "exception"

@dataclass
class QAPair:
    """Represents a generated question-answer pair."""
    question: str
    answer: str
    source_rules: List[Rule]
    context: str

class RuleRelationshipDetector:
    """Identifies relationships between rules using various heuristics."""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Common patterns that indicate rule relationships
        self.reference_patterns = [
            r"see (?:also )?(?:rule )?(\d+(?:\.\d+)*)",
            r"as described in (?:rule )?(\d+(?:\.\d+)*)",
            r"according to (?:rule )?(\d+(?:\.\d+)*)",
            r"follows the rules for (\w+)",
            r"uses the same (?:rules|mechanics) as (\w+)"
        ]
        
        # Patterns that suggest rule modifications or exceptions
        self.modification_patterns = [
            r"except",
            r"however",
            r"unless",
            r"instead of",
            r"replaces",
            r"modifies"
        ]
    
    def find_explicit_references(self, rule_text: str) -> List[str]:
        """Find explicit references to other rules in the text."""
        references = []
        for pattern in self.reference_patterns:
            matches = re.finditer(pattern, rule_text, re.IGNORECASE)
            references.extend([m.group(1) for m in matches])
        return references
    
    def detect_semantic_similarity(
        self,
        rule1: Rule,
        rule2: Rule,
        threshold: float = 0.7
    ) -> bool:
        """Detect if two rules are semantically related."""
        emb1 = self.embedding_model.encode(rule1.content)
        emb2 = self.embedding_model.encode(rule2.content)
        
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return similarity > threshold
    
    def identify_relationship_type(self, rule1: Rule, rule2: Rule) -> str:
        """Identify the type of relationship between two rules."""
        # Check for explicit references
        if any(ref in rule2.references for ref in [rule1.section]):
            return "prerequisite"
        
        # Check for modifications or exceptions
        if any(pattern in rule2.content.lower() for pattern in self.modification_patterns):
            return "modification"
        
        # Check for shared systems
        if set(rule1.systems) & set(rule2.systems):
            return "related_system"
        
        return "semantic_similarity"

class QAGenerator:
    """Generates question-answer pairs from related rules."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        max_length: int = 512
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.max_length = max_length
    
    def generate_qa_templates(self, relationship_type: str) -> List[str]:
        """Generate appropriate question templates based on relationship type."""
        templates = {
            "prerequisite": [
                "What requirements must be met before {action}?",
                "What rules apply when a character wants to {action}?",
                "Under what conditions can a character {action}?"
            ],
            "modification": [
                "How does {system} change when {condition}?",
                "What special rules apply to {action} in {condition}?",
                "What exceptions exist for {action}?"
            ],
            "related_system": [
                "How do {system1} and {system2} interact?",
                "What happens when combining {action1} with {action2}?",
                "How does {system} affect {action}?"
            ]
        }
        return templates.get(relationship_type, [
            "What are the rules for {action}?",
            "How does {system} work?",
            "What happens when {action}?"
        ])
    
    def extract_key_elements(self, rule: Rule) -> Dict[str, str]:
        """Extract key elements from a rule for template filling."""
        # This would use NLP to identify actions, systems, and conditions
        # For now, we'll use a simple keyword-based approach
        elements = {
            "action": "",
            "system": "",
            "condition": ""
        }
        
        # Simple extraction based on verbs and keywords
        # In practice, you'd want to use proper NLP here
        words = rule.content.split()
        verbs = []  # You'd use a POS tagger in practice
        
        elements["system"] = rule.systems[0] if rule.systems else ""
        elements["action"] = verbs[0] if verbs else ""
        
        return elements
    
    def generate_question(
        self,
        rule_pair: RulePair,
        template: str
    ) -> str:
        """Generate a specific question using a template."""
        elements1 = self.extract_key_elements(rule_pair.rule1)
        elements2 = self.extract_key_elements(rule_pair.rule2)
        
        # Combine elements from both rules
        elements = {
            "action": elements1["action"],
            "system": elements1["system"],
            "condition": elements2["condition"],
            "action1": elements1["action"],
            "action2": elements2["action"],
            "system1": elements1["system"],
            "system2": elements2["system"]
        }
        
        # Fill template with extracted elements
        return template.format(**elements)
    
    def generate_answer(
        self,
        rule_pair: RulePair,
        question: str
    ) -> str:
        """Generate an answer using the LLM."""
        prompt = f"""Given these two related rules:

Rule 1: {rule_pair.rule1.content}

Rule 2: {rule_pair.rule2.content}

Question: {question}

Please provide a clear and concise answer that explains how these rules work together:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class QAPairGenerator:
    """Main class for generating QA pairs from rules."""
    
    def __init__(self):
        self.relationship_detector = RuleRelationshipDetector()
        self.qa_generator = QAGenerator()
    
    def find_related_rules(
        self,
        rules: List[Rule],
        min_similarity: float = 0.7
    ) -> List[RulePair]:
        """Find related rules using various methods."""
        rule_pairs = []
        
        for i, rule1 in enumerate(rules):
            for j, rule2 in enumerate(rules[i+1:], i+1):
                # Check for relationships
                if (self.relationship_detector.detect_semantic_similarity(rule1, rule2, min_similarity) or
                    rule2.section in rule1.references or
                    rule1.section in rule2.references or
                    set(rule1.systems) & set(rule2.systems)):
                    
                    relationship_type = self.relationship_detector.identify_relationship_type(
                        rule1, rule2
                    )
                    
                    rule_pairs.append(RulePair(rule1, rule2, relationship_type))
        
        return rule_pairs
    
    def generate_qa_pairs(
        self,
        rule_pairs: List[RulePair],
        questions_per_pair: int = 2
    ) -> List[QAPair]:
        """Generate QA pairs from related rules."""
        qa_pairs = []
        
        for rule_pair in rule_pairs:
            # Get appropriate templates
            templates = self.qa_generator.generate_qa_templates(
                rule_pair.relationship_type
            )
            
            # Generate questions and answers
            for template in templates[:questions_per_pair]:
                question = self.qa_generator.generate_question(rule_pair, template)
                answer = self.qa_generator.generate_answer(rule_pair, question)
                
                qa_pairs.append(QAPair(
                    question=question,
                    answer=answer,
                    source_rules=[rule_pair.rule1, rule_pair.rule2],
                    context=f"Rules {rule_pair.rule1.section} and {rule_pair.rule2.section}"
                ))
        
        return qa_pairs

# Example usage
def main():
    # Example rules
    rules = [
        Rule(
            content="A character can make a melee attack against any adjacent enemy.",
            section="1.1",
            references=[],
            systems=["combat"]
        ),
        Rule(
            content="When making a melee attack, add your Strength modifier to the damage.",
            section="1.2",
            references=["1.1"],
            systems=["combat"]
        ),
        # Add more rules...
    ]
    
    # Initialize generator
    generator = QAPairGenerator()
    
    # Find related rules
    rule_pairs = generator.find_related_rules(rules)
    
    # Generate QA pairs
    qa_pairs = generator.generate_qa_pairs(rule_pairs)
    
    # Print results
    for qa in qa_pairs:
        print(f"\nQuestion: {qa.question}")
        print(f"Answer: {qa.answer}")
        print(f"Context: {qa.context}")
        print("-" * 80)

if __name__ == "__main__":
    main()