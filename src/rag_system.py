import json
import numpy as np
from sentence_transformers import SentenceTransformer
import networkx as nx
from typing import List, Dict, Set, Tuple, Any
import spacy

class RulesRAG:
    def __init__(self, processed_srd_path: str):
        """Initialize the Rules RAG system with processed SRD data."""
        # Load the processed SRD
        with open(processed_srd_path, 'r', encoding='utf-8') as f:
            self.srd_data = json.load(f)
        
        # Extract components
        self.rules = self.srd_data['rules']
        self.terms = self.srd_data['terms']
        self.relationships = self.srd_data['relationships']
        self.graph_data = self.srd_data['graph']
        
        # Initialize the embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create rule embeddings
        self.rule_embeddings = self._create_rule_embeddings()
        
        # Create a networkx graph for rule dependencies
        self.rule_graph = self._build_networkx_graph()
        
        # Load spaCy for query processing
        self.nlp = spacy.load("en_core_web_sm")
    
    def _create_rule_embeddings(self) -> Dict[str, np.ndarray]:
        """Create embeddings for all rules."""
        embeddings = {}
        
        print("Creating rule embeddings...")
        for i, rule in enumerate(self.rules):
            print(f"Processing rule {i+1}/{len(self.rules)}")
            # Create embedding from title and text
            text_to_embed = f"{rule['title']}. {rule['text']}"
            embedding = self.model.encode(text_to_embed)
            embeddings[rule['id']] = embedding
        
        return embeddings
    
    def _build_networkx_graph(self) -> nx.DiGraph:
        """Build a NetworkX directed graph from the relationship data."""
        G = nx.DiGraph()
        
        # Add nodes
        for node in self.graph_data['nodes']:
            G.add_node(node['id'], 
                       label=node['label'], 
                       type=node['type'],
                       scope=node.get('scope', 'UNKNOWN'),
                       complexity=node.get('complexity', 'MEDIUM'))
        
        # Add edges
        for edge in self.graph_data['edges']:
            G.add_edge(edge['source'], edge['target'], 
                      type=edge['type'],
                      term=edge.get('term', None))
        
        return G
    
    def _extract_query_features(self, query: str) -> Dict[str, Any]:
        """Extract relevant features from the query."""
        doc = self.nlp(query)
        
        features = {
            'detected_terms': [],
            'detected_actions': [],
            'detected_classes': [],
            'detected_races': [],
            'likely_scope': 'UNKNOWN'
        }
        
        # Check for known terms
        for term in self.terms.keys():
            if term.lower() in query.lower():
                features['detected_terms'].append(term)
        
        # Check for common action keywords
        action_keywords = ['attack', 'cast', 'move', 'dash', 'dodge', 'hide', 
                          'disengage', 'help', 'ready', 'search', 'use']
        for keyword in action_keywords:
            if keyword.lower() in query.lower():
                features['detected_actions'].append(keyword)
        
        # Check for class references
        classes = ['barbarian', 'bard', 'cleric', 'druid', 'fighter', 'monk', 
                 'paladin', 'ranger', 'rogue', 'sorcerer', 'warlock', 'wizard']
        for class_name in classes:
            if class_name.lower() in query.lower():
                features['detected_classes'].append(class_name)
        
        # Check for race references
        races = ['dwarf', 'elf', 'halfling', 'human', 'dragonborn', 
               'gnome', 'half-elf', 'half-orc', 'tiefling']
        for race_name in races:
            if race_name.lower() in query.lower():
                features['detected_races'].append(race_name)
        
        # Determine likely scope
        scope_keywords = {
            'COMBAT': ['attack', 'damage', 'hit', 'initiative', 'armor', 'hp', 
                      'round', 'turn', 'action'],
            'CHARACTER_CREATION': ['create', 'character', 'stats', 'abilities', 
                                 'skills', 'proficiency'],
            'SPELLCASTING': ['spell', 'cast', 'magic', 'scroll', 'component', 
                           'concentration'],
            'EQUIPMENT': ['weapon', 'armor', 'item', 'gear', 'gold', 'equipment'],
            'EXPLORATION': ['travel', 'rest', 'exhaustion', 'environment', 'climbing'],
            'SOCIAL': ['persuade', 'deceive', 'intimidate', 'interaction', 'charisma']
        }
        
        scope_scores = {}
        for scope, keywords in scope_keywords.items():
            score = sum(1 for keyword in keywords if keyword.lower() in query.lower())
            scope_scores[scope] = score
        
        if max(scope_scores.values(), default=0) > 0:
            features['likely_scope'] = max(scope_scores.items(), key=lambda x: x[1])[0]
        
        return features
    
    def _retrieve_candidate_rules(self, query: str, top_k: int = 10) -> List[Dict]:
        """Retrieve initial candidate rules using embedding similarity."""
        # Create query embedding
        query_embedding = self.model.encode(query)
        
        # Calculate similarity scores
        similarity_scores = {}
        for rule_id, rule_embedding in self.rule_embeddings.items():
            similarity = np.dot(query_embedding, rule_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(rule_embedding)
            )
            similarity_scores[rule_id] = similarity
        
        # Get top-k rule IDs by similarity
        top_rule_ids = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Get the actual rule objects
        top_rules = []
        for rule_id, score in top_rule_ids:
            rule = next((r for r in self.rules if r['id'] == rule_id), None)
            if rule:
                rule_copy = rule.copy()
                rule_copy['similarity_score'] = float(score)
                top_rules.append(rule_copy)
        
        return top_rules
    
    def _retrieve_dependent_rules(self, rule_ids: List[str], max_depth: int = 2) -> Set[str]:
      """Find all rules that these rules depend on (backward dependencies)."""
      dependent_rule_ids = set()
    
        # For each rule ID
      for rule_id in rule_ids:
        # Add rules that are reachable from this rule
        for target in self.rule_graph.nodes():
            if target != rule_id:  # Don't find paths to self
                try:
                    # Try to find paths from this rule to other rules
                    paths = nx.all_simple_paths(
                        self.rule_graph, 
                        source=rule_id, 
                        target=target,
                        cutoff=max_depth
                    )
                    # Add all nodes in these paths to dependent rules
                    for path in paths:
                        dependent_rule_ids.update(path)
                except nx.NetworkXNoPath:
                    # No path exists
                    continue
    
      return dependent_rule_ids
    
    def _retrieve_prerequisite_rules(self, rule_ids: List[str], max_depth: int = 2) -> Set[str]:
        """Find all rules that are prerequisites for understanding these rules."""
        prerequisite_rule_ids = set()
        
        for rule_id in rule_ids:
            # Find all nodes that point to this rule
            for node in self.rule_graph.nodes():
                if node != rule_id:  # Skip self-references
                    for path in nx.all_simple_paths(self.rule_graph, source=node, 
                                                  target=rule_id, cutoff=max_depth):
                        prerequisite_rule_ids.update(path)
        
        return prerequisite_rule_ids
    
    def _filter_by_relevance(self, rule_ids: Set[str], query_features: Dict) -> List[Dict]:
        """Filter rules by relevance to query features."""
        # If we have detected terms, prioritize rules that define or use those terms
        relevant_rules = []
        
        for rule_id in rule_ids:
            rule = next((r for r in self.rules if r['id'] == rule_id), None)
            if not rule:
                continue
            
            relevance_score = 0
            
            # If rule defines any of the detected terms, high relevance
            for term in query_features['detected_terms']:
                if term in self.terms and self.terms[term]['source_rule_id'] == rule_id:
                    relevance_score += 5
                elif term in rule.get('terms', []):
                    relevance_score += 2
            
            # If rule is in the likely scope, increase relevance
            if rule.get('scope') == query_features['likely_scope']:
                relevance_score += 3
            
            # If rule mentions any detected classes or races
            for class_name in query_features['detected_classes']:
                if class_name.lower() in rule['text'].lower():
                    relevance_score += 2
            
            for race_name in query_features['detected_races']:
                if race_name.lower() in rule['text'].lower():
                    relevance_score += 2
            
            # If rule mentions any detected actions
            for action in query_features['detected_actions']:
                if action.lower() in rule['text'].lower():
                    relevance_score += 1
            
            rule_copy = rule.copy()
            rule_copy['relevance_score'] = relevance_score
            relevant_rules.append(rule_copy)
        
        # Sort by relevance score
        return sorted(relevant_rules, key=lambda x: x['relevance_score'], reverse=True)
    
    def _minimize_rule_set(self, rules: List[Dict]) -> List[Dict]:
        """Remove redundant rules and sort in a logical order."""
        # If we have duplicate content, keep the most relevant one
        seen_content = set()
        minimal_rules = []
        
        # First, include CORE_RULE and DEFINITION types
        for rule in sorted(rules, key=lambda x: x.get('complexity', 'MEDIUM')):
            # Skip if we already have a similar rule
            rule_content_hash = hash(rule['text'][:100])  # Use first 100 chars as a proxy for content similarity
            if rule_content_hash in seen_content and rule['type'] not in ['CORE_RULE', 'DEFINITION']:
                continue
            
            seen_content.add(rule_content_hash)
            minimal_rules.append(rule)
        
        # Sort rules in a logical order
        sorted_rules = []
        
        # First definitions
        definitions = [r for r in minimal_rules if r['type'] == 'DEFINITION']
        sorted_rules.extend(sorted(definitions, key=lambda x: x.get('relevance_score', 0), reverse=True))
        
        # Then core rules
        core_rules = [r for r in minimal_rules if r['type'] == 'CORE_RULE']
        sorted_rules.extend(sorted(core_rules, key=lambda x: x.get('relevance_score', 0), reverse=True))
        
        # Then derived rules
        derived_rules = [r for r in minimal_rules if r['type'] == 'DERIVED_RULE']
        sorted_rules.extend(sorted(derived_rules, key=lambda x: x.get('relevance_score', 0), reverse=True))
        
        # Then exceptions
        exceptions = [r for r in minimal_rules if r['type'] == 'EXCEPTION']
        sorted_rules.extend(sorted(exceptions, key=lambda x: x.get('relevance_score', 0), reverse=True))
        
        # Then examples and tables
        examples_tables = [r for r in minimal_rules if r['type'] in ['EXAMPLE', 'TABLE']]
        sorted_rules.extend(sorted(examples_tables, key=lambda x: x.get('relevance_score', 0), reverse=True))
        
        return sorted_rules
    
    def query(self, query_text: str, max_rules: int = 10) -> Dict:
        """Process a query and return relevant rules."""
        print(f"Processing query: {query_text}")
        
        # Extract features from the query
        query_features = self._extract_query_features(query_text)
        print(f"Extracted features: {query_features}")
        
        # Get initial candidate rules
        candidate_rules = self._retrieve_candidate_rules(query_text, top_k=max_rules)
        candidate_rule_ids = [rule['id'] for rule in candidate_rules]
        print(f"Found {len(candidate_rules)} candidate rules")
        
        # Find dependent rules
        dependent_rule_ids = self._retrieve_dependent_rules(candidate_rule_ids)
        print(f"Found {len(dependent_rule_ids)} dependent rules")
        
        # Find prerequisite rules
        prerequisite_rule_ids = self._retrieve_prerequisite_rules(candidate_rule_ids)
        print(f"Found {len(prerequisite_rule_ids)} prerequisite rules")
        
        # Combine all unique rule IDs
        all_rule_ids = set(candidate_rule_ids) | dependent_rule_ids | prerequisite_rule_ids
        print(f"Total unique rules before filtering: {len(all_rule_ids)}")
        
        # Filter by relevance
        relevant_rules = self._filter_by_relevance(all_rule_ids, query_features)
        print(f"Rules after relevance filtering: {len(relevant_rules)}")
        
        # Minimize the rule set
        minimal_rules = self._minimize_rule_set(relevant_rules)
        if len(minimal_rules) > max_rules:
            minimal_rules = minimal_rules[:max_rules]
        print(f"Final rule count after minimization: {len(minimal_rules)}")
        
        # Prepare the response
        response = {
            'query': query_text,
            'features': query_features,
            'rules': minimal_rules,
            'rule_count': len(minimal_rules)
        }
        
        return response


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python rules_rag.py <path_to_processed_srd.json> [query]")
        sys.exit(1)
    
    srd_path = sys.argv[1]
    
    # Initialize the RAG system
    rag = RulesRAG(srd_path)
    
    # If a query was provided, process it
    if len(sys.argv) > 2:
        query = " ".join(sys.argv[2:])
        result = rag.query(query)
        
        print("\n=== Query Results ===")
        print(f"Query: {result['query']}")
        print(f"Features: {result['features']}")
        print(f"Found {result['rule_count']} relevant rules:")
        
        for i, rule in enumerate(result['rules'], 1):
            print(f"\n--- Rule {i} ---")
            print(f"Title: {rule['title']}")
            print(f"Path: {rule['path']}")
            print(f"Type: {rule['type']}")
            print(f"Score: {rule.get('relevance_score', 'N/A')}")
            
            # Print truncated text
            max_text_len = 200
            text = rule['text']
            if len(text) > max_text_len:
                text = text[:max_text_len] + "..."
            print(f"Text: {text}")
    else:
        # Interactive mode
        print("Enter queries (type 'exit' to quit):")
        while True:
            query = input("> ")
            if query.lower() in ['exit', 'quit', 'q']:
                break
            
            result = rag.query(query)
            
            print("\n=== Query Results ===")
            print(f"Found {result['rule_count']} relevant rules:")
            
            for i, rule in enumerate(result['rules'], 1):
                print(f"\n--- Rule {i} ---")
                print(f"Title: {rule['title']}")
                print(f"Path: {rule['path']}")
                print(f"Type: {rule['type']}")
                print(f"Score: {rule.get('relevance_score', 'N/A')}")
                
                # Print truncated text
                max_text_len = 200
                text = rule['text']
                if len(text) > max_text_len:
                    text = text[:max_text_len] + "..."
                print(f"Text: {text}")
