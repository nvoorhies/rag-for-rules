#!/usr/bin/env python3

"""
NN-Augmented RAG with Embedding Cache - enhances the neural network augmented RAG system
by integrating the embedding cache for faster initialization and retrieval.
"""

import os
import json
import time
import logging
import pickle
from typing import Dict, List, Any, Optional, Set, Tuple
import numpy as np
import networkx as nx
import spacy
import sys
from pathlib import Path

#from embedding_cache import EmbeddingCache
from .rule_embedder import RuleSectionEmbedder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("nn_augmented_rag")

class CachedNNAugmentedRAG:
    """Neural Network Augmented RAG with efficient embedding caching."""
    
    def __init__(self, 
                processed_srd_path: str,
                embeddings_path: Optional[str] = None,
                model_name: str = "all-MiniLM-L6-v2",
                cache_dir: str = "embedding_cache",
                max_seq_length: Optional[int] = None):
        """
        Initialize the NN-augmented RAG system with embedding cache.
        
        Args:
            processed_srd_path: Path to processed SRD JSON file
            embeddings_path: Optional path to pre-computed embeddings pickle
            model_name: Name of the sentence transformer model to use
            cache_dir: Directory for the embedding cache
            max_seq_length: Maximum sequence length for the model
        """
        start_time = time.time()
        
        # Load the processed SRD
        with open(processed_srd_path, 'r', encoding='utf-8') as f:
            self.srd_data = json.load(f)
        
        # Extract components
        self.rules = self.srd_data['rules']
        self.terms = self.srd_data['terms']
        self.relationships = self.srd_data['relationships']
        self.graph_data = self.srd_data['graph']
        
        # Build rule graph
        self.rule_graph = self._build_networkx_graph()
        
        # Initialize the embedder with caching
        self.embedder = RuleSectionEmbedder(
            model_name=model_name,
            cache_dir=cache_dir,
            max_seq_length=max_seq_length
        )
        
        # Access the model and cache directly
        self.model = self.embedder.model
        self.cache = self.embedder.cache
        self.model_params = self.embedder.model_params
        
        # Load or generate augmented embeddings
        self.augmented_embeddings = self._load_or_generate_embeddings(embeddings_path)
        
        # Load spaCy for query processing
        self.nlp = spacy.load("en_core_web_sm")
        
        # Extract domain-specific information
        self.domain_info = self._extract_domain_info()
        
        logger.info(f"Initialized Cached NN-Augmented RAG with {len(self.rules)} rules in {time.time() - start_time:.2f} seconds")
    
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
    
    def _load_or_generate_embeddings(self, embeddings_path: Optional[str]) -> Dict[str, np.ndarray]:
        """Load embeddings from file or generate and cache them."""
        if embeddings_path and os.path.exists(embeddings_path):
            # Try to load pre-computed embeddings
            try:
                with open(embeddings_path, 'rb') as f:
                    embeddings = pickle.load(f)
                logger.info(f"Loaded {len(embeddings)} pre-computed embeddings from {embeddings_path}")
                return embeddings
            except Exception as e:
                logger.warning(f"Failed to load embeddings from {embeddings_path}: {e}")
                logger.info("Will generate embeddings from scratch")
        
        # Generate embeddings using the embedder
        logger.info("Generating augmented embeddings with caching...")
        return self._generate_augmented_rule_embeddings(embeddings_path)
    
    def _generate_augmented_rule_embeddings(self, output_path: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Generate embeddings for rules augmented with graph information."""
        augmented_embeddings = {}
        augmented_texts = []
        rule_ids = []
        
        for rule in self.rules:
            # Get direct neighbors in the graph
            rule_id = rule['id']
            rule_ids.append(rule_id)
            
            # Create augmented text
            augmented_text = self._create_augmented_text(rule)
            augmented_texts.append(augmented_text)
        
        # Compute embeddings in batch using cache
        content_hash_to_embedding = self.cache.compute_missing_embeddings(
            augmented_texts, 
            self.embedder.model_name, 
            self.model_params,
            self.embedder.embed_function
        )
        
        # Map rule IDs to embeddings
        for i, rule_id in enumerate(rule_ids):
            content_hash = self.cache._compute_content_hash(augmented_texts[i])
            if content_hash in content_hash_to_embedding:
                augmented_embeddings[rule_id] = content_hash_to_embedding[content_hash]
        
        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, 'wb') as f:
                pickle.dump(augmented_embeddings, f)
            logger.info(f"Saved augmented embeddings to {output_path}")
        
        return augmented_embeddings
    
    def _create_augmented_text(self, rule: Dict[str, Any]) -> str:
        """Create augmented text for a rule that includes graph information."""
        rule_id = rule['id']
        predecessors = list(self.rule_graph.predecessors(rule_id))
        successors = list(self.rule_graph.successors(rule_id))
        
        # Create augmented text that includes:
        # 1. The rule itself
        # 2. Key terms from connected rules
        # 3. Relationship information
        
        augmented_text = f"{rule['title']}. {rule['text']}"
        
        # Add information about prerequisite rules
        if predecessors:
            prereq_titles = [self.rule_graph.nodes[p].get('label', '') for p in predecessors]
            augmented_text += f" Requires understanding of: {', '.join(prereq_titles)}."
        
        # Add information about dependent rules
        if successors:
            dependent_titles = [self.rule_graph.nodes[s].get('label', '') for s in successors]
            augmented_text += f" Relevant to: {', '.join(dependent_titles)}."
        
        # Add term definitions if this rule defines any terms
        defined_terms = [t for t, info in self.terms.items() 
                        if info.get('source_rule_id') == rule_id]
        if defined_terms:
            augmented_text += f" Defines terms: {', '.join(defined_terms)}."
        
        # Add rule type and scope information
        augmented_text += f" Rule type: {rule.get('type', 'UNKNOWN')}. Scope: {rule.get('scope', 'GENERAL')}."
        
        return augmented_text
    
    def _extract_domain_info(self) -> Dict[str, Any]:
        """Extract domain-specific information from the SRD."""
        # Initialize containers
        domain_info = {
            'classes': set(),
            'races': set(),
            'actions': set(),
            'scopes': {},
            'term_frequencies': {}
        }
        
        # Identify game-specific terminology
        for rule in self.rules:
            text = rule['text'].lower()
            
            # Count term frequencies
            for term in self.terms:
                if term.lower() in text:
                    domain_info['term_frequencies'][term] = domain_info['term_frequencies'].get(term, 0) + 1
            
            # Extract scope information
            scope = rule.get('scope', 'UNKNOWN')
            if scope != 'UNKNOWN':
                if scope not in domain_info['scopes']:
                    domain_info['scopes'][scope] = {'keywords': set(), 'count': 0}
                domain_info['scopes'][scope]['count'] += 1
                
                # Extract keywords from this scope
                doc = self.nlp(rule['text'])
                for token in doc:
                    if token.pos_ in ['NOUN', 'VERB'] and len(token.text) > 3:
                        domain_info['scopes'][scope]['keywords'].add(token.text.lower())
        
        # Convert sets to lists for JSON serialization
        for scope in domain_info['scopes']:
            domain_info['scopes'][scope]['keywords'] = list(domain_info['scopes'][scope]['keywords'])
        
        # Identify most common nouns and verbs across all rules
        all_nouns = {}
        all_verbs = {}
        
        for rule in self.rules:
            doc = self.nlp(rule['text'])
            for token in doc:
                if token.pos_ == 'NOUN' and len(token.text) > 3:
                    all_nouns[token.text.lower()] = all_nouns.get(token.text.lower(), 0) + 1
                elif token.pos_ == 'VERB' and len(token.text) > 3:
                    all_verbs[token.text.lower()] = all_verbs.get(token.text.lower(), 0) + 1
        
        # Get top nouns and verbs
        top_nouns = sorted(all_nouns.items(), key=lambda x: x[1], reverse=True)[:50]
        top_verbs = sorted(all_verbs.items(), key=lambda x: x[1], reverse=True)[:30]
        
        domain_info['common_nouns'] = [n[0] for n in top_nouns]
        domain_info['common_verbs'] = [v[0] for v in top_verbs]
        
        # Try to identify classes and races from top nouns
        # This is a simple heuristic and would need refinement for different games
        potential_classes = [
            'barbarian', 'bard', 'cleric', 'druid', 'fighter', 'monk', 
            'paladin', 'ranger', 'rogue', 'sorcerer', 'warlock', 'wizard'
        ]
        
        potential_races = [
            'dwarf', 'elf', 'halfling', 'human', 'dragonborn', 
            'gnome', 'half-elf', 'half-orc', 'tiefling'
        ]
        
        # Find classes and races mentioned in the rules
        for rule in self.rules:
            text = rule['text'].lower()
            
            for class_name in potential_classes:
                if class_name in text:
                    domain_info['classes'].add(class_name)
            
            for race_name in potential_races:
                if race_name in text:
                    domain_info['races'].add(race_name)
        
        # Common actions in RPGs
        common_actions = [
            'attack', 'cast', 'use', 'move', 'dash', 'dodge', 'hide', 
            'disengage', 'help', 'ready', 'search', 'use'
        ]
        
        for action in common_actions:
            if action in domain_info['common_verbs']:
                domain_info['actions'].add(action)
        
        # Convert sets to lists for JSON serialization
        domain_info['classes'] = list(domain_info['classes'])
        domain_info['races'] = list(domain_info['races'])
        domain_info['actions'] = list(domain_info['actions'])
        
        return domain_info
    
    def _extract_query_features(self, query_text: str) -> Dict[str, Any]:
        """Extract relevant features from the query."""
        doc = self.nlp(query_text)
        
        # Initialize features
        features = {
            'detected_terms': [],
            'detected_actions': [],
            'detected_classes': [],
            'detected_races': [],
            'likely_scope': 'UNKNOWN',
            'question_type': 'GENERAL',
            'nouns': [],
            'verbs': []
        }
        
        # Extract nouns and verbs for general keyword matching
        for token in doc:
            if token.pos_ == 'NOUN' and len(token.text) > 3:
                features['nouns'].append(token.lemma_.lower())
            elif token.pos_ == 'VERB' and len(token.text) > 3:
                features['verbs'].append(token.lemma_.lower())
        
        # Check for known terms
        for term in self.terms.keys():
            if term.lower() in query_text.lower():
                features['detected_terms'].append(term)
        
        # Check for common action keywords
        for action in self.domain_info['actions']:
            if action.lower() in query_text.lower():
                features['detected_actions'].append(action)
        
        # Check for class references
        for class_name in self.domain_info['classes']:
            if class_name.lower() in query_text.lower():
                features['detected_classes'].append(class_name)
        
        # Check for race references
        for race_name in self.domain_info['races']:
            if race_name.lower() in query_text.lower():
                features['detected_races'].append(race_name)
        
        # Determine likely scope based on keywords
        scope_scores = {}
        for scope, info in self.domain_info['scopes'].items():
            score = 0
            for keyword in info['keywords']:
                if keyword in query_text.lower():
                    score += 1
            scope_scores[scope] = score
        
        if scope_scores and max(scope_scores.values()) > 0:
            features['likely_scope'] = max(scope_scores.items(), key=lambda x: x[1])[0]
        
        # Classify question type
        if query_text.lower().startswith('how'):
            features['question_type'] = 'HOW_TO'
        elif query_text.lower().startswith('what'):
            features['question_type'] = 'DEFINITION'
        elif query_text.lower().startswith('can'):
            features['question_type'] = 'CAPABILITY'
        elif '?' in query_text:
            features['question_type'] = 'QUESTION'
        
        return features
    
    def generate_query_embedding(self, query_text: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate an enhanced embedding for the query."""
        # Extract query features
        features = self._extract_query_features(query_text)
        
        # Augment query with detected features
        augmented_query = query_text
        
        if features['detected_terms']:
            terms_str = ', '.join(features['detected_terms'][:3])  # Limit to top 3 for brevity
            augmented_query += f" Terms: {terms_str}."
        
        if features['detected_actions']:
            actions_str = ', '.join(features['detected_actions'])
            augmented_query += f" Actions: {actions_str}."
        
        if features['likely_scope'] != 'UNKNOWN':
            augmented_query += f" Scope: {features['likely_scope']}."
        
        if features['detected_classes']:
            classes_str = ', '.join(features['detected_classes'])
            augmented_query += f" Classes: {classes_str}."
        
        if features['detected_races']:
            races_str = ', '.join(features['detected_races'])
            augmented_query += f" Races: {races_str}."
        
        # Generate embedding through the cache
        query_embedding = self.model.encode(
            augmented_query, 
            normalize_embeddings=self.model_params['normalize_embeddings']
        )
        
        return query_embedding, features
    
    def retrieve_candidate_rules(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve initial candidate rules using embedding similarity."""
        similarity_scores = {}
        
        # Calculate similarities with all augmented rule embeddings
        for rule_id, rule_embedding in self.augmented_embeddings.items():
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
    
    def create_complete_rule_set(self, candidate_rules: List[Dict[str, Any]], max_depth: int = 2) -> List[Dict[str, Any]]:
        """Create a complete set of rules by adding prerequisites."""
        complete_rule_ids = set()
        
        # Add all candidate rules
        for rule in candidate_rules:
            complete_rule_ids.add(rule['id'])
        
        # Find prerequisites for these rules (ancestors in the graph)
        for rule_id in list(complete_rule_ids):  # Use list to avoid modifying during iteration
            try:
                # Get ancestors up to a certain depth using BFS
                ancestors = set()
                to_visit = [(rule_id, 0)]  # (node, depth)
                visited = {rule_id}
                
                while to_visit:
                    node, depth = to_visit.pop(0)
                    if depth >= max_depth:
                        continue
                    
                    for pred in self.rule_graph.predecessors(node):
                        if pred not in visited:
                            visited.add(pred)
                            ancestors.add(pred)
                            to_visit.append((pred, depth + 1))
                
                complete_rule_ids.update(ancestors)
            except Exception as e:
                logger.error(f"Error finding prerequisites for rule {rule_id}: {e}")
        
        # Return complete rule set
        complete_rules = []
        for rule_id in complete_rule_ids:
            rule = next((r for r in self.rules if r['id'] == rule_id), None)
            if rule:
                complete_rules.append(rule.copy())
        
        return complete_rules
    
    def categorize_rules(self, rules: List[Dict[str, Any]], primary_rule_ids: Set[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize rules into primary, modifications, and prerequisites."""
        categorized = {
            'primary': [],
            'modifications': [],
            'prerequisites': []
        }
        
        for rule in rules:
            if rule['id'] in primary_rule_ids:
                categorized['primary'].append(rule)
            elif rule.get('type') == 'EXCEPTION':
                categorized['modifications'].append(rule)
            else:
                categorized['prerequisites'].append(rule)
        
        return categorized
    
    def _minimize_rule_set(self, rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove redundant rules and sort in a logical order."""
        # If we have duplicate content, keep the most relevant one
        seen_content = set()
        minimal_rules = []
        
        # Sort rules by type importance and then by relevance/similarity score
        ordered_rules = sorted(
            rules, 
            key=lambda x: (
                # Type importance (higher is more important)
                {'DEFINITION': 5, 'CORE_RULE': 4, 'DERIVED_RULE': 3, 'EXCEPTION': 2, 'EXAMPLE': 1, 'TABLE': 0}.get(x.get('type', 'UNKNOWN'), -1),
                # Then by similarity score
                x.get('similarity_score', 0)
            ),
            reverse=True
        )
        
        for rule in ordered_rules:
            # Skip if we already have a similar rule
            rule_content_hash = hash(rule['text'][:100])  # Use first 100 chars as a proxy for content similarity
            if rule_content_hash in seen_content and rule.get('type') not in ['CORE_RULE', 'DEFINITION']:
                continue
            
            seen_content.add(rule_content_hash)
            minimal_rules.append(rule)
        
        return minimal_rules
    
    def query(self, query_text: str, max_rules: int = 10) -> Dict[str, Any]:
        """Process a query and return relevant rules."""
        start_time = time.time()
        
        # Generate query embedding and extract features
        query_embedding, features = self.generate_query_embedding(query_text)
        
        # Get initial candidate rules
        candidate_rules = self.retrieve_candidate_rules(query_embedding, top_k=max_rules)
        primary_rule_ids = {rule['id'] for rule in candidate_rules[:3]}  # Top 3 as primary
        
        # Create complete rule set
        complete_rules = self.create_complete_rule_set(candidate_rules)
        
        # Minimize and organize rule set
        minimal_rules = self._minimize_rule_set(complete_rules)
        if len(minimal_rules) > max_rules:
            minimal_rules = minimal_rules[:max_rules]
        
        # Categorize rules
        categorized_rules = self.categorize_rules(minimal_rules, primary_rule_ids)
        
        # Prepare the response
        response = {
            'query': query_text,
            'features': features,
            'rules': minimal_rules,
            'categorized_rules': categorized_rules,
            'rule_count': len(minimal_rules),
            'query_time': time.time() - start_time
        }
        
        return response
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding cache."""
        return self.cache.get_cache_stats()

# For running as a script
def main():
    """Main entry point for command-line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='NN-Augmented RAG with Embedding Cache')
    parser.add_argument('--srd', '-s', required=True, help='Path to processed SRD JSON file')
    parser.add_argument('--embeddings', '-e', help='Path to save/load embeddings')
    parser.add_argument('--cache-dir', '-c', default='embedding_cache', help='Cache directory')
    parser.add_argument('--model', '-m', default='all-MiniLM-L6-v2', help='Embedding model name')
    parser.add_argument('--query', '-q', help='Query text')
    parser.add_argument('--max-rules', type=int, default=10, help='Maximum rules to return')
    parser.add_argument('--stats', action='store_true', help='Print cache statistics')
    
    args = parser.parse_args()
    
    # Initialize the RAG system
    rag = CachedNNAugmentedRAG(
        processed_srd_path=args.srd,
        embeddings_path=args.embeddings,
        model_name=args.model,
        cache_dir=args.cache_dir
    )
    
    # Print cache stats if requested
    if args.stats:
        stats = rag.get_cache_stats()
        print("\n=== Embedding Cache Statistics ===")
        print(f"Total Models: {stats['total_models']}")
        print(f"Total Embeddings: {stats['total_embeddings']}")
        print(f"Total Cache Size: {stats['cache_size']}")
        
        for model_name, model_stats in stats['models'].items():
            print(f"  - {model_name}: {model_stats['embedding_count']} embeddings, {model_stats['size']}")
    
    # If a query was provided, process it
    if args.query:
        result = rag.query(args.query, max_rules=args.max_rules)
        
        print("\n=== Query Results ===")
        print(f"Query: {result['query']}")
        print(f"Query time: {result['query_time']:.3f} seconds")
        
        if result['features']:
            print("\nDetected Features:")
            
            if result['features']['detected_terms']:
                print(f"  Terms: {', '.join(result['features']['detected_terms'])}")
            
            if result['features']['detected_actions']:
                print(f"  Actions: {', '.join(result['features']['detected_actions'])}")
            
            if result['features']['likely_scope'] != 'UNKNOWN':
                print(f"  Likely scope: {result['features']['likely_scope']}")
        
        print(f"\nFound {result['rule_count']} relevant rules:")
        
        categorized = result['categorized_rules']
        
        if categorized['primary']:
            print("\n--- Primary Rules ---")
            for rule in categorized['primary']:
                print(f"- {rule['title']} ({rule.get('type', 'UNKNOWN')})")
                print(f"  Score: {rule.get('similarity_score', 0.0):.4f}")
        
        if categorized['modifications']:
            print("\n--- Modifications ---")
            for rule in categorized['modifications']:
                print(f"- {rule['title']} ({rule.get('type', 'UNKNOWN')})")
        
        if categorized['prerequisites']:
            print("\n--- Prerequisites ---")
            for rule in categorized['prerequisites']:
                print(f"- {rule['title']} ({rule.get('type', 'UNKNOWN')})")
        
        # Print first rule details
        if result['rules']:
            first_rule = result['rules'][0]
            print(f"\nTop Rule: {first_rule['title']}")
            print(f"Type: {first_rule.get('type', 'UNKNOWN')}")
            print(f"Score: {first_rule.get('similarity_score', 0.0):.4f}")
            
            # Print truncated text
            max_text_len = 300
            text = first_rule['text']
            if len(text) > max_text_len:
                text = text[:max_text_len] + "..."
            print(f"Text: {text}")

if __name__ == "__main__":
    import pickle  # Import here for pickle.dump
    main()
