#!/usr/bin/env python3

"""
Comparison framework for RPG Rules RAG systems.
This script provides unified interfaces to compare three different approaches:
1. Naive RAG (baseline)
2. NLP-enhanced Structure-aware RAG
3. Neural Network Augmented RAG
"""

import os
import argparse
import json
import time
import importlib
from typing import Dict, List, Any, Optional, Tuple

# Import the different RAG implementations
# Using dynamic imports to allow for modular testing
def import_rag_systems():
    """Import all three RAG system implementations."""
    systems = {}
    
    # Import the naive RAG implementation
    try:
        from naive_rag import NaiveRAG
        systems['naive'] = NaiveRAG
    except ImportError:
        print("Warning: NaiveRAG not found, some comparisons will be unavailable")
    
    # Import the structure-aware RAG implementation
    try:
        from rules_rag import RulesRAG
        systems['structure'] = RulesRAG
    except ImportError:
        print("Warning: Structure-aware RulesRAG not found, some comparisons will be unavailable")
    
    # Import the NN-augmented RAG implementation
    try:
        from nn_augmented_rag import NNAugmentedRAG
        systems['neural'] = NNAugmentedRAG
    except ImportError:
        print("Warning: NNAugmentedRAG not found, some comparisons will be unavailable")
    
    return systems

class RAGComparison:
    """Framework for comparing different RAG approaches on RPG rules."""
    
    def __init__(self, config_path: str):
        """Initialize the comparison framework with configuration."""
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Import RAG systems
        self.rag_systems = import_rag_systems()
        
        # Initialize RAG instances
        self.rag_instances = {}
        for system_name, system_class in self.rag_systems.items():
            try:
                # Different systems might have different initialization parameters
                if system_name == 'naive':
                    self.rag_instances[system_name] = system_class(
                        self.config['paths']['text_chunks_path']
                    )
                elif system_name == 'structure':
                    self.rag_instances[system_name] = system_class(
                        self.config['paths']['processed_srd_path']
                    )
                elif system_name == 'neural':
                    self.rag_instances[system_name] = system_class(
                        self.config['paths']['processed_srd_path'],
                        self.config['paths'].get('embeddings_path')
                    )
            except Exception as e:
                print(f"Error initializing {system_name} RAG: {e}")
    
    def run_comparison(self, query: str, max_rules: int = 10) -> Dict[str, Any]:
        """Run a comparison of all RAG systems on a single query."""
        results = {}
        timings = {}
        
        for system_name, rag_instance in self.rag_instances.items():
            print(f"Running query with {system_name} RAG...")
            start_time = time.time()
            
            try:
                # Different systems might have different query interfaces
                if system_name == 'naive':
                    system_result = rag_instance.query(query, top_k=max_rules)
                else:
                    system_result = rag_instance.query(query, max_rules=max_rules)
                
                query_time = time.time() - start_time
                timings[system_name] = query_time
                results[system_name] = system_result
                print(f"  - {system_name} retrieved {len(system_result.get('rules', []))} rules in {query_time:.2f} seconds")
            
            except Exception as e:
                print(f"Error querying {system_name} RAG: {e}")
                results[system_name] = {"error": str(e)}
        
        # Combine all results
        comparison = {
            'query': query,
            'results': results,
            'timings': timings
        }
        
        return comparison
    
    def batch_comparison(self, queries_path: str, output_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Run comparison on a batch of queries."""
        # Load queries
        with open(queries_path, 'r', encoding='utf-8') as f:
            queries = json.load(f)
        
        results = []
        for i, query_item in enumerate(queries):
            query = query_item['query']
            print(f"\n[{i+1}/{len(queries)}] Comparing RAG systems on: {query}")
            
            # Run comparison
            comparison = self.run_comparison(query, max_rules=query_item.get('max_rules', 10))
            results.append(comparison)
        
        # Save results if output path is provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"\nBatch comparison results saved to {output_path}")
        
        return results
    
    def analyze_results(self, comparison_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze comparison results and provide metrics."""
        if not comparison_results:
            return {"error": "No comparison results to analyze"}
        
        # Initialize metrics
        metrics = {
            'average_timings': {},
            'average_rule_counts': {},
            'result_similarities': {},
            'system_unique_rules': {}
        }
        
        # Calculate average timings and rule counts
        for system_name in self.rag_instances.keys():
            system_timings = []
            system_rule_counts = []
            
            for result in comparison_results:
                if system_name in result.get('timings', {}):
                    system_timings.append(result['timings'][system_name])
                
                if system_name in result.get('results', {}):
                    system_result = result['results'][system_name]
                    if 'rules' in system_result:
                        system_rule_counts.append(len(system_result['rules']))
            
            if system_timings:
                metrics['average_timings'][system_name] = sum(system_timings) / len(system_timings)
            
            if system_rule_counts:
                metrics['average_rule_counts'][system_name] = sum(system_rule_counts) / len(system_rule_counts)
        
        # Calculate result similarities and unique rules
        for system1 in self.rag_instances.keys():
            for system2 in self.rag_instances.keys():
                if system1 >= system2:  # Only calculate for unique pairs
                    continue
                
                pair_name = f"{system1}_vs_{system2}"
                similarity_scores = []
                
                for result in comparison_results:
                    if system1 in result.get('results', {}) and system2 in result.get('results', {}):
                        system1_rules = result['results'][system1].get('rules', [])
                        system2_rules = result['results'][system2].get('rules', [])
                        
                        # Get rule IDs
                        system1_rule_ids = set(rule.get('id') for rule in system1_rules if 'id' in rule)
                        system2_rule_ids = set(rule.get('id') for rule in system2_rules if 'id' in rule)
                        
                        # Calculate Jaccard similarity
                        union_size = len(system1_rule_ids.union(system2_rule_ids))
                        if union_size > 0:
                            intersection_size = len(system1_rule_ids.intersection(system2_rule_ids))
                            similarity = intersection_size / union_size
                            similarity_scores.append(similarity)
                
                if similarity_scores:
                    metrics['result_similarities'][pair_name] = sum(similarity_scores) / len(similarity_scores)
        
        # Calculate unique rules for each system
        for system_name in self.rag_instances.keys():
            unique_rule_counts = []
            
            for result in comparison_results:
                if system_name in result.get('results', {}):
                    this_system_rules = result['results'][system_name].get('rules', [])
                    this_system_rule_ids = set(rule.get('id') for rule in this_system_rules if 'id' in rule)
                    
                    # Count rules unique to this system
                    unique_rules = this_system_rule_ids.copy()
                    for other_system, other_result in result.get('results', {}).items():
                        if other_system != system_name:
                            other_rule_ids = set(rule.get('id') for rule in other_result.get('rules', []) if 'id' in rule)
                            unique_rules -= other_rule_ids
                    
                    unique_rule_counts.append(len(unique_rules))
            
            if unique_rule_counts:
                metrics['system_unique_rules'][system_name] = sum(unique_rule_counts) / len(unique_rule_counts)
        
        return metrics

def main():
    """Main entry point for comparison framework."""
    parser = argparse.ArgumentParser(description='RPG Rules RAG Comparison Framework')
    
    # Define subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Parser for the 'compare' command
    compare_parser = subparsers.add_parser('compare', help='Compare RAG systems on a single query')
    compare_parser.add_argument('--config', '-c', required=True, help='Path to configuration file')
    compare_parser.add_argument('--query', '-q', required=True, help='Query text')
    compare_parser.add_argument('--max-rules', '-m', type=int, default=10, help='Maximum number of rules to return')
    compare_parser.add_argument('--output', '-o', help='Output path for comparison results')
    
    # Parser for the 'batch' command
    batch_parser = subparsers.add_parser('batch', help='Run batch comparison on multiple queries')
    batch_parser.add_argument('--config', '-c', required=True, help='Path to configuration file')
    batch_parser.add_argument('--queries', '-q', required=True, help='Path to queries JSON file')
    batch_parser.add_argument('--output', '-o', help='Output path for batch comparison results')
    
    # Parser for the 'analyze' command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze comparison results')
    analyze_parser.add_argument('--results', '-r', required=True, help='Path to comparison results JSON file')
    analyze_parser.add_argument('--output', '-o', help='Output path for analysis results')
    
    # Parser for the 'serve' command
    serve_parser = subparsers.add_parser('serve', help='Start the comparison web interface')
    serve_parser.add_argument('--config', '-c', required=True, help='Path to configuration file')
    serve_parser.add_argument('--port', '-p', type=int, default=5000, help='Port to run the server on')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command == 'compare':
        framework = RAGComparison(args.config)
        result = framework.run_comparison(args.query, args.max_rules)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            print(f"Comparison results saved to {args.output}")
        else:
            print(json.dumps(result, indent=2))
    
    elif args.command == 'batch':
        framework = RAGComparison(args.config)
        framework.batch_comparison(args.queries, args.output)
    
    elif args.command == 'analyze':
        # Load comparison results
        with open(args.results, 'r', encoding='utf-8') as f:
            comparison_results = json.load(f)
        
        # Create framework just for analysis
        framework = RAGComparison(None)  # We don't need config for analysis
        framework.rag_instances = {}  # Empty instances, we just need the class names
        for system_name in comparison_results[0].get('results', {}).keys():
            framework.rag_instances[system_name] = None
        
        # Run analysis
        metrics = framework.analyze_results(comparison_results)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)
            print(f"Analysis results saved to {args.output}")
        else:
            print(json.dumps(metrics, indent=2))
    
    elif args.command == 'serve':
        # Import and run the web interface
        from comparison_web_interface import app
        
        # Set environment variable for the config path
        os.environ['CONFIG_PATH'] = args.config
        
        print(f"Starting comparison web interface on port {args.port}...")
        app.run(debug=True, host='0.0.0.0', port=args.port)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
