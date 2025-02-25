#!/usr/bin/env python3

import os
import argparse
import json
import time
from srd_processor import SRDProcessor
from rules_rag import RulesRAG
from rule_graph_visualizer import RuleGraphVisualizer

def main():
    """Main entry point for the RPG Rules RAG system."""
    parser = argparse.ArgumentParser(description='RPG Rules RAG System')
    
    # Define subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Parser for the 'process' command
    process_parser = subparsers.add_parser('process', help='Process SRD markdown file')
    process_parser.add_argument('--input', '-i', required=True, help='Path to SRD markdown file')
    process_parser.add_argument('--output', '-o', default='srd_processed.json', help='Output path for processed data')
    process_parser.add_argument('--visualize', '-v', action='store_true', help='Generate visualizations after processing')
    
    # Parser for the 'query' command
    query_parser = subparsers.add_parser('query', help='Query the processed SRD')
    query_parser.add_argument('--input', '-i', required=True, help='Path to processed SRD JSON file')
    query_parser.add_argument('--query', '-q', required=True, help='Query text')
    query_parser.add_argument('--max-rules', '-m', type=int, default=10, help='Maximum number of rules to return')
    query_parser.add_argument('--visualize', '-v', action='store_true', help='Generate a visualization of the query results')
    
    # Parser for the 'serve' command
    serve_parser = subparsers.add_parser('serve', help='Start the web interface')
    serve_parser.add_argument('--input', '-i', required=True, help='Path to processed SRD JSON file')
    serve_parser.add_argument('--port', '-p', type=int, default=5000, help='Port to run the server on')
    
    # Parser for the 'visualize' command
    viz_parser = subparsers.add_parser('visualize', help='Generate visualizations of the processed SRD')
    viz_parser.add_argument('--input', '-i', required=True, help='Path to processed SRD JSON file')
    viz_parser.add_argument('--output-dir', '-o', default='visualizations', help='Output directory for visualizations')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command == 'process':
        process_srd(args)
    elif args.command == 'query':
        query_srd(args)
    elif args.command == 'serve':
        serve_web_interface(args)
    elif args.command == 'visualize':
        visualize_srd(args)
    else:
        parser.print_help()

def process_srd(args):
    """Process an SRD markdown file."""
    print(f"Processing SRD file: {args.input}")
    start_time = time.time()
    
    # Process the SRD
    processor = SRDProcessor(args.input)
    processor.process()
    processor.save_output(args.output)
    
    processing_time = time.time() - start_time
    print(f"Processing completed in {processing_time:.2f} seconds")
    
    # Generate visualizations if requested
    if args.visualize:
        print("Generating visualizations...")
        viz_dir = os.path.splitext(args.output)[0] + "_viz"
        visualizer = RuleGraphVisualizer(args.output)
        
        os.makedirs(viz_dir, exist_ok=True)
        
        visualizer.plot_rule_distribution(os.path.join(viz_dir, "rule_distribution.png"))
        visualizer.create_interactive_graph(os.path.join(viz_dir, "full_rule_graph.html"))
        
        print(f"Visualizations saved to {viz_dir}")

def query_srd(args):
    """Query the processed SRD."""
    print(f"Querying SRD with: {args.query}")
    
    # Initialize RAG system
    rag = RulesRAG(args.input)
    
    # Process the query
    start_time = time.time()
    result = rag.query(args.query, max_rules=args.max_rules)
    query_time = time.time() - start_time
    
    # Print the results
    print(f"\n=== Query Results (found in {query_time:.2f} seconds) ===")
    print(f"Query: {result['query']}")
    print(f"Found {result['rule_count']} relevant rules:")
    
    for i, rule in enumerate(result['rules'], 1):
        print(f"\n--- Rule {i} ---")
        print(f"Title: {rule['title']}")
        print(f"Type: {rule['type']}")
        print(f"Relevance: {rule.get('relevance_score', 'N/A')}")
        
        # Print truncated text
        max_text_len = 200
        text = rule['text']
        if len(text) > max_text_len:
            text = text[:max_text_len] + "..."
        print(f"Text: {text}")
    
    # Generate visualization if requested
    if args.visualize and result['rules']:
        print("\nGenerating visualization of query results...")
        viz_path = f"query_result_{int(time.time())}.html"
        
        visualizer = RuleGraphVisualizer(args.input)
        rule_ids = [rule['id'] for rule in result['rules']]
        visualizer.visualize_query_subgraph(rule_ids, viz_path)
        
        print(f"Visualization saved to {viz_path}")

def serve_web_interface(args):
    """Start the web interface."""
    # Set environment variable for the SRD path
    os.environ['SRD_PATH'] = args.input
    
    print(f"Starting web interface on port {args.port}...")
    print(f"Using SRD data from: {args.input}")
    
    # Import Flask app and run it
    from web_interface import app
    app.run(debug=True, host='0.0.0.0', port=args.port)

def visualize_srd(args):
    """Generate visualizations of the processed SRD."""
    print(f"Generating visualizations for SRD: {args.input}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize visualizer
    visualizer = RuleGraphVisualizer(args.input)
    
    # Generate visualizations
    visualizer.plot_rule_distribution(os.path.join(args.output_dir, "rule_distribution.png"))
    visualizer.create_interactive_graph(os.path.join(args.output_dir, "full_rule_graph.html"))
    
    # Generate example subgraph
    print("Generating example subgraph...")
    example_rule_ids = [rule['id'] for rule in visualizer.rules[:5]]
    visualizer.visualize_query_subgraph(
        example_rule_ids, 
        os.path.join(args.output_dir, "example_subgraph.html")
    )
    
    # Generate example dependency tree
    print("Generating example dependency tree...")
    if visualizer.rules:
        first_rule_id = visualizer.rules[0]['id']
        visualizer.generate_dependency_tree(
            first_rule_id, 
            output_path=os.path.join(args.output_dir, "example_dependency_tree.html")
        )
    
    print(f"Visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
