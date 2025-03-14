#!/usr/bin/env python3                                                                                                                   
                                                                                                                                          
"""                                                                                                                                      
Evaluate QA pairs using the cached_naive_rag.py system.                                                                                  
This script takes QA pairs generated by single_rule.py and evaluates them with the RAG system.                                           
"""                                                                                                                                      
                                                                                                                                          
import json                                                                                                                              
import os                                                                                                                                
import sys                                                                                                                               
import argparse                                                                                                                          
import tempfile                                                                                                                          
from pathlib import Path                                                                                                                 
import subprocess                                                                                                                        
from typing import List, Dict, Any, Optional                                                                                             
                                                                                                                                          
def extract_questions(qa_pairs_file: str) -> List[Dict[str, str]]:
    """Extract questions from QA pairs file."""
    with open(qa_pairs_file, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    
    questions = []
    for qa_pair in qa_pairs:
        if 'question' in qa_pair:
            questions.append({
                'id': qa_pair.get('rules', ['unknown'])[0],
                'question': qa_pair['question']
            })
    
    return questions

def write_questions_to_temp_file(questions: List[Dict[str, str]]) -> str:
    """Write questions to a temporary JSON file."""
    questions_json = [{'id': q['id'], 'question': q['question']} for q in questions]
    
    # Create a temporary file
    fd, temp_path = tempfile.mkstemp(suffix='.json', prefix='questions_')
    with os.fdopen(fd, 'w') as f:
        json.dump(questions_json, f, indent=2)
                                                                                                                                          
    return temp_path

def run_cached_naive_rag(questions_file: str, chunks_file: str, output_file: str,
                         top_k: int = 5, model: str = 'all-mpnet-base-v2',
                         cache_dir: str = 'embedding_cache', verbose: bool = False) -> str:
    """Run the cached_naive_rag.py script on the questions."""
    cmd = [
        sys.executable,
        "src/cached_naive_rag.py",
        "query",
        "--chunks", chunks_file,
        "--queries-file", questions_file,
        "--top-k", str(top_k),
        "--output", output_file,
        "--model", model,
        "--cache-dir", cache_dir
    ]
    
    if verbose:
        cmd.append("--verbose")
    
    print(f"Running command: {' '.join(cmd)}")
    # Don't capture output so it displays in real-time
    result = subprocess.run(cmd, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to run cached_naive_rag.py")
    
    return output_file

def run_hierarchical_naive_rag(questions_file: str, srd_file: str, output_file: str,
                              top_k: int = 5, model: str = 'all-mpnet-base-v2',
                              cache_dir: str = 'embedding_cache', verbose: bool = False) -> str:
    """Run the hierarchical_naive_rag.py script on the questions."""
    cmd = [
        sys.executable,
        "src/hierarchical_naive_rag.py",
        "--srd", srd_file,
        "--queries-file", questions_file,
        "--output", output_file,
        "--top-k", str(top_k),
        "--model", model,
        "--cache-dir", cache_dir
    ]
    
    if verbose:
        cmd.append("--verbose")
    
    print(f"Running command: {' '.join(cmd)}")
    # Don't capture output so it displays in real-time
    result = subprocess.run(cmd, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to run hierarchical_naive_rag.py")
    
    return output_file


def run_augmented_naive_rag(questions_file: str, srd_file: str, output_file: str,
                              top_k: int = 5, model: str = 'all-MiniLM-L6-v2',
                              cache_dir: str = 'embedding_cache', verbose: bool = False,
                              max_seq_length = 256, profile: bool = False) -> str:
    """Run the augmented_hierarchical_rag.py script on the questions."""
    cmd = [
        sys.executable,
        "src/augmented_hierarchical_rag.py",
        "--srd", srd_file,
        "--queries-file", questions_file,
        "--output", output_file,
        "--top-k", str(top_k),
        "--model", model,
        "--cache-dir", cache_dir,
        "--max-seq-length", str(max_seq_length),
    ]
    
    if verbose:
        cmd.append("--verbose")
        
    if profile:
        cmd.append("--profile")
    
    print(f"Running command: {' '.join(cmd)}")
    # Don't capture output so it displays in real-time
    result = subprocess.run(cmd, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to run augmented_hierarchical_rag.py")
    
    return output_file


def run_cached_nn_augmented_rag(questions_file: str, srd_file: str, output_file: str,
                               top_k: int = 5, model: str = 'all-MiniLM-L6-v2',
                               cache_dir: str = 'embedding_cache', verbose: bool = False) -> str:
    """Run the cached_nn_augmented_rag.py script on the questions."""
    cmd = [
        sys.executable,
        "src/cached_nn_augmented_rag.py",
        "batch",
        "--srd", srd_file,
        "--queries-file", questions_file,
        "--output", output_file,
        "--max-rules", str(top_k),
        "--model", model,
        "--cache-dir", cache_dir
    ]
    
    if verbose:
        cmd.append("--stats")
    
    print(f"Running command: {' '.join(cmd)}")
    # Don't capture output so it displays in real-time
    result = subprocess.run(cmd, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to run cached_nn_augmented_rag.py")
    
    return output_file

def run_reranker_hierarchical_rag(questions_file: str, srd_file: str, output_file: str,
                                 top_k: int = 5, model: str = 'all-mpnet-base-v2',
                                 reranker_model: str = 'mixedbread-ai/mxbai-rerank-xsmall-v1',
                                 cache_dir: str = 'embedding_cache', verbose: bool = False,
                                 parallel: int = 1, device: Optional[str] = None) -> str:
    """Run the reranker_hierarchical_rag.py script on the questions."""
    cmd = [
        sys.executable,
        "src/reranker_hierarchical_rag.py",
        "--srd", srd_file,
        "--queries-file", questions_file,
        "--output", output_file,
        "--top-k", str(top_k),
        "--model", model,
        "--reranker", reranker_model,
        "--cache-dir", cache_dir,
        "--parallel", str(parallel)
    ]
    
    if device:
        cmd.extend(["--device", device])
    
    if verbose:
        cmd.append("--verbose")
    
    print(f"Running command: {' '.join(cmd)}")
    # Don't capture output so it displays in real-time
    result = subprocess.run(cmd, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to run reranker_hierarchical_rag.py")
    
    return output_file

def run_augmented_reranker_rag(questions_file: str, srd_file: str, output_file: str,
                              top_k: int = 5, model: str = 'all-mpnet-base-v2',
                              reranker_model: str = 'mixedbread-ai/mxbai-rerank-xsmall-v1',
                              cache_dir: str = 'embedding_cache', verbose: bool = False,
                              parallel: int = 1, device: Optional[str] = None) -> str:
    """Run the augmented_reranker_rag.py script on the questions."""
    cmd = [
        sys.executable,
        "src/augmented_reranker_rag.py",
        "--srd", srd_file,
        "--queries-file", questions_file,
        "--output", output_file,
        "--top-k", str(top_k),
        "--model", model,
        "--reranker", reranker_model,
        "--cache-dir", cache_dir,
        "--parallel", str(parallel)
    ]
    
    if verbose:
        cmd.append("--verbose")
    
    print(f"Running command: {' '.join(cmd)}")
    # Don't capture output so it displays in real-time
    result = subprocess.run(cmd, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to run augmented_reranker_rag.py")
    
    return output_file
                                                                                                                                          
def evaluate_results(rag_results_file: str, qa_pairs_file: str, output_file: Optional[str] = None) -> Dict[str, Any]:
    """Evaluate the RAG results against the original QA pairs."""
    # Load RAG results
    with open(rag_results_file, 'r', encoding='utf-8') as f:
        rag_results = json.load(f)
    
    # Load original QA pairs
    with open(qa_pairs_file, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    
    # Create a lookup for QA pairs by question
    qa_lookup = {qa_pair['question']: qa_pair for qa_pair in qa_pairs}
    
    # Evaluate each result
    evaluations = []
    for result in rag_results:
        #print(f"Processing result: {result}")
        if 'question' not in result:
            print(f"Skipping result without question")
            continue
        query = result['question']
        if query in qa_lookup:
            original_qa = qa_lookup[query]
            
            # Get the rule ID from the original QA pair
            rule_ids = original_qa.get('rules', ['unknown'])
            rule_id = rule_ids[0] if isinstance(rule_ids, list) and rule_ids else 'unknown'
            
            # Check if the correct rule is in the retrieved rules
            retrieved_rule_ids = [" > ".join(rule.get('path', []) + [rule.get('title', '')]) for rule in result.get('rules', [])]
            correct_rule_found = any(rule_id in r_id for r_id in retrieved_rule_ids)
            
            # Find position of the correct rule if present
            position = -1
            for i, r_id in enumerate(retrieved_rule_ids):
                if rule_id in r_id:
                    position = i + 1  # 1-based position
                    break
            
            evaluation = {
                'question': query,
                'rule_id': rule_id,
                'correct_rule_found': correct_rule_found,
                'position': position,
                'retrieved_rules': retrieved_rule_ids[:5],  # Show top 5 for brevity
                'query_time': result.get('query_time', 0)
            }
            #print(f"Evaluation: {evaluation}")
            evaluations.append(evaluation)
    
    # Calculate overall metrics
    total = len(evaluations)
    correct = sum(1 for e in evaluations if e['correct_rule_found'])
    top_1 = sum(1 for e in evaluations if e['position'] == 1)
    top_3 = sum(1 for e in evaluations if 1 <= e['position'] <= 3)
    top_5 = sum(1 for e in evaluations if 1 <= e['position'] <= 5)
    
    metrics = {
        'total_questions': total,
        'correct_rule_found': correct,
        'accuracy': correct / total if total > 0 else 0,
        'top_1_accuracy': top_1 / total if total > 0 else 0,
        'top_3_accuracy': top_3 / total if total > 0 else 0,
        'top_5_accuracy': top_5 / total if total > 0 else 0,
        'avg_query_time': sum(e['query_time'] for e in evaluations) / total if total > 0 else 0
    }
    
    # Combine metrics and evaluations
    results = {
        'metrics': metrics,
        'evaluations': evaluations
    }
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"Evaluation results saved to {output_file}")
    
    return results
                                                                                                                                          
def main():
    parser = argparse.ArgumentParser(description='Evaluate QA pairs using different RAG systems')
    parser.add_argument('--qa-pairs', '-q', required=True, help='Path to QA pairs JSON file from single_rule.py')
    parser.add_argument('--chunks', '-c', help='Path to chunks JSON file for cached_naive_rag.py')
    parser.add_argument('--srd', '-s', help='Path to processed SRD JSON file for hierarchical and NN-augmented RAG')
    parser.add_argument('--output', '-o', default='evaluation_results.json', help='Path to save evaluation results')
    parser.add_argument('--top-k', '-k', type=int, default=5, help='Number of results to return from RAG')
    parser.add_argument('--model', '-m', default='all-mpnet-base-v2', help='Embedding model to use')
    parser.add_argument('--cache-dir', '-d', default='embedding_cache', help='Embedding cache directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--limit', '-l', type=int, help='Limit number of questions to evaluate')
    parser.add_argument('--system', choices=['naive', 'hierarchical', 'augmented', 'nn-augmented', 'reranker', 'augmented-reranker', 'all'], 
                       default='all', help='RAG system to evaluate')
    parser.add_argument('--reranker', default='mixedbread-ai/mxbai-rerank-xsmall-v1', 
                       help='Reranker model to use with reranker system')
    parser.add_argument('--profile', action='store_true', help='Enable profiling to identify performance bottlenecks')
    parser.add_argument('--parallel', '-p', type=int, default=4, 
                       help='Number of parallel processes to use for evaluation')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], 
                       help='Device to use for reranker models (default: cpu to avoid MPS issues)')
    
    args = parser.parse_args()
    
    # Extract questions from QA pairs
    print(f"Extracting questions from {args.qa_pairs}")
    questions = extract_questions(args.qa_pairs)
    
    # Limit number of questions if specified
    if args.limit and args.limit > 0:
        questions = questions[:args.limit]
        print(f"Limited to {args.limit} questions")
    
    print(f"Found {len(questions)} questions")
    
    # Write questions to temp file
    questions_file = write_questions_to_temp_file(questions)
    print(f"Wrote questions to temporary file: {questions_file}")
    
    # Validate required arguments based on selected system
    if (args.system in ['naive']) and not args.chunks:
        parser.error("--chunks is required when evaluating naive RAG")
    
    if (args.system in ['hierarchical', 'nn-augmented', 'all']) and not args.srd:
        parser.error("--srd is required when evaluating hierarchical or nn-augmented RAG")
    try:
        all_results = {}
        
        # Run naive RAG if selected
        if args.system in ['naive']:
            naive_results_file = tempfile.mktemp(suffix='.json', prefix='naive_rag_results_')
            
            print(f"Running cached_naive_rag.py...")
            run_cached_naive_rag(
                questions_file=questions_file,
                chunks_file=args.chunks,
                output_file=naive_results_file,
                top_k=args.top_k,
                model=args.model,
                cache_dir=args.cache_dir,
                verbose=args.verbose
            )
            
            print(f"Evaluating naive RAG results...")
            naive_results = evaluate_results(
                rag_results_file=naive_results_file,
                qa_pairs_file=args.qa_pairs
            )
            
            all_results['naive'] = naive_results
            
            # Print summary metrics
            metrics = naive_results['metrics']
            print("\n=== Naive RAG Evaluation Summary ===")
            print(f"Total Questions: {metrics['total_questions']}")
            print(f"Correct Rule Found: {metrics['correct_rule_found']} ({metrics['accuracy']:.2%})")
            print(f"Top-1 Accuracy: {metrics['top_1_accuracy']:.2%}")
            print(f"Top-3 Accuracy: {metrics['top_3_accuracy']:.2%}")
            print(f"Top-5 Accuracy: {metrics['top_5_accuracy']:.2%}")
            print(f"Average Query Time: {metrics['avg_query_time']:.4f} seconds")
            
            # Clean up
            if os.path.exists(naive_results_file):
                os.remove(naive_results_file)
        
        # Run hierarchical RAG if selected
        if args.system in ['hierarchical', 'all']:
            hierarchical_results_file = tempfile.mktemp(suffix='.json', prefix='hierarchical_rag_results_')
            
            print(f"Running hierarchical_naive_rag.py...")
            run_hierarchical_naive_rag(
                questions_file=questions_file,
                srd_file=args.srd,
                output_file=hierarchical_results_file,
                top_k=args.top_k,
                model=args.model,
                cache_dir=args.cache_dir,
                verbose=args.verbose
            )
            
            print(f"Evaluating hierarchical RAG results...")
            hierarchical_results = evaluate_results(
                rag_results_file=hierarchical_results_file,
                qa_pairs_file=args.qa_pairs
            )
            
            all_results['hierarchical'] = hierarchical_results
            
            # Print summary metrics
            metrics = hierarchical_results['metrics']
            print("\n=== Hierarchical RAG Evaluation Summary ===")
            print(f"Total Questions: {metrics['total_questions']}")
            print(f"Correct Rule Found: {metrics['correct_rule_found']} ({metrics['accuracy']:.2%})")
            print(f"Top-1 Accuracy: {metrics['top_1_accuracy']:.2%}")
            print(f"Top-3 Accuracy: {metrics['top_3_accuracy']:.2%}")
            print(f"Top-5 Accuracy: {metrics['top_5_accuracy']:.2%}")
            print(f"Average Query Time: {metrics['avg_query_time']:.4f} seconds")
            
            # Clean up
            if os.path.exists(hierarchical_results_file):
                os.remove(hierarchical_results_file)
        
        # Run augmented RAG if selected
        if args.system in ['augmented', 'all']:
            augmented_results_file = tempfile.mktemp(suffix='.json', prefix='augmented_rag_results_')
            
            print(f"Running augmented_hierarchical_rag.py...")
            run_augmented_naive_rag(
                questions_file=questions_file,
                srd_file=args.srd,
                output_file=augmented_results_file,
                top_k=args.top_k,
                model=args.model,
                cache_dir=args.cache_dir,
                verbose=args.verbose,
                profile=args.profile
            )
            
            print(f"Evaluating augmented RAG results...")
            augmented_results = evaluate_results(
                rag_results_file=augmented_results_file,
                qa_pairs_file=args.qa_pairs
            )
            
            all_results['augmented'] = augmented_results
            
            # Print summary metrics
            metrics = augmented_results['metrics']
            print("\n=== Augmented RAG Evaluation Summary ===")
            print(f"Total Questions: {metrics['total_questions']}")
            print(f"Correct Rule Found: {metrics['correct_rule_found']} ({metrics['accuracy']:.2%})")
            print(f"Top-1 Accuracy: {metrics['top_1_accuracy']:.2%}")
            print(f"Top-3 Accuracy: {metrics['top_3_accuracy']:.2%}")
            print(f"Top-5 Accuracy: {metrics['top_5_accuracy']:.2%}")
            print(f"Average Query Time: {metrics['avg_query_time']:.4f} seconds")
            
            # Clean up
            if os.path.exists(augmented_results_file):
                os.remove(augmented_results_file)

        # Run NN-augmented RAG if selected

        if args.system in ['nn-augmented']:
            nn_augmented_results_file = tempfile.mktemp(suffix='.json', prefix='nn_augmented_rag_results_')
            
            print(f"Running cached_nn_augmented_rag.py...")
            run_cached_nn_augmented_rag(
                questions_file=questions_file,
                srd_file=args.srd,
                output_file=nn_augmented_results_file,
                top_k=args.top_k,
                model=args.model,
                cache_dir=args.cache_dir,
                verbose=args.verbose
            )
            
            print(f"Evaluating NN-augmented RAG results...")
            nn_augmented_results = evaluate_results(
                rag_results_file=nn_augmented_results_file,
                qa_pairs_file=args.qa_pairs
            )
            
            all_results['nn_augmented'] = nn_augmented_results
            
            # Print summary metrics
            metrics = nn_augmented_results['metrics']
            print("\n=== NN-Augmented RAG Evaluation Summary ===")
            print(f"Total Questions: {metrics['total_questions']}")
            print(f"Correct Rule Found: {metrics['correct_rule_found']} ({metrics['accuracy']:.2%})")
            print(f"Top-1 Accuracy: {metrics['top_1_accuracy']:.2%}")
            print(f"Top-3 Accuracy: {metrics['top_3_accuracy']:.2%}")
            print(f"Top-5 Accuracy: {metrics['top_5_accuracy']:.2%}")
            print(f"Average Query Time: {metrics['avg_query_time']:.4f} seconds")
            
            # Clean up
            if os.path.exists(nn_augmented_results_file):
                os.remove(nn_augmented_results_file)
                
        # Run Reranker Hierarchical RAG if selected
        if args.system in ['reranker', 'all']:
            reranker_results_file = tempfile.mktemp(suffix='.json', prefix='reranker_rag_results_')
            
            print(f"Running reranker_hierarchical_rag.py...")
            run_reranker_hierarchical_rag(
                questions_file=questions_file,
                srd_file=args.srd,
                output_file=reranker_results_file,
                top_k=args.top_k,
                model=args.model,
                reranker_model=args.reranker,
                cache_dir=args.cache_dir,
                verbose=args.verbose,
                parallel=args.parallel,
                device=args.device
            )
            
            print(f"Evaluating Reranker Hierarchical RAG results...")
            reranker_results = evaluate_results(
                rag_results_file=reranker_results_file,
                qa_pairs_file=args.qa_pairs
            )
            
            all_results['reranker'] = reranker_results
            
            # Print summary metrics
            metrics = reranker_results['metrics']
            print("\n=== Reranker Hierarchical RAG Evaluation Summary ===")
            print(f"Total Questions: {metrics['total_questions']}")
            print(f"Correct Rule Found: {metrics['correct_rule_found']} ({metrics['accuracy']:.2%})")
            print(f"Top-1 Accuracy: {metrics['top_1_accuracy']:.2%}")
            print(f"Top-3 Accuracy: {metrics['top_3_accuracy']:.2%}")
            print(f"Top-5 Accuracy: {metrics['top_5_accuracy']:.2%}")
            print(f"Average Query Time: {metrics['avg_query_time']:.4f} seconds")
            
            # Clean up
            if os.path.exists(reranker_results_file):
                os.remove(reranker_results_file)
                
        # Run Augmented Reranker RAG if selected
        if args.system in ['augmented-reranker', 'all']:
            augmented_reranker_results_file = tempfile.mktemp(suffix='.json', prefix='augmented_reranker_results_')
            
            print(f"Running augmented_reranker_rag.py...")
            run_augmented_reranker_rag(
                questions_file=questions_file,
                srd_file=args.srd,
                output_file=augmented_reranker_results_file,
                top_k=args.top_k,
                model=args.model,
                reranker_model=args.reranker,
                cache_dir=args.cache_dir,
                verbose=args.verbose,
                parallel=args.parallel,
                device=args.device
            )
            
            print(f"Evaluating Augmented Reranker RAG results...")
            augmented_reranker_results = evaluate_results(
                rag_results_file=augmented_reranker_results_file,
                qa_pairs_file=args.qa_pairs
            )
            
            all_results['augmented_reranker'] = augmented_reranker_results
            
            # Print summary metrics
            metrics = augmented_reranker_results['metrics']
            print("\n=== Augmented Reranker RAG Evaluation Summary ===")
            print(f"Total Questions: {metrics['total_questions']}")
            print(f"Correct Rule Found: {metrics['correct_rule_found']} ({metrics['accuracy']:.2%})")
            print(f"Top-1 Accuracy: {metrics['top_1_accuracy']:.2%}")
            print(f"Top-3 Accuracy: {metrics['top_3_accuracy']:.2%}")
            print(f"Top-5 Accuracy: {metrics['top_5_accuracy']:.2%}")
            print(f"Average Query Time: {metrics['avg_query_time']:.4f} seconds")
            
            # Clean up
            if os.path.exists(augmented_reranker_results_file):
                os.remove(augmented_reranker_results_file)
        
        # Save combined results if evaluating multiple systems
        if args.system == 'all':
            # Add comparison metrics
            comparison = {
                'systems': list(all_results.keys()),
                'metrics_comparison': {
                    'accuracy': {system: results['metrics']['accuracy'] for system, results in all_results.items()},
                    'top_1_accuracy': {system: results['metrics']['top_1_accuracy'] for system, results in all_results.items()},
                    'top_3_accuracy': {system: results['metrics']['top_3_accuracy'] for system, results in all_results.items()},
                    'top_5_accuracy': {system: results['metrics']['top_5_accuracy'] for system, results in all_results.items()},
                    'avg_query_time': {system: results['metrics']['avg_query_time'] for system, results in all_results.items()}
                }
            }
            
            all_results['comparison'] = comparison
            
            # Print comparison
            print("\n=== Systems Comparison ===")
            for metric in ['accuracy', 'top_1_accuracy', 'top_3_accuracy', 'top_5_accuracy']:
                print(f"\n{metric.replace('_', ' ').title()}:")
                for system in comparison['systems']:
                    value = comparison['metrics_comparison'][metric][system]
                    print(f"  {system}: {value:.2%}")
            
            print("\nAverage Query Time:")
            for system in comparison['systems']:
                value = comparison['metrics_comparison']['avg_query_time'][system]
                print(f"  {system}: {value:.4f} seconds")
        
        # Save results to output file
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    finally:
        # Clean up temporary files
        if os.path.exists(questions_file):
            os.remove(questions_file)

if __name__ == "__main__":
    main()
