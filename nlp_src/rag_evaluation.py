#!/usr/bin/env python3

"""
RAG Evaluation Script - Evaluates how well different RAG approaches 
enable a local LLM to correctly answer RPG rules questions.
"""

import json
import time
import argparse
import os
import requests
from typing import List, Dict, Any, Optional, Tuple
import re
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_eval.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag_eval")

class RAGEvaluator:
    """Evaluates RAG systems using a local LLM and QA pairs."""
    
    def __init__(self, config_path: str):
        """Initialize the evaluator with configuration."""
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Import RAG systems
        self.rag_systems = self._import_rag_systems()
        
        # Initialize Ollama settings
        self.ollama_base_url = self.config.get('ollama_base_url', 'http://localhost:11434')
        self.model_name = self.config.get('model_name', 'llama2:7b-q4_0')
        
        logger.info(f"Initialized RAG Evaluator with {len(self.rag_systems)} systems")
        logger.info(f"Using LLM: {self.model_name}")
    
    def _import_rag_systems(self):
        """Import RAG systems based on configuration."""
        systems = {}
        
        # Import the naive RAG implementation
        try:
            from naive_rag import NaiveRAG
            systems['naive'] = NaiveRAG(self.config['paths']['text_chunks_path'])
            logger.info("Loaded Naive RAG system")
        except ImportError as e:
            logger.warning(f"NaiveRAG not imported: {e}")
        except Exception as e:
            logger.error(f"Error initializing NaiveRAG: {e}")
        
        # Import the structure-aware RAG implementation
        try:
            from rules_rag import RulesRAG
            systems['structure'] = RulesRAG(self.config['paths']['processed_srd_path'])
            logger.info("Loaded Structure-aware RAG system")
        except ImportError as e:
            logger.warning(f"Structure-aware RulesRAG not imported: {e}")
        except Exception as e:
            logger.error(f"Error initializing RulesRAG: {e}")
        
        # Import the NN-augmented RAG implementation
        try:
            from nn_augmented_rag import NNAugmentedRAG
            systems['neural'] = NNAugmentedRAG(
                self.config['paths']['processed_srd_path'],
                self.config['paths'].get('embeddings_path')
            )
            logger.info("Loaded NN-augmented RAG system")
        except ImportError as e:
            logger.warning(f"NNAugmentedRAG not imported: {e}")
        except Exception as e:
            logger.error(f"Error initializing NNAugmentedRAG: {e}")
        
        return systems
    
    def load_qa_pairs(self, qa_path: str) -> List[Dict[str, str]]:
        """Load question-answer pairs from file."""
        with open(qa_path, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)
        
        logger.info(f"Loaded {len(qa_pairs)} QA pairs from {qa_path}")
        return qa_pairs
    
    def query_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Query the Ollama API with a prompt."""
        url = f"{self.ollama_base_url}/api/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            logger.error(f"Error querying Ollama: {e}")
            return f"Error: {str(e)}"
    
    def parse_llm_response(self, response: str) -> Dict[str, str]:
        """Parse the LLM response into answer and justification parts."""
        answer = ""
        justification = ""
        
        # Try to find the ANSWER section
        answer_match = re.search(r"ANSWER:\s*(.+?)(?=JUSTIFICATION:|$)", response, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
        
        # Try to find the JUSTIFICATION section
        justification_match = re.search(r"JUSTIFICATION:\s*(.+)$", response, re.DOTALL)
        if justification_match:
            justification = justification_match.group(1).strip()
        
        # If we couldn't find structured parts, use the whole response as answer
        if not answer and not justification:
            answer = response.strip()
        
        return {
            "answer": answer,
            "justification": justification
        }
    
    def prepare_prompt(self, question: str, retrieved_rules: List[Dict[str, Any]]) -> str:
        """Prepare a prompt for the LLM using the question and retrieved rules."""
        # System prompt for Ollama
        system_prompt = "You are a helpful Game Master for a tabletop RPG. Your task is to answer rules questions based ONLY on the provided rules text. Do not use any other knowledge."
        
        # Start with instructions
        prompt = "Using ONLY the rules provided below, answer the following RPG rules question:\n\n"
        
        # Add the question
        prompt += f"QUESTION: {question}\n\n"
        
        # Add the retrieved rules
        prompt += "RULES:\n"
        
        for i, rule in enumerate(retrieved_rules, 1):
            rule_text = rule.get('text', '')
            rule_title = rule.get('title', f"Rule {i}")
            
            prompt += f"Rule {i}: {rule_title}\n{rule_text}\n\n"
        
        # Structured output instructions
        prompt += """
Based ONLY on these rules, respond in the following format:

ANSWER: [Your concise answer - just Yes, No, or a very brief factual statement]

JUSTIFICATION: [Your detailed explanation with reasoning, referencing specific rules]

Important: If the provided rules are insufficient to answer the question, your ANSWER should state "Cannot determine" and your JUSTIFICATION should explain why.
"""
        
        return prompt, system_prompt
    
    def evaluate_single_qa(self, qa_pair: Dict[str, str], rag_system_name: str, max_rules: int = 5) -> Dict[str, Any]:
        """Evaluate a single QA pair with a specific RAG system."""
        question = qa_pair['question']
        expected_answer = qa_pair['answer']
        expected_justification = qa_pair.get('justification', '')
        
        start_time = time.time()
        
        # Query the RAG system
        rag_system = self.rag_systems[rag_system_name]
        try:
            if rag_system_name == 'naive':
                rag_result = rag_system.query(question, top_k=max_rules)
            else:
                rag_result = rag_system.query(question, max_rules=max_rules)
            
            retrieved_rules = rag_result.get('rules', [])
            
            # Prepare the prompt with retrieved rules
            prompt, system_prompt = self.prepare_prompt(question, retrieved_rules)
            
            # Query the LLM
            llm_response = self.query_ollama(prompt, system_prompt)
            
            # Parse the response
            parsed_response = self.parse_llm_response(llm_response)
            llm_answer = parsed_response['answer']
            llm_justification = parsed_response['justification']
            
            # Calculate metrics
            rag_time = rag_result.get('query_time', time.time() - start_time)
            total_time = time.time() - start_time
            llm_time = total_time - rag_time
            
            # Calculate answer similarity
            answer_similarity = self.calculate_answer_similarity(expected_answer, llm_answer)
            justification_similarity = 0.0
            if expected_justification and llm_justification:
                justification_similarity = self.calculate_answer_similarity(expected_justification, llm_justification)
            
            # Determine if the answer is correct (using a threshold)
            is_correct = answer_similarity > self.config.get('answer_similarity_threshold', 0.7)
            
            # Return evaluation results
            return {
                'question': question,
                'expected_answer': expected_answer,
                'expected_justification': expected_justification,
                'llm_answer': llm_answer,
                'llm_justification': llm_justification,
                'system': rag_system_name,
                'num_rules': len(retrieved_rules),
                'rag_time': rag_time,
                'llm_time': llm_time,
                'total_time': total_time,
                'answer_similarity': answer_similarity,
                'justification_similarity': justification_similarity,
                'is_correct': is_correct,
                'rules': [{'title': r.get('title', ''), 'id': r.get('id', '')} for r in retrieved_rules]
            }
            
        except Exception as e:
            logger.error(f"Error evaluating question with {rag_system_name}: {e}")
            return {
                'question': question,
                'expected_answer': expected_answer,
                'expected_justification': expected_justification,
                'llm_answer': f"Error: {str(e)}",
                'llm_justification': "",
                'system': rag_system_name,
                'num_rules': 0,
                'rag_time': 0,
                'llm_time': 0,
                'total_time': time.time() - start_time,
                'answer_similarity': 0.0,
                'justification_similarity': 0.0,
                'is_correct': False,
                'rules': [],
                'error': str(e)
            }
    
    def calculate_answer_similarity(self, expected: str, actual: str) -> float:
        """Calculate similarity between expected and actual answers."""
        # This is a simple implementation - could be improved with embeddings or other NLP techniques
        # Normalize both strings
        expected = expected.lower().strip()
        actual = actual.lower().strip()
        
        # Simple token-based Jaccard similarity
        expected_tokens = set(re.findall(r'\b\w+\b', expected))
        actual_tokens = set(re.findall(r'\b\w+\b', actual))
        
        if not expected_tokens and not actual_tokens:
            return 1.0
        
        # Calculate Jaccard similarity
        intersection = expected_tokens.intersection(actual_tokens)
        union = expected_tokens.union(actual_tokens)
        
        return len(intersection) / len(union)
    
    def evaluate_qa_pairs(self, qa_pairs: List[Dict[str, str]], 
                          systems: Optional[List[str]] = None,
                          max_rules: int = 5,
                          num_workers: int = 1,
                          max_pairs: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate multiple QA pairs with specified RAG systems."""
        if systems is None:
            systems = list(self.rag_systems.keys())
        
        # Limit the number of QA pairs if specified
        if max_pairs:
            qa_pairs = qa_pairs[:max_pairs]
        
        logger.info(f"Evaluating {len(qa_pairs)} QA pairs with systems: {', '.join(systems)}")
        
        # Create tasks - each task is (qa_pair, system_name)
        tasks = [(qa_pair, system) for qa_pair in qa_pairs for system in systems]
        
        all_results = []
        
        if num_workers > 1:
            # Use parallel processing
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(self.evaluate_single_qa, qa_pair, system, max_rules): (qa_pair, system)
                    for qa_pair, system in tasks
                }
                
                # Process results as they complete
                for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks)):
                    qa_pair, system = future_to_task[future]
                    try:
                        result = future.result()
                        all_results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing {qa_pair['question'][:50]}... with {system}: {e}")
        else:
            # Use sequential processing
            for qa_pair, system in tqdm(tasks, desc="Evaluating QA pairs"):
                result = self.evaluate_single_qa(qa_pair, system, max_rules)
                all_results.append(result)
        
        # Calculate similarities between expected and actual answers
        for result in all_results:
            result['answer_similarity'] = self.calculate_answer_similarity(
                result['expected_answer'], result['llm_answer']
            )
        
        return all_results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze evaluation results and generate statistics."""
        analysis = {
            'total_questions': len(set(r['question'] for r in results)),
            'total_evaluations': len(results),
            'by_system': {}
        }
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(results)
        
        # Group by system
        for system in df['system'].unique():
            system_df = df[df['system'] == system]
            
            system_analysis = {
                'count': len(system_df),
                'accuracy': len(system_df[system_df['is_correct']]) / len(system_df) * 100,
                'avg_rules_retrieved': system_df['num_rules'].mean(),
                'avg_rag_time': system_df['rag_time'].mean(),
                'avg_llm_time': system_df['llm_time'].mean(),
                'avg_total_time': system_df['total_time'].mean(),
                'avg_answer_similarity': system_df['answer_similarity'].mean(),
                'avg_justification_similarity': system_df['justification_similarity'].mean(),
                'accuracy_distribution': {
                    'correct': len(system_df[system_df['is_correct']]),
                    'incorrect': len(system_df[~system_df['is_correct']])
                },
                'similarity_distribution': {
                    'high (>0.8)': len(system_df[system_df['answer_similarity'] > 0.8]),
                    'medium (0.5-0.8)': len(system_df[(system_df['answer_similarity'] > 0.5) & 
                                                      (system_df['answer_similarity'] <= 0.8)]),
                    'low (<0.5)': len(system_df[system_df['answer_similarity'] <= 0.5])
                }
            }
            
            analysis['by_system'][system] = system_analysis
        
        return analysis
        
    def generate_html_report(self, results: List[Dict[str, Any]], 
                             analysis: Dict[str, Any], 
                             output_path: str):
        """Generate an HTML report of evaluation results."""
        # Convert results to DataFrame
        df = pd.DataFrame(results)
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>RAG Evaluation Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3, h4 {{
                    color: #2c3e50;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 20px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .system-comparison {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .system-card {{
                    flex: 1;
                    min-width: 300px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .system-header {{
                    font-size: 1.2rem;
                    font-weight: bold;
                    margin-bottom: 10px;
                    padding-bottom: 5px;
                    border-bottom: 1px solid #eee;
                }}
                .metric {{
                    margin-bottom: 5px;
                }}
                .metric-name {{
                    font-weight: bold;
                }}
                .sample-qa {{
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                    margin-bottom: 20px;
                }}
                .question {{
                    font-weight: bold;
                    margin-bottom: 10px;
                }}
                .answer {{
                    margin-bottom: 10px;
                    padding-left: 15px;
                    border-left: 3px solid #eee;
                }}
                .answer.correct {{
                    border-left: 3px solid #2ecc71;
                }}
                .answer.incorrect {{
                    border-left: 3px solid #e74c3c;
                }}
                .justification {{
                    margin-bottom: 10px;
                    padding-left: 15px;
                    padding-top: 5px;
                    font-size: 0.9em;
                    color: #555;
                }}
                .rules-list {{
                    font-size: 0.9rem;
                    color: #666;
                }}
                .accuracy-bar {{
                    height: 20px;
                    background-color: #f1f1f1;
                    border-radius: 4px;
                    margin-top: 5px;
                    overflow: hidden;
                }}
                .accuracy-fill {{
                    height: 100%;
                    background-color: #4CAF50;
                    text-align: center;
                    line-height: 20px;
                    color: white;
                }}
            </style>
        </head>
        <body>
            <h1>RPG Rules RAG Evaluation Report</h1>
            
            <h2>Overview</h2>
            <p>
                Total Questions: {analysis['total_questions']}<br>
                Total Evaluations: {analysis['total_evaluations']}
            </p>
            
            <h2>System Comparison</h2>
            <div class="system-comparison">
        """
        
        # Add system cards
        for system, stats in analysis['by_system'].items():
            html_content += f"""
                <div class="system-card">
                    <div class="system-header">{system.capitalize()} RAG</div>
                    
                    <div class="metric">
                        <span class="metric-name">Accuracy:</span> 
                        {stats['accuracy']:.1f}%
                        <div class="accuracy-bar">
                            <div class="accuracy-fill" style="width: {stats['accuracy']}%">
                                {stats['accuracy']:.1f}%
                            </div>
                        </div>
                    </div>
                    
                    <div class="metric">
                        <span class="metric-name">Average Answer Similarity:</span> 
                        {stats['avg_answer_similarity']:.2f}
                    </div>
                    
                    <div class="metric">
                        <span class="metric-name">Average Justification Similarity:</span> 
                        {stats['avg_justification_similarity']:.2f}
                    </div>
                    
                    <div class="metric">
                        <span class="metric-name">Average Rules Retrieved:</span> 
                        {stats['avg_rules_retrieved']:.2f}
                    </div>
                    
                    <div class="metric">
                        <span class="metric-name">Average RAG Time:</span> 
                        {stats['avg_rag_time']:.2f}s
                    </div>
                    
                    <div class="metric">
                        <span class="metric-name">Average LLM Time:</span> 
                        {stats['avg_llm_time']:.2f}s
                    </div>
                    
                    <div class="metric">
                        <span class="metric-name">Correct/Incorrect:</span>
                        {stats['accuracy_distribution']['correct']} / {stats['accuracy_distribution']['incorrect']}
                    </div>
                </div>
            """
        
        html_content += """
            </div>
            
            <h2>System Performance Comparison</h2>
            <table>
                <tr>
                    <th>System</th>
                    <th>Accuracy</th>
                    <th>Avg Answer Similarity</th>
                    <th>Avg Justification Sim.</th>
                    <th>Avg Rules Retrieved</th>
                    <th>Avg RAG Time (s)</th>
                    <th>Avg LLM Time (s)</th>
                </tr>
        """
        
        # Add system comparison rows
        for system, stats in analysis['by_system'].items():
            html_content += f"""
                <tr>
                    <td>{system.capitalize()}</td>
                    <td>{stats['accuracy']:.1f}%</td>
                    <td>{stats['avg_answer_similarity']:.3f}</td>
                    <td>{stats['avg_justification_similarity']:.3f}</td>
                    <td>{stats['avg_rules_retrieved']:.1f}</td>
                    <td>{stats['avg_rag_time']:.2f}</td>
                    <td>{stats['avg_llm_time']:.2f}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Sample Question-Answer Pairs</h2>
        """
        
        # Add sample QA pairs (first 10)
        questions_shown = set()
        for idx, result in enumerate(results):
            question = result['question']
            
            # Only show each question once
            if question in questions_shown or len(questions_shown) >= 10:
                continue
            
            questions_shown.add(question)
            
            html_content += f"""
                <div class="sample-qa">
                    <div class="question">Question: {question}</div>
                    
                    <div class="answer">
                        <strong>Expected Answer:</strong><br>
                        {result['expected_answer']}
                    </div>
                    
                    <div class="justification">
                        <strong>Expected Justification:</strong><br>
                        {result.get('expected_justification', 'N/A')}
                    </div>
                    
                    <h4>System Answers:</h4>
            """
            
            # Get all system results for this question
            question_results = [r for r in results if r['question'] == question]
            
            for qr in question_results:
                system = qr['system']
                answer_similarity = qr['answer_similarity']
                is_correct = qr.get('is_correct', False)
                answer_class = "correct" if is_correct else "incorrect"
                
                html_content += f"""
                    <div class="answer {answer_class}">
                        <strong>{system.capitalize()} (Similarity: {answer_similarity:.2f}):</strong><br>
                        {qr['llm_answer']}
                    </div>
                    
                    <div class="justification">
                        {qr['llm_justification']}
                    </div>
                    
                    <div class="rules-list">
                        <strong>Retrieved Rules:</strong> 
                        {', '.join([r['title'] for r in qr['rules']])}
                    </div>
                """
            
            html_content += """
                </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated at {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    def save_analysis(self, analysis: Dict[str, Any], output_path: str):
        """Save analysis results to a file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"Analysis saved to {output_path}")
    
    def run_evaluation(self, qa_path: str, output_dir: str, 
                       systems: Optional[List[str]] = None,
                       max_rules: int = 5,
                       num_workers: int = 1,
                       max_pairs: Optional[int] = None):
        """Run the full evaluation pipeline."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load QA pairs
        qa_pairs = self.load_qa_pairs(qa_path)
        
        # Evaluate QA pairs
        results = self.evaluate_qa_pairs(
            qa_pairs, systems, max_rules, num_workers, max_pairs
        )
        
        # Save raw results
        results_path = os.path.join(output_dir, "evaluation_results.json")
        self.save_results(results, results_path)
        
        # Analyze results
        analysis = self.analyze_results(results)
        
        # Save analysis
        analysis_path = os.path.join(output_dir, "evaluation_analysis.json")
        self.save_analysis(analysis, analysis_path)
        
        # Generate HTML report
        report_path = os.path.join(output_dir, "evaluation_report.html")
        self.generate_html_report(results, analysis, report_path)
        
        return results, analysis
    
    def generate_html_report(self, results: List[Dict[str, Any]], 
                             analysis: Dict[str, Any], 
                             output_path: str):
        """Generate an HTML report of evaluation results."""
        # Convert results to DataFrame
        df = pd.DataFrame(results)
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>RAG Evaluation Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3, h4 {{
                    color: #2c3e50;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 20px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .system-comparison {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .system-card {{
                    flex: 1;
                    min-width: 300px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .system-header {{
                    font-size: 1.2rem;
                    font-weight: bold;
                    margin-bottom: 10px;
                    padding-bottom: 5px;
                    border-bottom: 1px solid #eee;
                }}
                .metric {{
                    margin-bottom: 5px;
                }}
                .metric-name {{
                    font-weight: bold;
                }}
                .sample-qa {{
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                    margin-bottom: 20px;
                }}
                .question {{
                    font-weight: bold;
                    margin-bottom: 10px;
                }}
                .answer {{
                    margin-bottom: 10px;
                    padding-left: 15px;
                    border-left: 3px solid #eee;
                }}
                .rules-list {{
                    font-size: 0.9rem;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <h1>RAG Evaluation Report</h1>
            
            <h2>Overview</h2>
            <p>
                Total Questions: {analysis['total_questions']}<br>
                Total Evaluations: {analysis['total_evaluations']}
            </p>
            
            <h2>System Comparison</h2>
            <div class="system-comparison">
        """
        
        # Add system cards
        for system, stats in analysis['by_system'].items():
            html_content += f"""
                <div class="system-card">
                    <div class="system-header">{system.capitalize()} RAG</div>
                    <div class="metric">
                        <span class="metric-name">Average Answer Similarity:</span> 
                        {stats['avg_answer_similarity']:.2f}
                    </div>
                    <div class="metric">
                        <span class="metric-name">Average Rules Retrieved:</span> 
                        {stats['avg_rules_retrieved']:.2f}
                    </div>
                    <div class="metric">
                        <span class="metric-name">Average RAG Time:</span> 
                        {stats['avg_rag_time']:.2f}s
                    </div>
                    <div class="metric">
                        <span class="metric-name">Average LLM Time:</span> 
                        {stats['avg_llm_time']:.2f}s
                    </div>
                    <div class="metric">
                        <span class="metric-name">Similarity Distribution:</span><br>
                        - High (>0.8): {stats['similarity_distribution']['high (>0.8)']} 
                          ({stats['similarity_distribution']['high (>0.8)'] / stats['count'] * 100:.1f}%)<br>
                        - Medium (0.5-0.8): {stats['similarity_distribution']['medium (0.5-0.8)']} 
                          ({stats['similarity_distribution']['medium (0.5-0.8)'] / stats['count'] * 100:.1f}%)<br>
                        - Low (<0.5): {stats['similarity_distribution']['low (<0.5)']} 
                          ({stats['similarity_distribution']['low (<0.5)'] / stats['count'] * 100:.1f}%)
                    </div>
                </div>
            """
        
        html_content += """
            </div>
            
            <h2>System Performance Comparison</h2>
            <table>
                <tr>
                    <th>System</th>
                    <th>Avg Answer Similarity</th>
                    <th>Avg Rules Retrieved</th>
                    <th>Avg RAG Time (s)</th>
                    <th>Avg LLM Time (s)</th>
                    <th>High Similarity %</th>
                </tr>
        """
        
        # Add system comparison rows
        for system, stats in analysis['by_system'].items():
            high_sim_percent = stats['similarity_distribution']['high (>0.8)'] / stats['count'] * 100
            html_content += f"""
                <tr>
                    <td>{system.capitalize()}</td>
                    <td>{stats['avg_answer_similarity']:.3f}</td>
                    <td>{stats['avg_rules_retrieved']:.1f}</td>
                    <td>{stats['avg_rag_time']:.2f}</td>
                    <td>{stats['avg_llm_time']:.2f}</td>
                    <td>{high_sim_percent:.1f}%</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Sample Question-Answer Pairs</h2>
        """
        
        # Add sample QA pairs (first 10)
        questions_shown = set()
        for idx, result in enumerate(results):
            question = result['question']
            
            # Only show each question once
            if question in questions_shown or len(questions_shown) >= 10:
                continue
            
            questions_shown.add(question)
            
            html_content += f"""
                <div class="sample-qa">
                    <div class="question">Question: {question}</div>
                    
                    <div class="answer">
                        <strong>Expected Answer:</strong><br>
                        {result['expected_answer']}
                    </div>
                    
                    <h4>System Answers:</h4>
            """
            
            # Get all system results for this question
            question_results = [r for r in results if r['question'] == question]
            
            for qr in question_results:
                system = qr['system']
                similarity = qr['answer_similarity']
                
                html_content += f"""
                    <div class="answer">
                        <strong>{system.capitalize()} (Similarity: {similarity:.2f}):</strong><br>
                        {qr['llm_answer']}
                        
                        <div class="rules-list">
                            <strong>Retrieved Rules:</strong> 
                            {', '.join([r['title'] for r in qr['rules']])}
                        </div>
                    </div>
                """
            
            html_content += """
                </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated at {output_path}")

def main():
    """Main entry point for the evaluation script."""
    parser = argparse.ArgumentParser(description='RAG Evaluation with Local LLM')
    
    parser.add_argument('--config', '-c', required=True, help='Path to configuration file')
    parser.add_argument('--qa-pairs', '-q', required=True, help='Path to QA pairs JSON file')
    parser.add_argument('--output-dir', '-o', default='evaluation_results', help='Output directory for results')
    parser.add_argument('--systems', '-s', nargs='+', help='RAG systems to evaluate (default: all available)')
    parser.add_argument('--max-rules', '-m', type=int, default=5, help='Maximum rules to retrieve')
    parser.add_argument('--workers', '-w', type=int, default=1, help='Number of worker processes')
    parser.add_argument('--max-pairs', '-p', type=int, help='Maximum number of QA pairs to evaluate')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = RAGEvaluator(args.config)
    
    # Run evaluation
    evaluator.run_evaluation(
        qa_path=args.qa_pairs,
        output_dir=args.output_dir,
        systems=args.systems,
        max_rules=args.max_rules,
        num_workers=args.workers,
        max_pairs=args.max_pairs
    )

if __name__ == "__main__":
    main()
