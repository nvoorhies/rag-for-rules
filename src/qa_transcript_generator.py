import os
import re
import json
import asyncio
import logging
import time
import glob
from pathlib import Path
from typing import List, Dict, Any
import google.generativeai as genai
import argparse

class TranscriptQAGenerator:
    """Generates rules-based QA pairs from D&D gameplay transcripts."""
    
    def __init__(self, api_key: str, srd_path: str):
        """Initialize with API key and path to SRD document.
        
        Args:
            api_key: API key for Gemini model
            srd_path: Path to the SRD markdown file
        """
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Load SRD content
        with open(srd_path, 'r', encoding='utf-8') as f:
            self.srd_content = f.read()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting settings
        self.requests_per_minute = 10  # Conservative limit
        self.last_request_time = 0
        self._rate_limit_lock = asyncio.Lock()
    
    async def _rate_limit(self):
        """Implement rate limiting for API calls with async sleep."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < (60 / self.requests_per_minute):
            await asyncio.sleep((60 / self.requests_per_minute) - time_since_last)
        self.last_request_time = time.time()
    
    async def generate_qa_from_transcript(self, transcript_path: str) -> List[Dict[str, str]]:
        """Generate QA pairs from a transcript file.
        
        Args:
            transcript_path: Path to the transcript file
            
        Returns:
            List of dictionaries with 'question', 'answer', and 'rules' keys
        """
        # Read transcript content
        with open(transcript_path, 'r', encoding='utf-8', errors='replace') as f:
            transcript_content = f.read()
        
        # Construct prompt
        prompt = f"""Give me a list of all the Dungeons & dragons rules questions in the following gameplay transcript, and rules applications, with the question, answer, and a list of rules used for the roll or rules judgement. When specific player or character names are used, replace them with the information about the character's level, class, race, and abilities and skills but only those relevant to the roll, skill check, or rules question in the question section. Add or remove detail so that the question should be answerable unambiguously using only the information in the rules SRD. When specific rolls are made, the question and answer should refer to the dice rolled, target value, adjustments, and so on rather than the result of the dice rolling in this particular instance.

Format each QA pair as a JSON object with the following structure:
{{
  "question": "rules question",
  "answer": "rules answer",
  "rules": ["rules", "from",  "SRD", "used", "to", "generate", "the", "answer"],
  "rules_explanation: "Free form text explaining how the rules were applied to the question"
}}

Output a list of these JSON objects. Only include actual rules questions that appear in the transcript - do not invent examples.
Rules names should be the fully hierarchy in the SRD document for each rules string in the format "Section > Subsection > Rule Name" for the appropriate rules in the document hierarchy.

GAMEPLAY TRANSCRIPT:
{transcript_content}

SRD RULES FOR REFERENCE:
{self.srd_content}  # Truncate if needed for token limits
"""
        
        # Apply rate limiting
        async with self._rate_limit_lock:
            await self._rate_limit()
        
        # Call Gemini API
        max_retries = 3
        retry_count = 0
        backoff_time = 2
        
        while retry_count < max_retries:
            try:
                self.logger.info(f"Processing transcript: {os.path.basename(transcript_path)}")
                response = await self.model.generate_content_async(prompt)
                
                # Find JSON objects in response
                qa_pairs = self._extract_json_objects(response.text)
                
                if qa_pairs:
                    self.logger.info(f"Generated {len(qa_pairs)} QA pairs from {os.path.basename(transcript_path)}")
                    return qa_pairs
                else:
                    self.logger.warning(f"Failed to extract valid JSON from response. Retrying ({retry_count+1}/{max_retries})")
                    retry_count += 1
                    await asyncio.sleep(backoff_time * retry_count)
            except Exception as e:
                self.logger.warning(f"Error processing transcript {os.path.basename(transcript_path)} (attempt {retry_count+1}): {str(e)}")
                retry_count += 1
                await asyncio.sleep(backoff_time * retry_count)
        
        self.logger.error(f"Failed to process transcript after {max_retries} attempts: {os.path.basename(transcript_path)}")
        return []
    
    def _extract_json_objects(self, text: str) -> List[Dict[str, str]]:
        """Extract JSON objects from response text.
        
        Args:
            text: Response text that might contain JSON
            
        Returns:
            List of extracted JSON objects
        """
        # First, try to parse as a JSON array
        try:
            # Look for a JSON array pattern
            array_match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
            if array_match:
                return json.loads(array_match.group())
        except json.JSONDecodeError:
            pass
        
        # If that fails, try to find individual JSON objects
        qa_pairs = []
        json_pattern = r'\{\s*"question":\s*"[^"]*",\s*"answer":\s*"[^"]*",\s*"rules":\s*"[^"]*"\s*\}'
        matches = re.finditer(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                qa_pair = json.loads(match.group())
                qa_pairs.append(qa_pair)
            except json.JSONDecodeError:
                continue
        
        return qa_pairs
    
    async def process_transcript_directory(
        self,
        transcript_dir: str,
        output_dir: str,
        file_pattern: str = "*.txt",
        concurrency_limit: int = 5
    ):
        """Process all transcript files in a directory.
        
        Args:
            transcript_dir: Directory containing transcript files
            output_dir: Directory to save output files
            file_pattern: Glob pattern to match transcript files
            concurrency_limit: Maximum number of concurrent requests
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get list of transcript files
        transcript_files = glob.glob(os.path.join(transcript_dir, file_pattern))
        self.logger.info(f"Found {len(transcript_files)} transcript files to process")
        
        if not transcript_files:
            self.logger.warning(f"No transcript files found in {transcript_dir} with pattern {file_pattern}")
            return
        
        # Set up semaphore for concurrency control
        semaphore = asyncio.Semaphore(concurrency_limit)
        
        async def process_with_semaphore(transcript_path):
            """Helper function to process with semaphore-controlled concurrency."""

            basename = os.path.splitext(os.path.basename(transcript_path))[0]
            output_path = os.path.join(output_dir, f"{basename}_qa.json")
            if os.path.exists(output_path):
                self.logger.info(f"Skipping {os.path.basename(transcript_path)} - output file already exists")
                return 0
            async with semaphore:
                qa_pairs = await self.generate_qa_from_transcript(transcript_path)
                if qa_pairs:
                    # Save results to output file
                    basename = os.path.splitext(os.path.basename(transcript_path))[0]
                    output_path = os.path.join(output_dir, f"{basename}_qa.json")
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
                    
                    self.logger.info(f"Saved QA pairs to {output_path}")
                return len(qa_pairs)
        
        # Create tasks for all transcript files
        tasks = [process_with_semaphore(path) for path in transcript_files]
        
        # Execute all tasks and gather results
        self.logger.info(f"Starting parallel processing with max {concurrency_limit} concurrent requests")
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time
        
        # Report summary
        total_qa_pairs = sum(results)
        self.logger.info(f"Processing completed in {duration:.2f} seconds")
        self.logger.info(f"Generated a total of {total_qa_pairs} QA pairs from {len(transcript_files)} transcripts")
        
        # Create a combined file with all QA pairs
        if total_qa_pairs > 0:
            all_qa_pairs = []
            for transcript_path in transcript_files:
                basename = os.path.splitext(os.path.basename(transcript_path))[0]
                qa_file = os.path.join(output_dir, f"{basename}_qa.json")
                if os.path.exists(qa_file):
                    with open(qa_file, 'r', encoding='utf-8') as f:
                        try:
                            all_qa_pairs.extend(json.load(f))
                        except json.JSONDecodeError:
                            self.logger.warning(f"Could not load QA pairs from {qa_file}")
            
            # Save combined file
            combined_path = os.path.join(output_dir, "all_qa_pairs.json")
            with open(combined_path, 'w', encoding='utf-8') as f:
                json.dump(all_qa_pairs, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Created combined file with all QA pairs: {combined_path}")

async def main():
    parser = argparse.ArgumentParser(description='Generate D&D rules QA pairs from gameplay transcripts')
    parser.add_argument('--transcripts', required=True, help='Directory containing transcript files')
    parser.add_argument('--srd', required=True, help='Path to SRD markdown file')
    parser.add_argument('--output', default='qa_output', help='Output directory')
    parser.add_argument('--pattern', default='*.txt', help='File pattern for transcript files')
    parser.add_argument('--concurrency', type=int, default=5, help='Maximum concurrent requests')
    parser.add_argument('--api-key', help='Gemini API key (or use GEMINI_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Get API key from args or environment
    api_key = args.api_key or os.environ.get('GEMINI_API_KEY')
    
    if not api_key:
        raise ValueError("Please provide API key via --api-key or GEMINI_API_KEY environment variable")
    
    # Initialize generator
    generator = TranscriptQAGenerator(api_key, args.srd)
    
    # Process transcripts
    await generator.process_transcript_directory(
        args.transcripts,
        args.output,
        args.pattern,
        args.concurrency
    )

if __name__ == "__main__":
    asyncio.run(main())