import google.generativeai as genai
from typing import List, Dict, Tuple, Optional, Set
import json
from pathlib import Path
import re
from dataclasses import dataclass
import asyncio
from datetime import datetime
import logging
import time
import os
import argparse

@dataclass
class RuleDependency:
    """Represents dependencies and cross-references for a rule."""
    section: str
    title: str
    explicit_references: List[str]  # Direct references to other rules
    implicit_dependencies: List[str]  # Rules needed for context
    systems_affected: List[str]  # Game systems this rule affects/modifies
    rule_text: str  # Full text of the rule section
    rule_hierarchy: str  # Path of the rule in the document hierarchy

class DependencyAnalyzer:
    """Analyzes SRD rules for dependencies using Gemini API."""
    
    def __init__(self, api_key: str):
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting settings
        self.requests_per_minute = 60
        self.last_request_time = 0
        self._rate_limit_lock = asyncio.Lock()
        
        # Progress tracking
        self.total_sections = 0
        self.completed_sections = 0
    
    def _split_into_sections(self, content: str) -> List[Tuple[str, str, str, str]]:
        """Split markdown content into sections with hierarchy tracking.
        
        Returns:
            List of tuples containing (level, title, content, hierarchy_path)
        """
        # Find all headers (any level)
        section_pattern = r'^(#{1,6})\s+(.+)$'
        sections = []
        current_section = []
        current_title = ""
        current_level = 0
        
        # Track section hierarchy
        hierarchy = [""] * 6  # Track up to 6 levels of headers
        
        for line in content.split('\n'):
            header_match = re.match(section_pattern, line, re.MULTILINE)
            if header_match:
                # Save previous section if it exists
                if current_title and current_section:
                    # Build hierarchy path
                    hierarchy_path = " > ".join([h for h in hierarchy[:current_level] if h])
                    
                    sections.append((
                        str(current_level),
                        current_title,
                        '\n'.join(current_section),
                        hierarchy_path
                    ))
                
                # Start new section
                header_level = len(header_match.group(1))
                current_title = header_match.group(2)
                current_level = header_level
                
                # Update hierarchy at this level
                hierarchy[current_level-1] = current_title
                # Clear lower levels in hierarchy
                for i in range(current_level, len(hierarchy)):
                    hierarchy[i] = ""
                
                current_section = [line]
            else:
                if current_section:  # Only append if we have a current section
                    current_section.append(line)
                # Ignore content before first header
        
        # Add final section
        if current_title and current_section:
            # Build hierarchy path
            hierarchy_path = " > ".join([h for h in hierarchy[:current_level] if h])
            
            sections.append((
                str(current_level),
                current_title,
                '\n'.join(current_section),
                hierarchy_path
            ))
        
        self.logger.info(f"Split content into {len(sections)} sections")
        return sections
    
    async def _rate_limit(self):
        """Implement rate limiting for API calls with async sleep."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < (60 / self.requests_per_minute):
            await asyncio.sleep((60 / self.requests_per_minute) - time_since_last)
        self.last_request_time = time.time()
    
    def _get_dependency_filename(self, section_level: str, section_title: str) -> str:
        """Generate a consistent filename for a dependency file."""
        # Create a valid filename from the rule title
        safe_title = re.sub(r'[^a-zA-Z0-9_\-]', '_', section_title)
        #safe_title = re.sub(r' +', '_', safe_title)  # Replace multiple spaces with single one
        safe_title = re.sub(r'_+', '_', safe_title)  # Replace multiple underscores with single one
        #safe_title = re.sub(r'>', '_', safe_title)  # Replace hierarchy separator with underscore
        
        # Create filename with section and title
        return f"{section_level}__{safe_title}.json"
    
    async def _write_dependency_file(self, dependency: RuleDependency, output_dir: str):
        """Write a single dependency analysis to a file."""
        # Get filename
        filename = self._get_dependency_filename(dependency.section, dependency.rule_hierarchy)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Write to file
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vars(dependency), f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Wrote dependency analysis for '{dependency.title}' to {filepath}")
    
    def _check_dependency_exists(self, section_level: str, section_title: str, output_dir: str) -> bool:
        """Check if dependency analysis file already exists."""
        filename = self._get_dependency_filename(section_level, section_title)
        filepath = os.path.join(output_dir, filename)
        
        if os.path.exists(filepath):
            try:
                # Verify file contains valid JSON
                with open(filepath, 'r', encoding='utf-8') as f:
                    dependency_data = json.load(f)
                    
                    # Check if file has the required fields
                    required_fields = ["explicit_references", "implicit_dependencies", "systems_affected"]
                    if all(field in dependency_data for field in required_fields):
                        return True
            except (json.JSONDecodeError, OSError):
                # If file is corrupt or can't be read, consider it non-existent
                return False
                
        return False
    
    def _load_existing_dependency(self, section_level: str, section_title: str, output_dir: str) -> Optional[RuleDependency]:
        """Load existing dependency analysis from file."""
        filename = self._get_dependency_filename(section_level, section_title)
        filepath = os.path.join(output_dir, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                return RuleDependency(
                    section=data.get("section", section_level),
                    title=data.get("title", section_title),
                    explicit_references=data.get("explicit_references", []),
                    implicit_dependencies=data.get("implicit_dependencies", []),
                    systems_affected=data.get("systems_affected", []),
                    rule_text=data.get("rule_text", ""),
                    rule_hierarchy=data.get("rule_hierarchy", "")
                )
        except (json.JSONDecodeError, OSError) as e:
            self.logger.warning(f"Error loading existing dependency file for {section_title}: {str(e)}")
            return None
    
    async def analyze_section(
        self,
        section_level: str,
        section_title: str,
        section_content: str,
        section_hierarchy: str,
        full_srd: str
    ) -> RuleDependency:
        """Analyze a section for dependencies using Gemini."""
        # Apply rate limiting with async-friendly approach
        async with self._rate_limit_lock:
            await self._rate_limit()
        
        prompt = fprompt = f"""Please analyze this rule section from a D&D System Reference Document (SRD) and identify all dependencies and cross-references.

SECTION TO ANALYZE:
{section_content}

SECTION HIERARCHY:
{section_hierarchy}

I need you to identify:
1. Explicit references: Any direct mentions in the rule text or references to other rules sections, including:
   - Mentions of other specific rules by name
   - References to pages or sections
   - Phrases like "see the rules for X" or "as described in Y"
   - References to specific spells, abilities, or mechanics defined elsewhere

2. Implicit dependencies: Rules that would be needed to fully understand or implement this rule, including:
   - Prerequisite rules concepts or mechanics
   - Related rules that this rule builds upon

3. Game systems affected: Which game systems this rule modifies or interacts with, such as:
   - Combat
   - Spellcasting
   - Skills or ability checks
   - Character creation/advancement
   - Movement or exploration
   - etc.

If you are uncertain, aim for having 1-2 explicit references, 2-3 implicit dependencies, and 1-2 affected systems.
When listing implicit and explicit dependencies on other rules, use their name in the rules hierarchy, such as "Fighter > Superior Critical > Survivor".
No dependencies should be listed that are not other rules found in the SRD.

Provide your response in this exact JSON format:
{{
    "explicit_references": ["list", "of", "referenced", "sections"],
    "implicit_dependencies": ["list", "of", "required", "rules"],
    "systems_affected": ["list", "of", "affected", "systems"]
}}

EXAMPLES OF REFERENCES:
- "See chapter X for more information"
- "As described in the combat rules"
- "Uses the same mechanics as the Wizard class feature"
- "Follows the normal rules for spellcasting"

FULL SRD FOR CONTEXT:
{full_srd}
"""

        max_retries = 3
        retry_count = 0
        backoff_time = 2
        
        while retry_count < max_retries:
            try:
                response = await self.model.generate_content_async(prompt)
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    
                    # Update progress
                    self.completed_sections += 1
                    completion_percentage = (self.completed_sections / self.total_sections) * 100
                    self.logger.info(f"Completed {self.completed_sections}/{self.total_sections} sections ({completion_percentage:.1f}%)")
                    
                    return RuleDependency(
                        section=section_level,
                        title=section_title,
                        explicit_references=result['explicit_references'],
                        implicit_dependencies=result['implicit_dependencies'],
                        systems_affected=result['systems_affected'],
                        rule_text=result.get('rule_text', section_content),
                        rule_hierarchy=result.get('rule_hierarchy', section_hierarchy)
                    )
                else:
                    self.logger.warning(f"Failed to parse JSON from response for section: {section_title}")
                    retry_count += 1
                    await asyncio.sleep(backoff_time * retry_count)
            except Exception as e:
                self.logger.warning(f"Error analyzing section {section_title} (attempt {retry_count+1}): {str(e)}")
                retry_count += 1
                await asyncio.sleep(backoff_time * retry_count)
        
        self.logger.error(f"Failed to analyze section after {max_retries} attempts: {section_title}")
        # Update progress even for failed analyses
        self.completed_sections += 1
        
        return RuleDependency(
            section=section_level,
            title=section_title,
            explicit_references=[],
            implicit_dependencies=[],
            systems_affected=[],
            rule_text=section_content,
            rule_hierarchy=section_hierarchy
        )
    
    async def analyze_srd(
        self, 
        filepath: str, 
        concurrency_limit: int = 10, 
        output_dir: str = "rule_dependencies"
    ) -> List[RuleDependency]:
        """Analyze entire SRD file for dependencies with parallel processing.
        
        Args:
            filepath: Path to the SRD markdown file
            concurrency_limit: Maximum number of concurrent requests
            output_dir: Directory to save individual dependency files
            
        Returns:
            List of RuleDependency objects
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Read SRD content
        self.logger.info(f"Reading SRD file: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            full_content = f.read()
        
        # Split into sections
        sections = self._split_into_sections(full_content)
        self.logger.info(f"Found {len(sections)} sections to analyze")
        
        # Set total sections for progress tracking
        self.total_sections = len(sections)
        self.completed_sections = 0
        
        # Set up semaphore for concurrency control
        semaphore = asyncio.Semaphore(concurrency_limit)
        
        async def analyze_with_semaphore(level, title, content, hierarchy):
            """Helper function to analyze with semaphore-controlled concurrency."""
            async with semaphore:
                # Check if analysis already exists
                if self._check_dependency_exists(level, hierarchy, output_dir):
                    self.logger.info(f"Skipping analysis for existing section: {title}")
                    self.completed_sections += 1
                    # Load existing analysis
                    existing = self._load_existing_dependency(level, hierarchy, output_dir)
                    if existing:
                        return existing
                
                # Perform new analysis
                self.logger.info(f"Analyzing section: {hierarchy}")
                result = await self.analyze_section(level, title, content, hierarchy, full_content)
                result.rule_hierarchy = hierarchy  # Overwrite hierarchy with actual value
                result.rule_text = content  # Overwrite rule text with actual value
                result.title = title  # Overwrite title with actual value
                # Write dependency file immediately after analysis
                await self._write_dependency_file(result, output_dir)
                
                return result
        
        # Create tasks for all sections
        tasks = [
            analyze_with_semaphore(level, title, content, hierarchy)
            for level, title, content, hierarchy in sections
        ]
        
        # Execute all tasks and gather results
        self.logger.info(f"Starting parallel analysis with max {concurrency_limit} concurrent requests")
        dependencies = await asyncio.gather(*tasks)
        self.logger.info(f"Completed analysis of {len(dependencies)} sections")
        
        return dependencies
    
    def save_results(self, dependencies: List[RuleDependency], output_dir: str):
        """Save analysis results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save detailed JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = output_path / f"dependencies_{timestamp}.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(
                [vars(d) for d in dependencies],
                f,
                indent=2,
                ensure_ascii=False
            )
        
        # Generate summary markdown
        markdown_path = output_path / f"dependency_summary_{timestamp}.md"
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write("# SRD Rule Dependencies Analysis\n\n")
            
            for dep in dependencies:
                f.write(f"## {dep.title}\n\n")
                f.write(f"**Section Level:** {dep.section}\n\n")
                f.write(f"**Hierarchy:** {dep.rule_hierarchy}\n\n")
                
                if dep.explicit_references:
                    f.write("### Explicit References\n")
                    for ref in dep.explicit_references:
                        f.write(f"- {ref}\n")
                    f.write("\n")
                
                if dep.implicit_dependencies:
                    f.write("### Implicit Dependencies\n")
                    for dep_rule in dep.implicit_dependencies:
                        f.write(f"- {dep_rule}\n")
                    f.write("\n")
                
                if dep.systems_affected:
                    f.write("### Affected Systems\n")
                    for system in dep.systems_affected:
                        f.write(f"- {system}\n")
                    f.write("\n")
                
                f.write("### Rule Text\n")
                f.write("```\n")
                f.write(dep.rule_text[:500] + "..." if len(dep.rule_text) > 500 else dep.rule_text)
                f.write("\n```\n\n")
                
                f.write("---\n\n")
        
        # Generate network visualization data
        self._generate_network_data(dependencies, output_path / f"network_data_{timestamp}.json")
        
        self.logger.info(f"Results saved to {output_path}")
    
    def _generate_network_data(self, dependencies: List[RuleDependency], output_file: str):
        """Generate network visualization data from dependencies."""
        nodes = []
        links = []
        
        # Create node map for faster lookup
        node_map = {}
        
        # Add nodes
        for i, dep in enumerate(dependencies):
            node_id = f"rule_{i}"
            node_map[dep.title] = node_id
            
            # Determine node size based on number of connections
            connections = len(dep.explicit_references) + len(dep.implicit_dependencies)
            
            nodes.append({
                "id": node_id,
                "name": dep.title,
                "group": dep.section,
                "hierarchy": dep.rule_hierarchy,
                "systems": dep.systems_affected,
                "size": min(20, 5 + connections)  # Base size + connections, capped at 20
            })
        
        # Add links
        for i, dep in enumerate(dependencies):
            source_id = f"rule_{i}"
            
            # Add explicit references
            for ref in dep.explicit_references:
                # Find target node by title
                for j, target_dep in enumerate(dependencies):
                    if ref.lower() in target_dep.title.lower():
                        target_id = f"rule_{j}"
                        links.append({
                            "source": source_id,
                            "target": target_id,
                            "value": 2,  # Stronger connection for explicit references
                            "type": "explicit"
                        })
                        break
            
            # Add implicit dependencies
            for imp in dep.implicit_dependencies:
                # Find target node by title
                for j, target_dep in enumerate(dependencies):
                    if imp.lower() in target_dep.title.lower():
                        target_id = f"rule_{j}"
                        links.append({
                            "source": source_id,
                            "target": target_id,
                            "value": 1,  # Weaker connection for implicit dependencies
                            "type": "implicit"
                        })
                        break
        
        # Save network data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "nodes": nodes,
                "links": links
            }, f, indent=2)
        
        self.logger.info(f"Network visualization data saved to {output_file}")


async def main():
    parser = argparse.ArgumentParser(description='Analyze SRD for rule dependencies')
    parser.add_argument('--srd', required=True, help='Path to SRD markdown file')
    parser.add_argument('--output', default='analysis_output', help='Output directory for summary files')
    parser.add_argument('--rule-output', default='rule_dependencies', help='Output directory for individual rule files')
    parser.add_argument('--concurrency', type=int, default=10, help='Maximum concurrent requests')
    parser.add_argument('--api-key', help='Gemini API key (or use GEMINI_API_KEY env var)')
    args = parser.parse_args()
    
    # Get API key from args or environment
    api_key = args.api_key or os.environ.get('GEMINI_API_KEY')
    
    if not api_key:
        raise ValueError("Please provide API key via --api-key or GEMINI_API_KEY environment variable")
    
    # Initialize analyzer
    analyzer = DependencyAnalyzer(api_key)
    
    # Analyze SRD with specified concurrency
    start_time = time.time()
    dependencies = await analyzer.analyze_srd(args.srd, args.concurrency, args.rule_output)
    duration = time.time() - start_time
    print(f"Analysis completed in {duration:.2f} seconds")
    
    # Save summary results
    analyzer.save_results(dependencies, args.output)
    
    print(f"Analysis summary saved to {args.output}")
    print(f"Individual rule files saved to {args.rule_output}")


if __name__ == "__main__":
    asyncio.run(main())