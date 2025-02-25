# RPG Rules RAG System

A semantically-enhanced retrieval-augmented generation (RAG) system for tabletop RPG rules, designed to return minimal but complete rule sets needed to make game rulings.

## Overview

This system is designed to extract, organize, and retrieve tabletop RPG rules from a System Reference Document (SRD) in a way that:

1. Understands the semantic structure of the rules
2. Identifies dependencies between rules
3. Recognizes game-specific terminology
4. Returns the minimal set of rules needed to make a ruling

It's particularly suited for complex rule systems like Dungeons & Dragons 5th Edition, where rules often reference each other and rely on specific definitions.

## Components

The system consists of several key components:

### 1. SRD Processor (`srd_processor.py`)

Processes a markdown SRD file to extract semantic structure:
- Parses headings and content hierarchically
- Identifies rule boundaries and types (core, derived, exceptions, etc.)
- Extracts game-specific terminology and definitions
- Detects cross-references between rules
- Builds a knowledge graph of rule relationships

### 2. Rules RAG System (`rules_rag.py`)

Implements the retrieval-augmented generation system:
- Creates embeddings for rules
- Processes queries to extract relevant features
- Retrieves candidate rules based on semantic similarity
- Identifies dependencies and prerequisites
- Filters and minimizes rule sets to provide minimal but complete responses

### 3. Graph Visualizer (`rule_graph_visualizer.py`)

Visualizes the rule relationships:
- Creates interactive network graphs
- Generates distribution charts
- Builds query-specific subgraphs
- Creates dependency trees

### 4. Web Interface (`web_interface.py`)

Provides a user-friendly interface:
- Simple query input
- Formatted rule display
- System statistics
- Interactive results

### 5. System Integration (`rpg_rag.py`)

Ties everything together with a command-line interface:
- Process SRD files
- Query the processed rules
- Serve the web interface
- Generate visualizations

## Installation

### Prerequisites

```
pip install -r requirements.txt
```

Required packages:
- nltk
- spacy
- markdown
- beautifulsoup4
- sentence-transformers
- networkx
- matplotlib
- pyvis
- flask

You'll also need to download the spaCy English model:

```
python -m spacy download en_core_web_sm
```

## Usage

### 1. Process an SRD

```
python rpg_rag.py process --input dnd5e_srd.md --output dnd5e_processed.json
```

Options:
- `--visualize` / `-v`: Generate visualizations after processing

### 2. Query the Rules

```
python rpg_rag.py query --input dnd5e_processed.json --query "How does sneak attack work with advantage?"
```

Options:
- `--max-rules` / `-m`: Maximum number of rules to return (default: 10)
- `--visualize` / `-v`: Generate a visualization of query results

### 3. Start the Web Interface

```
python rpg_rag.py serve --input dnd5e_processed.json
```

Then open your browser to http://localhost:5000

### 4. Generate Visualizations

```
python rpg_rag.py visualize --input dnd5e_processed.json --output-dir visualizations
```

## Examples

### Example Query

```
python rpg_rag.py query --input dnd5e_processed.json --query "Can a wizard cast ritual spells without preparing them first?"
```

### Example Rule Relationships

The system identifies various relationships between rules:
- Explicit references ("see chapter X")
- Term dependencies (a rule using a term defined elsewhere)
- Prerequisites (rules needed to understand other rules)
- Exceptions (rules that override other rules)

## Evaluation

The effectiveness of this system can be evaluated on:

1. **Completeness**: Do the returned rules contain all necessary information?
2. **Minimality**: Are there unnecessary or redundant rules included?
3. **Structure**: Are the rules presented in a logical order?
4. **Query Understanding**: Does the system correctly identify the relevant domain?

## Future Improvements

- Fine-tuning of embeddings specifically for RPG rules
- Additional relationship types between rules
- Rule templating for common query patterns
- Integration with LLMs for natural language explanations
- Support for multiple rule systems (Pathfinder, GURPS, etc.)

## License

MIT

## Acknowledgments

- The system was designed for educational and research purposes.
- If using with published RPG content, ensure you have the appropriate rights and comply with the publisher's licensing terms.
