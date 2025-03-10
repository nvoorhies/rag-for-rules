import json
import torch
import argparse
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from typing import List, Dict, Any

"""
SentenceTransformer Fine-Tuning Script

This script allows you to fine-tune a SentenceTransformer model using either
Multiple Negatives Ranking Loss or TripletLoss on your custom dataset.

Example JSON data format:
[
    {
        "question": "what is love?",
        "pos": "baby don't hurt me",
        "neg": "no more"
    },
    ...
]

Usage:
    # Basic usage with default parameters (Multiple Negatives Ranking Loss)
    python finetune_embedding_model.py --data your_data.json
    
    # Using TripletLoss with custom parameters
    python finetune_embedding_model.py --data your_data.json --use-triplets --margin 0.7 --epochs 20
    
    # Using a different base model
    python finetune_embedding_model.py --data your_data.json --model all-mpnet-base-v2
    
    # Skip evaluation after training
    python finetune_embedding_model.py --data your_data.json --skip-eval
"""

# 1. Load and prepare your data
def load_training_data(json_file_path: str, use_triplets: bool = False) -> List[InputExample]:
    """
    Load training data from JSON file and convert to InputExample format
    
    Args:
        json_file_path: Path to JSON data file
        use_triplets: If True, create triplet examples for TripletLoss
                      If False, create pair examples for MultipleNegativesRankingLoss
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    examples = []
    
    for item in data:
        question = item['question']
        positive = item['pos']
        negative = item['neg']
        
        if use_triplets:
            # For TripletLoss: specify anchor, positive, and negative
            examples.append(InputExample(texts=[question, positive, negative]))
        else:
            # For MultipleNegativesRankingLoss: only need query and positive
            examples.append(InputExample(texts=[question, positive]))
    
    return examples

# 2. Initialize the base model
def initialize_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Initialize a pre-trained SentenceTransformer model
    """
    model = SentenceTransformer(model_name)
    return model

# 3. Train the model
def train_model(
    model: SentenceTransformer, 
    train_examples: List[InputExample],
    output_path: str,
    use_triplets: bool = False,
    num_epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_steps: int = 100,
    triplet_margin: float = 0.5
):
    """
    Fine-tune the model using either TripletLoss or Multiple Negatives Ranking Loss
    
    Args:
        model: SentenceTransformer model to train
        train_examples: List of InputExample objects
        output_path: Where to save the model
        use_triplets: If True, use TripletLoss; otherwise use MultipleNegativesRankingLoss
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps for learning rate scheduler
        triplet_margin: Margin for TripletLoss
    """
    # Create a DataLoader for our training data
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    # Select the appropriate loss function
    if use_triplets:
        # Use TripletLoss for (anchor, positive, negative) triplets
        train_loss = losses.TripletLoss(model=model, triplet_margin=triplet_margin)
        print(f"Training with TripletLoss (margin={triplet_margin})")
    else:
        # Use Multiple Negatives Ranking Loss
        # This will treat other positive examples in the batch as negatives
        train_loss = losses.MultipleNegativesRankingLoss(model)
        print("Training with MultipleNegativesRankingLoss")
    
    # Configure the training
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) if warmup_steps is None else warmup_steps
    
    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': learning_rate},
        output_path=output_path,
        show_progress_bar=True
    )
    
    return model

# 4. Evaluate the model (optional)
def evaluate_model(model: SentenceTransformer, test_data: List[Dict[str, Any]], verbose: bool = True):
    """
    Simple evaluation to check if positive examples are ranked higher than negative examples
    """
    results = []
    
    for item in test_data:
        question = item['question']
        positive = item['pos']
        negative = item['neg']
        
        # Encode the sentences
        embeddings = model.encode([question, positive, negative])
        
        # Calculate cosine similarities
        query_embedding = embeddings[0]
        pos_embedding = embeddings[1]
        neg_embedding = embeddings[2]
        
        pos_similarity = torch.nn.functional.cosine_similarity(
            torch.tensor(query_embedding).unsqueeze(0),
            torch.tensor(pos_embedding).unsqueeze(0)
        ).item()
        
        neg_similarity = torch.nn.functional.cosine_similarity(
            torch.tensor(query_embedding).unsqueeze(0),
            torch.tensor(neg_embedding).unsqueeze(0)
        ).item()
        
        results.append({
            'question': question,
            'positive': positive,
            'negative': negative,
            'positive_similarity': pos_similarity,
            'negative_similarity': neg_similarity,
            'ranking_correct': pos_similarity > neg_similarity
        })
    
    # Calculate how many times the positive example was ranked higher than the negative
    correct_rankings = sum(1 for r in results if r['ranking_correct'])
    accuracy = correct_rankings / len(results) if results else 0
    
    if verbose:
        print(f"Evaluation Results:")
        print(f"Accuracy: {accuracy:.4f} ({correct_rankings}/{len(results)})")
        
        # Print a few examples
        print("\nExample results:")
        for i, result in enumerate(results[:5]):  # Show first 5 examples
            print(f"Example {i+1}:")
            print(f"  Question: '{result['question']}'")
            print(f"  Positive: '{result['positive']}' (similarity: {result['positive_similarity']:.4f})")
            print(f"  Negative: '{result['negative']}' (similarity: {result['negative_similarity']:.4f})")
            print(f"  Correctly ranked: {result['ranking_correct']}")
    
    return results

# Main function to run the entire pipeline
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fine-tune a SentenceTransformer model")
    
    # Data and model parameters
    parser.add_argument("--data", type=str, default="your_data.json", 
                        help="Path to your JSON data file")
    parser.add_argument("--output", type=str, default="./finetuned-sentence-transformer", 
                        help="Where to save the model")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", 
                        help="Base model to fine-tune")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=10, 
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, 
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-5, 
                        help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100, 
                        help="Number of warmup steps for learning rate scheduler")
    
    # Loss function parameters
    parser.add_argument("--use-triplets", action="store_true", 
                        help="Use TripletLoss instead of MultipleNegativesRankingLoss")
    parser.add_argument("--margin", type=float, default=0.5, 
                        help="Margin for TripletLoss")
    
    # Evaluation
    parser.add_argument("--skip-eval", action="store_true", 
                        help="Skip evaluation after training")
    
    args = parser.parse_args()
    
    # Load training data
    print(f"Loading training data from {args.data}...")
    train_examples = load_training_data(args.data, use_triplets=args.use_triplets)
    print(f"Loaded {len(train_examples)} training examples.")
    
    # Initialize the model
    print(f"Initializing model '{args.model}'...")
    model = initialize_model(args.model)
    
    # Train the model
    print("Starting training...")
    model = train_model(
        model=model,
        train_examples=train_examples,
        output_path=args.output,
        use_triplets=args.use_triplets,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        triplet_margin=args.margin
    )
    print(f"Model trained and saved to '{args.output}'")
    
    # Optional: Evaluate the model
    if not args.skip_eval:
        with open(args.data, 'r') as f:
            test_data = json.load(f)
        
        print("Evaluating model...")
        evaluation_results = evaluate_model(model, test_data)

if __name__ == "__main__":
    main()
