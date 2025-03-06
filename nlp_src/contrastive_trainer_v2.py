from datasets import load_dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import MultipleNegativesRankingLoss
import os
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("contrastive_trainer_v2")

def train(data_file: str, model_name: str, output_dir: str, batch_size: int = 32, epochs: int = 3):
    """
    Train a SentenceTransformer model on a dataset of question-positive pairs.
    
    Args:
        data_file: Path to JSON file containing list of dicts with 'question' and 'pos' keys
        model_name: Name of the SentenceTransformer model to fine-tune
        output_dir: Directory to save the trained model
        batch_size: Training batch size
        epochs: Number of training epochs
    """
    logger.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    logger.info(f"Loading dataset from: {data_file}")
    # Load dataset and create text pairs
    dataset = load_dataset("json", data_files=data_file)
    
    # Process dataset to create sentence pairs
    def process_qa_pairs(examples):
        sentence_pairs = []
        for question, pos_text in zip(examples['question'], examples['pos']):
            sentence_pairs.append([question, pos_text])
        return {'sentence_pairs': sentence_pairs}
    
    processed_dataset = dataset['train'].map(
        process_qa_pairs, 
        batched=True, 
        remove_columns=['question', 'pos']
    )
    
    # Split into train and test
    logger.info("Splitting dataset into train and test sets (90/10)")
    split_dataset = processed_dataset.train_test_split(test_size=0.1)
    
    # Define loss function
    logger.info("Creating MultipleNegativesRankingLoss")
    loss = MultipleNegativesRankingLoss(model)
    
    # Create trainer and train
    logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}")
    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        loss=loss,
        batch_size=batch_size,
        epochs=epochs,
    )
    
    trainer.train()
    
    # Save the trained model
    output_path = os.path.join(output_dir, model_name)
    logger.info(f"Saving model to: {output_path}")
    model.save_pretrained(output_path, safe_serialization=True)
    logger.info("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a SentenceTransformer model with contrastive learning")
    parser.add_argument("--data-file", type=str, required=True, help="Path to JSON file with QA pairs")
    parser.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--output-dir", type=str, default="model_output", help="Output directory for trained model")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    
    args = parser.parse_args()
    
    train(
        data_file=args.data_file,
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
