from datasets import load_dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import MultipleNegativesRankingLoss
import os



def train(data_files: str, field: str, model_name: str, output_dir: str):
    """Train a SentenceTransformer model on a dataset."""
    # Load a model to finetune
    model = SentenceTransformer(model_name)
    # 2. Load a dataset to finetune on
    dataset = load_dataset("json", data_files=data_files, field=field)
    dataset.train_test_split(test_size=0.1)
    

# 3. Define a loss function
    loss = MultipleNegativesRankingLoss(model)

# 4. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        loss=loss,
       )
    trainer.train()

# 5. Save the trained model
    model.save_pretrained(os.path.join(output_dir, model_name), safe_serialization=True)

if __name__ == "__main__":
    train(data_files="tmp/training_triplets.json", field="data", model_name="all-MiniLM-L6-v2", output_dir="tmp")