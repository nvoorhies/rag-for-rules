from datasets import load_dataset
from sentence_transformers.cross_encoder import CrossEncoder
from torch.utils.data import DataLoader
from sentence_transformers.readers import InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryAccuracyEvaluator


# 2. Format TRAINING data for CrossEncoder (triplets as tuples)
def format_for_training(examples):
    formatted_data = []
    
    # Create positive pairs with label 1
    for q, p in zip(examples['question'], examples['pos']):
        formatted_data.append((q, p, 1))
    
    # Create negative pairs with label 0
    for q, n in zip(examples['question'], examples['neg']):
        formatted_data.append((q, n, 0))
    
    return formatted_data

# 3. Format EVALUATION data (InputExample objects)
def format_for_evaluation(examples):
    eval_examples = []
    
    for idx, (q, p, n) in enumerate(zip(examples['question'], examples['pos'], examples['neg'])):
        # Positive pair - label 1
        eval_examples.append(InputExample(
            texts=[q, p],
            label=1,
            guid=f"pos-{idx}"
        ))
        
        # Negative pair - label 0
        eval_examples.append(InputExample(
            texts=[q, n],
            label=0,
            guid=f"neg-{idx}"
        ))
    
    return eval_examples

def train_cross_encoder(data_path: str, output_dir: str, batch_size: int, epochs: int, warmup_steps: int, evaluation_steps: int, model_name: str):
    dataset = load_dataset('json', data_files=data_path)

    full_dataset = dataset['train'] if 'train' in dataset else dataset
    #split_dataset = full_dataset.train_test_split(test_size=0.1)
    train_dataset = full_dataset #split_dataset['train']
    #test_dataset = split_dataset['test']



    # Apply the formatting
    train_examples = format_for_evaluation(train_dataset)
    #test_examples = format_for_evaluation(test_dataset)

    # Step 3: Create DataLoader
    #print(f"test_examples: {len(test_examples)} : {test_examples[0]}")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    evaluator = CEBinaryAccuracyEvaluator.from_input_examples
    # Step 4: Initialize and train the CrossEncoder
    model = CrossEncoder(model_name, num_labels=1)
    model.fit(
        train_dataloader=train_dataloader,
        epochs=epochs,
        warmup_steps=warmup_steps,
        evaluation_steps=0,
    )
    
    model.save(output_dir)
    print('Model saved to', output_dir)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('qa_triples', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--evaluation_steps', type=int, default=1000)
    parser.add_argument('--model_name', type=str, default='mixedbread-ai/mxbai-rerank-xsmall-v1')
    args = parser.parse_args()

    train_cross_encoder(args.qa_triples,
                        args.output_dir,
                        args.batch_size,
                        args.epochs,
                        args.warmup_steps,
                        args.evaluation_steps,
                        args.model_name)

if __name__ == '__main__':
    main()

