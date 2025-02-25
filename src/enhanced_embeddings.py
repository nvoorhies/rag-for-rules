from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch import nn
import torch
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass
from torch.utils.data import DataLoader

@dataclass
class RuleContext:
    """Represents the structural context of a rule."""
    section_id: int
    total_sections: int
    affected_systems: List[bool]

class StructuralEmbeddingModel(nn.Module):
    """Custom model that combines text embeddings with structural information."""
    
    def __init__(
        self,
        base_embedding_dim: int,
        num_sections: int,
        num_systems: int,
        projection_dim: int = 256
    ):
        super().__init__()
        
        # Dimensions for different components
        self.base_embedding_dim = base_embedding_dim
        self.structural_dim = num_sections + num_systems
        
        # Projection layers for each component
        self.text_projection = nn.Sequential(
            nn.Linear(base_embedding_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU()
        )
        
        self.structural_projection = nn.Sequential(
            nn.Linear(self.structural_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU()
        )
        
        # Final combination layer
        self.combination_layer = nn.Sequential(
            nn.Linear(projection_dim * 2, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        # Learnable weights for combining components
        self.component_weights = nn.Parameter(torch.ones(2))
        self.softmax = nn.Softmax(dim=0)
    
    def forward(
        self,
        text_embedding: torch.Tensor,
        structural_features: torch.Tensor
    ) -> torch.Tensor:
        """Combine text and structural embeddings."""
        # Project each component
        text_projected = self.text_projection(text_embedding)
        structural_projected = self.structural_projection(structural_features)
        
        # Get learned weights
        weights = self.softmax(self.component_weights)
        
        # Weighted combination
        combined = torch.cat([
            weights[0] * text_projected,
            weights[1] * structural_projected
        ], dim=-1)
        
        # Final projection
        return self.combination_layer(combined)

class EnhancedEmbeddingModel:
    """Enhanced embedding model that incorporates structural information."""
    
    def __init__(
        self,
        base_model_name: str = "all-MiniLM-L6-v2",
        num_sections: int = 10,
        num_systems: int = 5,
        projection_dim: int = 256
    ):
        # Initialize base text embedding model
        self.base_model = SentenceTransformer(base_model_name)
        self.base_embedding_dim = self.base_model.get_sentence_embedding_dimension()
        
        # Initialize structural combination model
        self.structural_model = StructuralEmbeddingModel(
            base_embedding_dim=self.base_embedding_dim,
            num_sections=num_sections,
            num_systems=num_systems,
            projection_dim=projection_dim
        )
        
        # Add structural model to sentence transformer pipeline
        self.base_model.add_module('structural_combination', self.structural_model)
        
        self.num_sections = num_sections
        self.num_systems = num_systems
    
    def _create_structural_features(
        self,
        context: RuleContext
    ) -> torch.Tensor:
        """Convert structural context into feature tensor."""
        # Create one-hot encoding for section
        section_features = torch.zeros(self.num_sections)
        section_features[context.section_id] = 1.0
        
        # Add system flags
        system_features = torch.tensor(context.affected_systems)
        
        # Combine features
        return torch.cat([section_features, system_features])
    
    def prepare_training_data(
        self,
        positive_pairs: List[Tuple[str, str, RuleContext, RuleContext]],
        negative_pairs: List[Tuple[str, str, RuleContext, RuleContext]]
    ) -> DataLoader:
        """Prepare training data from positive and negative pairs."""
        training_examples = []
        
        # Process positive pairs
        for text1, text2, context1, context2 in positive_pairs:
            structural1 = self._create_structural_features(context1)
            structural2 = self._create_structural_features(context2)
            
            training_examples.append(InputExample(
                texts=[text1, text2],
                label=1.0,
                features={
                    'structural1': structural1,
                    'structural2': structural2
                }
            ))
        
        # Process negative pairs
        for text1, text2, context1, context2 in negative_pairs:
            structural1 = self._create_structural_features(context1)
            structural2 = self._create_structural_features(context2)
            
            training_examples.append(InputExample(
                texts=[text1, text2],
                label=0.0,
                features={
                    'structural1': structural1,
                    'structural2': structural2
                }
            ))
        
        return DataLoader(training_examples, batch_size=16, shuffle=True)
    
    def train_similarity(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader = None,
        epochs: int = 10,
        warmup_steps: int = 100
    ):
        """Train the enhanced similarity model."""
        # Define loss function
        train_loss = losses.CosineSimilarityLoss(self.base_model)
        
        # If evaluation data is provided, create evaluator
        evaluator = None
        if eval_dataloader is not None:
            evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
                eval_dataloader,
                batch_size=16
            )
        
        # Train the model
        self.base_model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=epochs,
            warmup_steps=warmup_steps,
            show_progress_bar=True
        )
    
    def encode_with_structure(
        self,
        texts: List[str],
        contexts: List[RuleContext]
    ) -> np.ndarray:
        """Generate enhanced embeddings for texts with structural context."""
        # Generate base text embeddings
        text_embeddings = self.base_model.encode(
            texts,
            convert_to_tensor=True
        )
        
        # Create structural features
        structural_features = torch.stack([
            self._create_structural_features(ctx)
            for ctx in contexts
        ])
        
        # Generate combined embeddings
        with torch.no_grad():
            enhanced_embeddings = self.structural_model(
                text_embeddings,
                structural_features
            )
        
        return enhanced_embeddings.cpu().numpy()

# Example usage showing how to train with ground truth data
def main():
    # Initialize enhanced embedding model
    model = EnhancedEmbeddingModel(
        num_sections=5,  # Example: 5 sections
        num_systems=3    # Example: 3 affected systems
    )
    
    # Example ground truth data
    positive_pairs = [
        (
            "Users must authenticate with 2FA",
            "All access requires two-factor authentication",
            RuleContext(section_id=0, total_sections=5, affected_systems=[1, 0, 1]),
            RuleContext(section_id=0, total_sections=5, affected_systems=[1, 0, 1])
        ),
        # Add more positive pairs...
    ]
    
    negative_pairs = [
        (
            "Users must authenticate with 2FA",
            "Database backups run daily at midnight",
            RuleContext(section_id=0, total_sections=5, affected_systems=[1, 0, 1]),
            RuleContext(section_id=2, total_sections=5, affected_systems=[0, 1, 0])
        ),
        # Add more negative pairs...
    ]
    
    # Prepare training data
    train_dataloader = model.prepare_training_data(positive_pairs, negative_pairs)
    
    # Train the model
    model.train_similarity(train_dataloader, epochs=5)
    
    # Example of generating enhanced embeddings
    texts = ["Users must authenticate with 2FA"]
    contexts = [RuleContext(section_id=0, total_sections=5, affected_systems=[1, 0, 1])]
    
    embeddings = model.encode_with_structure(texts, contexts)
    print("Enhanced embeddings shape:", embeddings.shape)

if __name__ == "__main__":
    main()