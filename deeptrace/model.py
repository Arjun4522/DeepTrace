import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
import os
import sys

# Register ModelConfig in __main__ namespace for unpickling compatibility
class ModelConfig:
    """Configuration class for model (compatibility with saved checkpoints)"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# Make ModelConfig available in __main__ for checkpoint loading
sys.modules['__main__'].ModelConfig = ModelConfig

class FlowEmbeddingModel(nn.Module):
    """
    Transformer-based model for generating 128-D embeddings from network flows.
    Handles three-tier features: Temporal, Statistical, and Protocol.
    """
    def __init__(
        self,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        max_seq_len=100,
        statistical_dim=10,
        num_protocols=5,
        num_app_protocols=10,
        dropout=0.1
    ):
        super(FlowEmbeddingModel, self).__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Temporal feature projection (4 features: dt, pkt_size, direction, entropy)
        self.temporal_projection = nn.Linear(4, d_model)
        
        # Positional encoding for temporal sequence
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # Transformer encoder for temporal features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Statistical features MLP
        self.statistical_mlp = nn.Sequential(
            nn.Linear(statistical_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Protocol embeddings
        self.protocol_embedding = nn.Embedding(num_protocols, 32)
        self.app_protocol_embedding = nn.Embedding(num_app_protocols, 32)
        
        # Final projection to 128-D embedding
        # Transformer output (d_model) + statistical (64) + protocol (32) + app_protocol (32)
        combined_dim = d_model + 64 + 32 + 32
        self.final_projection = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128)
        )
        
        # Protocol mappings
        self.protocol_map = {"TCP": 0, "UDP": 1, "ICMP": 2, "OTHER": 3}
        self.app_protocol_map = {
            "HTTP": 0, "HTTPS": 1, "SSH": 2, "FTP": 3, "SMTP": 4,
            "DNS": 5, "DHCP": 6, "NTP": 7, "UNKNOWN": 8
        }
        
    def forward(self, temporal_seq, statistical, protocol_ids, app_protocol_ids, seq_lengths):
        """
        Args:
            temporal_seq: (batch_size, max_seq_len, 4) - Temporal features
            statistical: (batch_size, statistical_dim) - Statistical features
            protocol_ids: (batch_size,) - Protocol indices
            app_protocol_ids: (batch_size,) - Application protocol indices
            seq_lengths: (batch_size,) - Actual sequence lengths for masking
        
        Returns:
            embeddings: (batch_size, 128) - Flow embeddings
        """
        batch_size = temporal_seq.size(0)
        
        # Process temporal features
        temporal_emb = self.temporal_projection(temporal_seq)  # (B, L, d_model)
        temporal_emb = self.pos_encoder(temporal_emb)
        
        # Create attention mask for padding
        mask = self._create_padding_mask(seq_lengths, self.max_seq_len, temporal_seq.device)
        
        # Transformer encoding
        temporal_encoded = self.transformer_encoder(temporal_emb, src_key_padding_mask=mask)
        
        # Global average pooling over temporal dimension
        temporal_pooled = temporal_encoded.mean(dim=1)  # (B, d_model)
        
        # Process statistical features
        statistical_emb = self.statistical_mlp(statistical)  # (B, 64)
        
        # Process protocol features
        protocol_emb = self.protocol_embedding(protocol_ids)  # (B, 32)
        app_protocol_emb = self.app_protocol_embedding(app_protocol_ids)  # (B, 32)
        
        # Concatenate all features
        combined = torch.cat([
            temporal_pooled,
            statistical_emb,
            protocol_emb,
            app_protocol_emb
        ], dim=1)
        
        # Final projection to 128-D
        embeddings = self.final_projection(combined)  # (B, 128)
        
        return embeddings
    
    def _create_padding_mask(self, seq_lengths, max_len, device):
        """Create padding mask for transformer (True = ignore)"""
        batch_size = len(seq_lengths)
        mask = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)
        mask = mask >= seq_lengths.unsqueeze(1)
        return mask


class PositionalEncoding(nn.Module):
    """Positional encoding for temporal sequences"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SimpleFlowEmbeddingModel(nn.Module):
    """
    Simple Transformer model matching the trained checkpoint architecture.
    This matches the model trained in your notebook.
    """
    def __init__(self, input_dim=12, hidden_dim=64, output_dim=64, 
                 num_layers=2, num_heads=4, dropout=0.1):
        super(SimpleFlowEmbeddingModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection with normalization
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, input_dim) - Input features
            mask: Optional attention mask
        
        Returns:
            embeddings: (batch_size, output_dim) - Flow embeddings
        """
        # Project input
        x = self.input_proj(x)
        x = self.input_norm(x)
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Output projection with normalization
        x = self.output_norm(x)
        embeddings = self.output_proj(x)
        
        return embeddings


class ModelConfig:
    """Configuration class for model (compatibility with saved checkpoints)"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class ModelInference:
    """Wrapper for loading and running inference with trained model"""
    
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = None
        self.checkpoint_path = checkpoint_path
        self.load_model()
        
    def load_model(self):
        """Load model from checkpoint"""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        print(f"Loading model from {self.checkpoint_path}...")
        
        # Use weights_only=False for checkpoints with custom classes
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # Extract model configuration if available
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"Found model config:")
            print(f"  input_dim: {config.input_dim}")
            print(f"  hidden_dim: {config.hidden_dim}")
            print(f"  output_dim: {config.output_dim}")
            print(f"  num_layers: {config.num_layers}")
            print(f"  num_heads: {config.num_heads}")
            
            # Initialize model with config parameters
            self.model = SimpleFlowEmbeddingModel(
                input_dim=config.input_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.output_dim,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
            self.embedding_dim = config.output_dim
        else:
            # Initialize model with default architecture
            print("No config found, using default architecture")
            self.model = SimpleFlowEmbeddingModel()
            self.embedding_dim = 64
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            # Assume checkpoint is the state dict itself
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        print("✓ Model loaded successfully!")
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Parameters: {total_params:,}")
        print(f"  Embedding dimension: {self.embedding_dim}")

        
    def prepare_flow_data(self, flow) -> Dict[str, torch.Tensor]:
        """
        Convert flow object to model input tensors.
        Creates a flattened 12-D feature vector from statistical features.
        """
        # Extract the 12 statistical features (input_dim=12 from config)
        # Based on typical flow features: duration, bytes, packets, sizes, timings, entropy, ratio
        features = [
            flow.statistical.get('flow_duration', 0.0),
            flow.statistical.get('total_bytes', 0.0),
            flow.statistical.get('packet_count', 0.0),
            flow.statistical.get('mean_pkt_size', 0.0),
            flow.statistical.get('std_pkt_size', 0.0),
            flow.statistical.get('mean_dt', 0.0),
            flow.statistical.get('std_dt', 0.0),
            flow.statistical.get('entropy_mean', 0.0),
            flow.statistical.get('entropy_std', 0.0),
            flow.statistical.get('byte_ratio', 0.0),
            # Add 2 more features to make it 12-D (protocol one-hot or similar)
            1.0 if flow.protocol.get('proto') == 'TCP' else 0.0,
            1.0 if flow.protocol.get('proto') == 'UDP' else 0.0,
        ]
        
        # Create tensor (batch_size=1, seq_len=1, features=12)
        # The model expects (batch, seq, features), so we add a sequence dimension
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        return {
            'features': features_tensor.to(self.device)
        }
    
    @torch.no_grad()
    def embed_flow(self, flow) -> np.ndarray:
        """Generate embedding for a single flow"""
        inputs = self.prepare_flow_data(flow)
        embedding = self.model(inputs['features'])
        return embedding.cpu().numpy()[0]
    
    @torch.no_grad()
    def embed_flows_batch(self, flows: List) -> np.ndarray:
        """Generate embeddings for multiple flows (batched)"""
        if not flows:
            return np.array([])
        
        # Prepare all features
        all_features = []
        for flow in flows:
            inputs = self.prepare_flow_data(flow)
            all_features.append(inputs['features'])
        
        # Stack into single batch (remove the extra sequence dimension first)
        # Each flow has shape (1, 1, 12), we want (batch_size, 1, 12)
        batch_features = torch.cat(all_features, dim=0)  # (batch_size, 1, 12)
        
        # Get embeddings
        embeddings = self.model(batch_features)  # (batch_size, output_dim)
        
        return embeddings.cpu().numpy()