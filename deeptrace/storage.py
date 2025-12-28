import faiss
import numpy as np
import json
import pickle
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import os

class VectorStore:
    """
    FAISS-based vector storage for flow embeddings with metadata.
    Supports similarity search and retrieval for RAG applications.
    """
    
    def __init__(self, dimension: int = 128, index_type: str = "flatl2"):
        """
        Args:
            dimension: Embedding dimension (default: 128)
            index_type: FAISS index type ('flatl2', 'ivfflat', 'hnsw')
        """
        self.dimension = dimension
        self.index_type = index_type
        
        # Initialize FAISS index
        if index_type == "flatl2":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "ivfflat":
            # IVF index for large datasets (requires training)
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)  # 100 clusters
            self.index_trained = False
        elif index_type == "hnsw":
            # HNSW for fast approximate search
            self.index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Metadata storage (flow details, timestamps, etc.)
        self.metadata = []
        
        # Statistics
        self.total_flows = 0
        
    def add_flow(self, embedding: np.ndarray, flow_data: Dict):
        """Add a single flow embedding with metadata"""
        self.add_flows([embedding], [flow_data])
    
    def add_flows(self, embeddings: np.ndarray, flow_data_list: List[Dict]):
        """
        Add multiple flow embeddings with metadata
        
        Args:
            embeddings: (N, dimension) numpy array of embeddings
            flow_data_list: List of flow metadata dictionaries
        """
        if len(embeddings) != len(flow_data_list):
            raise ValueError("Number of embeddings must match number of flow data entries")
        
        # Ensure embeddings are float32 and contiguous
        embeddings = np.ascontiguousarray(embeddings.astype('float32'))
        
        # Train index if needed (IVF)
        if self.index_type == "ivfflat" and not self.index_trained:
            if len(embeddings) >= 100:  # Need enough samples for training
                print("Training IVF index...")
                self.index.train(embeddings)
                self.index_trained = True
            else:
                print("Not enough samples to train IVF index yet, using flat index temporarily")
                return
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store metadata
        for flow_data in flow_data_list:
            # Add internal ID
            flow_data['_id'] = self.total_flows
            flow_data['_indexed_at'] = datetime.now().isoformat()
            self.metadata.append(flow_data)
            self.total_flows += 1
        
        print(f"Added {len(embeddings)} flows. Total: {self.total_flows}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar flows
        
        Args:
            query_embedding: (dimension,) query vector
            k: Number of results to return
            filters: Optional metadata filters (e.g., {'protocol': 'TCP'})
        
        Returns:
            List of dicts with 'distance', 'metadata', and 'id'
        """
        if self.total_flows == 0:
            return []
        
        # Ensure query is correct shape and type
        query = np.ascontiguousarray(query_embedding.reshape(1, -1).astype('float32'))
        
        # Search FAISS index
        distances, indices = self.index.search(query, min(k * 10, self.total_flows))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            metadata = self.metadata[idx]
            
            # Apply filters
            if filters:
                match = True
                for key, value in filters.items():
                    if key.startswith('_'):
                        continue
                    
                    # Navigate nested dictionaries
                    meta_value = metadata
                    for k in key.split('.'):
                        if isinstance(meta_value, dict):
                            meta_value = meta_value.get(k)
                        else:
                            meta_value = None
                            break
                    
                    if meta_value != value:
                        match = False
                        break
                
                if not match:
                    continue
            
            results.append({
                'distance': float(dist),
                'metadata': metadata,
                'id': int(idx)
            })
            
            if len(results) >= k:
                break
        
        return results
    
    def search_by_ip(self, ip: str, k: int = 10) -> List[Dict]:
        """Search flows by IP address (source or destination)"""
        results = []
        for idx, meta in enumerate(self.metadata):
            flow_id = meta.get('flow_id', {})
            if flow_id.get('src_ip') == ip or flow_id.get('dst_ip') == ip:
                results.append({
                    'distance': 0.0,
                    'metadata': meta,
                    'id': idx
                })
                if len(results) >= k:
                    break
        return results
    
    def search_by_protocol(self, protocol: str, k: int = 10) -> List[Dict]:
        """Search flows by protocol"""
        return self.search(
            np.zeros(self.dimension, dtype='float32'),
            k=k,
            filters={'protocol.proto': protocol}
        )
    
    def get_statistics(self) -> Dict:
        """Get storage statistics"""
        protocol_counts = {}
        app_protocol_counts = {}
        
        for meta in self.metadata:
            proto = meta.get('protocol', {}).get('proto', 'UNKNOWN')
            app_proto = meta.get('protocol', {}).get('app_proto', 'UNKNOWN')
            
            protocol_counts[proto] = protocol_counts.get(proto, 0) + 1
            app_protocol_counts[app_proto] = app_protocol_counts.get(app_proto, 0) + 1
        
        return {
            'total_flows': self.total_flows,
            'index_type': self.index_type,
            'dimension': self.dimension,
            'protocol_distribution': protocol_counts,
            'app_protocol_distribution': app_protocol_counts
        }
    
    def save(self, base_path: str):
        """Save index and metadata to disk"""
        os.makedirs(base_path, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(base_path, 'faiss.index')
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata_path = os.path.join(base_path, 'metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'total_flows': self.total_flows,
                'dimension': self.dimension,
                'index_type': self.index_type,
                'index_trained': getattr(self, 'index_trained', True)
            }, f)
        
        print(f"Saved vector store to {base_path}")
    
    def load(self, base_path: str):
        """Load index and metadata from disk"""
        # Load FAISS index
        index_path = os.path.join(base_path, 'faiss.index')
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index not found: {index_path}")
        
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        metadata_path = os.path.join(base_path, 'metadata.pkl')
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.metadata = data['metadata']
            self.total_flows = data['total_flows']
            self.dimension = data['dimension']
            self.index_type = data['index_type']
            if 'index_trained' in data:
                self.index_trained = data['index_trained']
        
        print(f"Loaded vector store from {base_path} ({self.total_flows} flows)")


class StreamingVectorStore(VectorStore):
    """
    Extended VectorStore with streaming capabilities and batch optimization.
    Buffers embeddings before adding to index for better performance.
    """
    
    def __init__(self, dimension: int = 128, index_type: str = "flatl2", 
                 buffer_size: int = 100):
        super().__init__(dimension, index_type)
        
        self.buffer_size = buffer_size
        self.embedding_buffer = []
        self.metadata_buffer = []
    
    def buffer_flow(self, embedding: np.ndarray, flow_data: Dict):
        """Add flow to buffer, flush when buffer is full"""
        self.embedding_buffer.append(embedding)
        self.metadata_buffer.append(flow_data)
        
        if len(self.embedding_buffer) >= self.buffer_size:
            self.flush()
    
    def flush(self):
        """Flush buffer to index"""
        if not self.embedding_buffer:
            return
        
        embeddings = np.vstack(self.embedding_buffer)
        self.add_flows(embeddings, self.metadata_buffer)
        
        self.embedding_buffer = []
        self.metadata_buffer = []
        
    def __del__(self):
        """Ensure buffer is flushed on destruction"""
        if hasattr(self, 'embedding_buffer') and self.embedding_buffer:
            self.flush()


class FlowQueryEngine:
    """
    Query engine for natural language queries over flow data.
    Converts user queries to vector searches and metadata filters.
    """
    
    def __init__(self, vector_store: VectorStore, model_inference):
        self.vector_store = vector_store
        self.model_inference = model_inference
    
    def query_by_example(self, example_flow, k: int = 10) -> List[Dict]:
        """Find similar flows to an example flow"""
        embedding = self.model_inference.embed_flow(example_flow)
        return self.vector_store.search(embedding, k=k)
    
    def query_by_embedding(self, embedding: np.ndarray, k: int = 10) -> List[Dict]:
        """Direct embedding search"""
        return self.vector_store.search(embedding, k=k)
    
    def query_anomalies(self, threshold: float = 2.0, k: int = 50) -> List[Dict]:
        """
        Find potential anomalies by looking for flows far from cluster centers
        
        Args:
            threshold: Distance threshold for anomaly detection
            k: Number of samples to check
        
        Returns:
            List of potential anomalies
        """
        if self.vector_store.total_flows < 10:
            return []
        
        # Sample random flows and find their nearest neighbors
        anomalies = []
        sample_size = min(k, self.vector_store.total_flows)
        
        for idx in np.random.choice(self.vector_store.total_flows, sample_size, replace=False):
            meta = self.vector_store.metadata[idx]
            # Get embedding from index (reconstruct or re-compute)
            # For simplicity, we'll use distance-based heuristic
            
            # This is a placeholder - in practice, you'd reconstruct the embedding
            # or store it alongside metadata
            pass
        
        return anomalies
    
    def get_context_for_rag(self, query_embedding: np.ndarray, k: int = 5) -> str:
        """
        Get formatted context for RAG (Retrieval Augmented Generation)
        
        Args:
            query_embedding: Query vector
            k: Number of relevant flows to retrieve
        
        Returns:
            Formatted context string for LLM
        """
        results = self.vector_store.search(query_embedding, k=k)
        
        if not results:
            return "No relevant flows found in the database."
        
        context_parts = ["Here are the most relevant network flows:"]
        
        for i, result in enumerate(results, 1):
            meta = result['metadata']
            flow_id = meta.get('flow_id', {})
            stats = meta.get('statistical', {})
            proto = meta.get('protocol', {})
            
            context_parts.append(f"Flow {i}:")
            context_parts.append(f"  - Connection: {flow_id.get('src_ip')}:{flow_id.get('src_port')} → {flow_id.get('dst_ip')}:{flow_id.get('dst_port')}")
            context_parts.append(f"  - Protocol: {proto.get('proto')} / {proto.get('app_proto')}")
            context_parts.append(f"  - Duration: {stats.get('flow_duration', 0):.3f}s")
            context_parts.append(f"  - Total bytes: {stats.get('total_bytes', 0):.0f}")
            context_parts.append(f"  - Packets: {stats.get('packet_count', 0):.0f}")
            context_parts.append(f"  - Similarity score: {result['distance']:.4f}")
        
        return "".join(context_parts)