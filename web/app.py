#!/usr/bin/env python3
"""
DeepTrace Web API Server
Provides REST API and WebSocket endpoints for real-time network traffic dashboard
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import numpy as np

from deeptrace.storage import VectorStore
from deeptrace.model import ModelInference
from deeptrace.features import Flow
from deeptrace.capture import PacketCapture

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global instances
vector_store = None
model_inference = None
packet_capture = None

class WebAPI:
    """Web API for DeepTrace dashboard"""
    
    def __init__(self, store_path: str, checkpoint_path: Optional[str] = None):
        self.store_path = store_path
        self.checkpoint_path = checkpoint_path
        
        # Initialize components
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize vector store and model"""
        global vector_store, model_inference
        
        try:
            logger.info("Loading vector store...")
            vector_store = VectorStore(dimension=64)
            vector_store.load(self.store_path)
            logger.info(f"Loaded vector store with {vector_store.total_flows} flows")
            
            if self.checkpoint_path and os.path.exists(self.checkpoint_path):
                logger.info("Loading model...")
                model_inference = ModelInference(self.checkpoint_path, device='cpu')
                logger.info("Model loaded successfully")
            else:
                logger.warning("No model checkpoint provided, similarity search disabled")
                
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

# API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'flows_count': vector_store.total_flows if vector_store else 0,
        'model_loaded': model_inference is not None
    })

@app.route('/api/flows', methods=['GET'])
def get_flows():
    """Get paginated list of flows"""
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 20))
    
    start_idx = (page - 1) * limit
    end_idx = min(start_idx + limit, vector_store.total_flows)
    
    flows = []
    for idx in range(start_idx, end_idx):
        flow_data = vector_store.metadata[idx]
        flows.append({
            'id': idx,
            'data': flow_data
        })
    
    return jsonify({
        'flows': flows,
        'total': vector_store.total_flows,
        'page': page,
        'limit': limit,
        'pages': (vector_store.total_flows + limit - 1) // limit
    })

@app.route('/api/flows/<int:flow_id>', methods=['GET'])
def get_flow(flow_id: int):
    """Get detailed information for a specific flow"""
    if flow_id < 0 or flow_id >= vector_store.total_flows:
        return jsonify({'error': 'Flow ID out of range'}), 404
    
    flow_data = vector_store.metadata[flow_id]
    return jsonify({
        'id': flow_id,
        'data': flow_data
    })

@app.route('/api/search/ip/<ip_address>', methods=['GET'])
def search_by_ip(ip_address: str):
    """Search flows by IP address"""
    limit = int(request.args.get('limit', 10))
    
    results = []
    for idx, flow_data in enumerate(vector_store.metadata):
        flow_id = flow_data.get('flow_id', {})
        if flow_id.get('src_ip') == ip_address or flow_id.get('dst_ip') == ip_address:
            results.append({
                'id': idx,
                'data': flow_data,
                'distance': 0.0  # No distance metric for IP search
            })
            if len(results) >= limit:
                break
    
    return jsonify({
        'query': {'type': 'ip', 'value': ip_address},
        'results': results,
        'total': len(results)
    })

@app.route('/api/search/protocol/<protocol>', methods=['GET'])
def search_by_protocol(protocol: str):
    """Search flows by protocol"""
    limit = int(request.args.get('limit', 10))
    protocol = protocol.upper()
    
    results = []
    for idx, flow_data in enumerate(vector_store.metadata):
        proto = flow_data.get('protocol', {}).get('proto', '')
        app_proto = flow_data.get('protocol', {}).get('app_protocol', '')
        
        if proto == protocol or app_proto == protocol:
            results.append({
                'id': idx,
                'data': flow_data,
                'distance': 0.0
            })
            if len(results) >= limit:
                break
    
    return jsonify({
        'query': {'type': 'protocol', 'value': protocol},
        'results': results,
        'total': len(results)
    })

@app.route('/api/search/port/<int:port>', methods=['GET'])
def search_by_port(port: int):
    """Search flows by port number"""
    limit = int(request.args.get('limit', 10))
    
    results = []
    for idx, flow_data in enumerate(vector_store.metadata):
        flow_id = flow_data.get('flow_id', {})
        if flow_id.get('src_port') == port or flow_id.get('dst_port') == port:
            results.append({
                'id': idx,
                'data': flow_data,
                'distance': 0.0
            })
            if len(results) >= limit:
                break
    
    return jsonify({
        'query': {'type': 'port', 'value': port},
        'results': results,
        'total': len(results)
    })

@app.route('/api/similar/<int:flow_id>', methods=['GET'])
def find_similar(flow_id: int):
    """Find flows similar to a given flow"""
    if not model_inference:
        return jsonify({'error': 'Model not loaded. Provide checkpoint path.'}), 400
    
    if flow_id < 0 or flow_id >= vector_store.total_flows:
        return jsonify({'error': 'Flow ID out of range'}), 404
    
    limit = int(request.args.get('limit', 5))
    
    # Get reference flow
    reference_flow = vector_store.metadata[flow_id]
    
    # Reconstruct flow object for embedding
    try:
        flow_obj = reconstruct_flow(reference_flow)
        if not flow_obj:
            return jsonify({'error': 'Could not reconstruct flow'}), 500
        
        # Generate embedding
        embedding = model_inference.embed_flow(flow_obj)
        
        # Search for similar flows
        results = vector_store.search(embedding, k=limit + 1)  # +1 to exclude self
        
        similar_flows = []
        for result in results:
            result_idx = result['id']
            if result_idx == flow_id:
                continue  # Skip reference flow
            
            similar_flows.append({
                'id': result_idx,
                'data': result['metadata'],
                'distance': float(result['distance'])
            })
            
            if len(similar_flows) >= limit:
                break
        
        return jsonify({
            'reference_flow': {
                'id': flow_id,
                'data': reference_flow
            },
            'similar_flows': similar_flows,
            'total': len(similar_flows)
        })
        
    except Exception as e:
        logger.error(f"Error finding similar flows: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """Get vector store statistics"""
    stats = vector_store.get_statistics()
    
    # Calculate additional statistics
    total_bytes = sum(f.get('statistical', {}).get('total_bytes', 0) 
                     for f in vector_store.metadata)
    total_packets = sum(f.get('statistical', {}).get('packet_count', 0) 
                       for f in vector_store.metadata)
    
    stats['total_bytes'] = total_bytes
    stats['total_packets'] = total_packets
    stats['avg_bytes_per_flow'] = total_bytes / vector_store.total_flows if vector_store.total_flows > 0 else 0
    stats['avg_packets_per_flow'] = total_packets / vector_store.total_flows if vector_store.total_flows > 0 else 0
    
    return jsonify(stats)

@app.route('/api/anomalies', methods=['GET'])
def find_anomalies():
    """Find potential anomalies"""
    if not model_inference:
        return jsonify({'error': 'Model not loaded. Provide checkpoint path.'}), 400
    
    threshold = float(request.args.get('threshold', 2.0))
    limit = int(request.args.get('limit', 10))
    sample_size = min(100, vector_store.total_flows)
    
    anomalies = []
    
    for idx in range(sample_size):
        flow_data = vector_store.metadata[idx]
        flow_obj = reconstruct_flow(flow_data)
        
        if not flow_obj:
            continue
        
        # Get embedding and find nearest neighbor
        embedding = model_inference.embed_flow(flow_obj)
        results = vector_store.search(embedding, k=2)  # Self + 1 neighbor
        
        # Skip self and get distance to nearest neighbor
        for result in results:
            if result['id'] != idx:
                distance = result['distance']
                if distance > threshold:
                    anomalies.append({
                        'id': idx,
                        'data': flow_data,
                        'distance': float(distance)
                    })
                break
    
    # Sort by distance (most anomalous first)
    anomalies.sort(key=lambda x: x['distance'], reverse=True)
    
    return jsonify({
        'anomalies': anomalies[:limit],
        'threshold': threshold,
        'total_found': len(anomalies),
        'sample_size': sample_size
    })

# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {
        'flows_count': vector_store.total_flows if vector_store else 0,
        'model_loaded': model_inference is not None
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnect"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('start_capture')
def handle_start_capture(data):
    """Start packet capture"""
    interface = data.get('interface', 'wlo1')
    emit('capture_started', {'interface': interface})
    
    # Start capture in background
    asyncio.run(start_capture_async(interface))

@socketio.on('stop_capture')
def handle_stop_capture():
    """Stop packet capture"""
    global packet_capture
    if packet_capture:
        packet_capture.stop()
        emit('capture_stopped', {'message': 'Capture stopped'})

async def start_capture_async(interface: str):
    """Async packet capture"""
    global packet_capture
    
    try:
        packet_capture = PacketCapture(interface=interface)
        packet_capture.start()
        
        # Simulate real-time updates (replace with actual capture events)
        while packet_capture.running:
            await asyncio.sleep(2)
            
            # Emit mock capture stats
            emit('capture_update', {
                'packets_captured': 100,
                'flows_extracted': 10,
                'timestamp': datetime.now().isoformat()
            })
            
    except Exception as e:
        logger.error(f"Capture error: {e}")
        emit('capture_error', {'error': str(e)})

# Helper functions
def reconstruct_flow(flow_data: Dict) -> Optional[Flow]:
    """Reconstruct Flow object from metadata"""
    try:
        flow_id_tuple = (
            flow_data['flow_id']['src_ip'],
            flow_data['flow_id']['src_port'],
            flow_data['flow_id']['dst_ip'],
            flow_data['flow_id']['dst_port'],
            flow_data['flow_id']['protocol']
        )
        
        flow = Flow(flow_id_tuple)
        flow.temporal = flow_data.get('temporal', {})
        flow.statistical = flow_data.get('statistical', {})
        flow.protocol = flow_data.get('protocol', {})
        flow.timestamp = flow_data.get('timestamp', '')
        
        return flow
    except Exception as e:
        logger.error(f"Error reconstructing flow: {e}")
        return None

if __name__ == '__main__':
    # Initialize API
    store_path = os.getenv('STORE_PATH', './vector_store')
    checkpoint_path = os.getenv('CHECKPOINT_PATH')
    
    api = WebAPI(store_path, checkpoint_path)
    
    # Start server
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting DeepTrace Web API on {host}:{port}")
    socketio.run(app, host=host, port=port, debug=debug)