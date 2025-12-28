#!/usr/bin/env python3
"""
Simple Flask API for DeepTrace Web Dashboard
This is a simplified version that works without full DeepTrace dependencies
"""

import os
import json
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Mock data for development
def load_mock_data():
    """Load mock data from JSON files"""
    try:
        # Try to load from mock data directory
        mock_dir = os.path.join(os.path.dirname(__file__), 'mock_data')
        
        if not os.path.exists(mock_dir):
            os.makedirs(mock_dir)
            
            # Create sample mock data
            sample_flows = []
            for i in range(50):
                sample_flows.append({
                    'id': i,
                    'data': {
                        'flow_id': {
                            'src_ip': f'192.168.1.{i % 50 + 1}',
                            'src_port': 50000 + i,
                            'dst_ip': f'10.0.0.{i % 20 + 1}',
                            'dst_port': 80 if i % 3 == 0 else 443 if i % 3 == 1 else 53,
                            'protocol': 'TCP' if i % 3 == 0 else 'UDP'
                        },
                        'statistical': {
                            'total_bytes': 1000 + i * 100,
                            'packet_count': 10 + i % 20,
                            'flow_duration': 1.5 + i * 0.1,
                            'mean_pkt_size': 150 + i % 50,
                            'byte_ratio': 0.8 if i % 2 == 0 else 0.2
                        },
                        'protocol': {
                            'proto': 'TCP' if i % 3 == 0 else 'UDP',
                            'app_proto': 'HTTP' if i % 3 == 0 else 'HTTPS' if i % 3 == 1 else 'DNS'
                        },
                        'timestamp': datetime.now().isoformat()
                    }
                })
            
            # Save mock data
            with open(os.path.join(mock_dir, 'flows.json'), 'w') as f:
                json.dump(sample_flows, f)
            
            # Create mock statistics
            stats = {
                'total_flows': 50,
                'total_packets': 750,
                'total_bytes': 1250000,
                'avg_bytes_per_flow': 25000,
                'avg_packets_per_flow': 15,
                'protocol_distribution': {'TCP': 30, 'UDP': 20},
                'app_protocol_distribution': {'HTTP': 20, 'HTTPS': 15, 'DNS': 15}
            }
            
            with open(os.path.join(mock_dir, 'stats.json'), 'w') as f:
                json.dump(stats, f)
        
        # Load mock data
        with open(os.path.join(mock_dir, 'flows.json'), 'r') as f:
            flows = json.load(f)
        
        with open(os.path.join(mock_dir, 'stats.json'), 'r') as f:
            stats = json.load(f)
        
        return flows, stats
        
    except Exception as e:
        print(f"Error loading mock data: {e}")
        # Return empty data
        return [], {
            'total_flows': 0,
            'total_packets': 0,
            'total_bytes': 0,
            'avg_bytes_per_flow': 0,
            'avg_packets_per_flow': 0,
            'protocol_distribution': {},
            'app_protocol_distribution': {}
        }

# Load mock data
mock_flows, mock_stats = load_mock_data()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'flows_count': len(mock_flows),
        'model_loaded': False
    })

@app.route('/api/flows', methods=['GET'])
def get_flows():
    """Get paginated list of flows"""
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 20))
    
    start_idx = (page - 1) * limit
    end_idx = min(start_idx + limit, len(mock_flows))
    
    flows = mock_flows[start_idx:end_idx]
    
    return jsonify({
        'flows': flows,
        'total': len(mock_flows),
        'page': page,
        'limit': limit,
        'pages': (len(mock_flows) + limit - 1) // limit
    })

@app.route('/api/flows/<int:flow_id>', methods=['GET'])
def get_flow(flow_id: int):
    """Get detailed information for a specific flow"""
    if flow_id < 0 or flow_id >= len(mock_flows):
        return jsonify({'error': 'Flow ID out of range'}), 404
    
    return jsonify(mock_flows[flow_id])

@app.route('/api/search/ip/<ip_address>', methods=['GET'])
def search_by_ip(ip_address: str):
    """Search flows by IP address"""
    limit = int(request.args.get('limit', 10))
    
    results = []
    for flow in mock_flows:
        flow_id_data = flow['data'].get('flow_id', {})
        if flow_id_data.get('src_ip') == ip_address or flow_id_data.get('dst_ip') == ip_address:
            results.append(flow)
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
    for flow in mock_flows:
        proto_data = flow['data'].get('protocol', {})
        proto = proto_data.get('proto', '')
        app_proto = proto_data.get('app_proto', '')
        
        if proto == protocol or app_proto == protocol:
            results.append(flow)
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
    for flow in mock_flows:
        flow_id_data = flow['data'].get('flow_id', {})
        if flow_id_data.get('src_port') == port or flow_id_data.get('dst_port') == port:
            results.append(flow)
            if len(results) >= limit:
                break
    
    return jsonify({
        'query': {'type': 'port', 'value': port},
        'results': results,
        'total': len(results)
    })

@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """Get vector store statistics"""
    return jsonify(mock_stats)

@app.route('/api/similar/<int:flow_id>', methods=['GET'])
def find_similar(flow_id: int):
    """Find similar flows (mock implementation)"""
    if flow_id < 0 or flow_id >= len(mock_flows):
        return jsonify({'error': 'Flow ID out of range'}), 404
    
    limit = int(request.args.get('limit', 5))
    
    # Mock similar flows (just return some random flows)
    similar_flows = []
    for i in range(1, limit + 1):
        similar_id = (flow_id + i) % len(mock_flows)
        if similar_id != flow_id:
            similar_flows.append({
                **mock_flows[similar_id],
                'distance': 0.1 + i * 0.1  # Mock distance
            })
    
    return jsonify({
        'reference_flow': mock_flows[flow_id],
        'similar_flows': similar_flows,
        'total': len(similar_flows)
    })

@app.route('/api/anomalies', methods=['GET'])
def find_anomalies():
    """Find potential anomalies (mock implementation)"""
    threshold = float(request.args.get('threshold', 2.0))
    limit = int(request.args.get('limit', 10))
    
    # Mock anomalies (flows with high byte counts)
    anomalies = []
    for flow in mock_flows[:limit]:
        byte_count = flow['data']['statistical']['total_bytes']
        if byte_count > 5000:  # Arbitrary threshold
            anomalies.append({
                **flow,
                'distance': float(byte_count / 1000)  # Mock distance
            })
    
    return jsonify({
        'anomalies': anomalies,
        'threshold': threshold,
        'total_found': len(anomalies),
        'sample_size': len(mock_flows)
    })

if __name__ == '__main__':
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'True').lower() == 'true'
    
    print(f"Starting DeepTrace Mock API on {host}:{port}")
    print("Note: This is a mock API with sample data for development")
    print("Use the real app.py with proper dependencies for production")
    
    app.run(host=host, port=port, debug=debug)