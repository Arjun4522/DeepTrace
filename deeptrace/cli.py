#!/usr/bin/env python3
"""
Interactive Query Engine for DeepTrace Vector Store
Search and analyze captured network flows using embeddings.
"""

import argparse
import json
import sys
from datetime import datetime
from typing import List, Dict, Optional, Union
import numpy as np

from deeptrace.storage import VectorStore
from deeptrace.model import ModelInference
from deeptrace.features import Flow


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class QueryEngine:
    """Interactive query engine for vector store"""
    
    def __init__(self, store_path: str, checkpoint_path: Optional[str] = None):
        """
        Initialize query engine
        
        Args:
            store_path: Path to vector store directory
            checkpoint_path: Optional path to model for generating query embeddings
        """
        print(f"{Colors.BLUE}Loading vector store...{Colors.ENDC}")
        self.store = VectorStore(dimension=64)
        self.store.load(store_path)
        
        self.model = None
        if checkpoint_path:
            print(f"{Colors.BLUE}Loading model...{Colors.ENDC}")
            self.model = ModelInference(checkpoint_path, device='cpu')
        
        print(f"{Colors.GREEN}✓ Ready!{Colors.ENDC}\n")
        
    def print_flow(self, flow_data: Dict, index: Optional[int] = None, distance: Optional[float] = None, verbose: bool = False):
        """Pretty print a flow with optional verbose output"""
        flow_id = flow_data.get('flow_id', {})
        temporal = flow_data.get('temporal', {})
        stats = flow_data.get('statistical', {})
        proto = flow_data.get('protocol', {})
        timestamp = flow_data.get('timestamp', 'Unknown')
        
        # Header
        if index is not None:
            print(f"\n{Colors.BOLD}{Colors.CYAN}{'─'*70}")
            if distance is not None:
                print(f"Flow #{index} (Distance: {distance:.4f})")
            else:
                print(f"Flow #{index}")
            print(f"{'─'*70}{Colors.ENDC}")
        
        # Connection info
        src_ip = flow_id.get('src_ip', 'N/A')
        src_port = flow_id.get('src_port', 'N/A')
        dst_ip = flow_id.get('dst_ip', 'N/A')
        dst_port = flow_id.get('dst_port', 'N/A')
        
        print(f"{Colors.BOLD}Connection:{Colors.ENDC}")
        print(f"  {src_ip}:{src_port} → {dst_ip}:{dst_port}")
        print(f"  Protocol: {flow_id.get('protocol', 'N/A')}")
        
        # Protocol info
        print(f"\n{Colors.BOLD}Protocol:{Colors.ENDC}")
        print(f"  Transport: {proto.get('proto', 'N/A')}")
        print(f"  Application: {proto.get('app_proto', 'N/A')}")
        if proto.get('flags_summary') and proto.get('flags_summary') != 'NONE':
            print(f"  TCP Flags: {proto.get('flags_summary')}")
        
        # Statistics
        print(f"\n{Colors.BOLD}Statistics:{Colors.ENDC}")
        print(f"  Duration:     {stats.get('flow_duration', 0):.3f}s")
        print(f"  Total bytes:  {stats.get('total_bytes', 0):,.0f}")
        print(f"  Packets:      {stats.get('packet_count', 0):.0f}")
        print(f"  Avg pkt size: {stats.get('mean_pkt_size', 0):.1f} bytes")
        print(f"  Std pkt size: {stats.get('std_pkt_size', 0):.1f} bytes")
        print(f"  Mean Δt:      {stats.get('mean_Δt', 0):.3f}s")
        print(f"  Std Δt:       {stats.get('std_Δt', 0):.3f}s")
        print(f"  Mean entropy: {stats.get('mean_entropy', 0):.3f}")
        print(f"  Std entropy:  {stats.get('std_entropy', 0):.3f}")
        print(f"  Byte ratio:   {stats.get('byte_ratio', 0):.2f}")
        
        # Temporal features (verbose mode)
        if verbose:
            print(f"\n{Colors.BOLD}Temporal Features:{Colors.ENDC}")
            for feature_type in ['Δt', 'size', 'direction', 'entropy']:
                values = temporal.get(feature_type, [])
                if values:
                    print(f"  {feature_type}: {values[:10]}..." if len(values) > 10 else f"  {feature_type}: {values}")
        
        # System metadata
        print(f"\n{Colors.BOLD}System:{Colors.ENDC}")
        print(f"  Captured:    {timestamp}")
        print(f"  Indexed at:  {flow_data.get('_indexed_at', 'N/A')}")
        print(f"  Internal ID: {flow_data.get('_id', 'N/A')}")
    
    def list_all(self, limit: int = 10):
        """List all flows in the store"""
        total = self.store.total_flows
        print(f"\n{Colors.BOLD}Total flows in store: {total:,}{Colors.ENDC}\n")
        
        if total == 0:
            print("No flows in store.")
            return
        
        print(f"Showing first {min(limit, total)} flows:\n")
        
        for i in range(min(limit, total)):
            flow_data = self.store.metadata[i]
            flow_id = flow_data.get('flow_id', {})
            proto = flow_data.get('protocol', {})
            stats = flow_data.get('statistical', {})
            
            src = f"{flow_id.get('src_ip', 'N/A')}:{flow_id.get('src_port', 'N/A')}"
            dst = f"{flow_id.get('dst_ip', 'N/A')}:{flow_id.get('dst_port', 'N/A')}"
            
            print(f"{Colors.CYAN}{i:4d}.{Colors.ENDC} {src:30s} → {dst:30s} "
                  f"[{proto.get('proto', 'N/A'):4s}] "
                  f"{stats.get('total_bytes', 0):8,.0f}B "
                  f"{stats.get('packet_count', 0):4.0f}pkts")
        
        if total > limit:
            print(f"\n... and {total - limit} more flows")
    
    def show_flow(self, index: int, verbose: bool = False):
        """Show detailed information for a specific flow"""
        if index < 0 or index >= self.store.total_flows:
            print(f"{Colors.RED}Error: Flow index out of range (0-{self.store.total_flows-1}){Colors.ENDC}")
            return
        
        flow_data = self.store.metadata[index]
        self.print_flow(flow_data, index=index, verbose=verbose)
    
    def search_by_ip(self, ip: str, k: int = 10):
        """Search flows by IP address"""
        print(f"\n{Colors.BOLD}Searching for flows involving IP: {ip}{Colors.ENDC}\n")
        
        results = []
        for idx, flow_data in enumerate(self.store.metadata):
            flow_id = flow_data.get('flow_id', {})
            if flow_id.get('src_ip') == ip or flow_id.get('dst_ip') == ip:
                results.append((idx, flow_data))
                if len(results) >= k:
                    break
        
        if not results:
            print(f"{Colors.YELLOW}No flows found for IP {ip}{Colors.ENDC}")
            return
        
        print(f"Found {len(results)} flow(s):\n")
        for idx, flow_data in results:
            self.print_flow(flow_data, index=idx)
    
    def search_by_protocol(self, protocol: str, k: int = 10):
        """Search flows by protocol"""
        protocol = protocol.upper()
        print(f"\n{Colors.BOLD}Searching for {protocol} flows...{Colors.ENDC}\n")
        
        results = []
        for idx, flow_data in enumerate(self.store.metadata):
            proto = flow_data.get('protocol', {}).get('proto', '')
            app_proto = flow_data.get('protocol', {}).get('app_proto', '')
            
            if proto == protocol or app_proto == protocol:
                results.append((idx, flow_data))
                if len(results) >= k:
                    break
        
        if not results:
            print(f"{Colors.YELLOW}No {protocol} flows found{Colors.ENDC}")
            return
        
        print(f"Found {len(results)} flow(s):\n")
        for idx, flow_data in results:
            self.print_flow(flow_data, index=idx)
    
    def search_by_port(self, port: int, k: int = 10):
        """Search flows by port number"""
        print(f"\n{Colors.BOLD}Searching for flows on port {port}...{Colors.ENDC}\n")
        
        results = []
        for idx, flow_data in enumerate(self.store.metadata):
            flow_id = flow_data.get('flow_id', {})
            if flow_id.get('src_port') == port or flow_id.get('dst_port') == port:
                results.append((idx, flow_data))
                if len(results) >= k:
                    break
        
        if not results:
            print(f"{Colors.YELLOW}No flows found on port {port}{Colors.ENDC}")
            return
        
        print(f"Found {len(results)} flow(s):\n")
        for idx, flow_data in results:
            self.print_flow(flow_data, index=idx)
    
    def find_similar(self, index: int, k: int = 5):
        """Find flows similar to a given flow"""
        if index < 0 or index >= self.store.total_flows:
            print(f"{Colors.RED}Error: Flow index out of range (0-{self.store.total_flows-1}){Colors.ENDC}")
            return
        
        print(f"\n{Colors.BOLD}Finding flows similar to flow #{index}...{Colors.ENDC}")
        
        # Get the reference flow
        reference_flow = self.store.metadata[index]
        self.print_flow(reference_flow, index=index)
        
        # To find similar flows, we need to get the embedding
        # Since we don't store embeddings separately, we'll reconstruct it
        if not self.model:
            print(f"\n{Colors.RED}Error: Model required for similarity search. "
                  f"Provide --checkpoint argument.{Colors.ENDC}")
            return
        
        # Reconstruct flow object for embedding generation
        flow_obj = self._reconstruct_flow(reference_flow)
        if not flow_obj:
            print(f"{Colors.RED}Error: Could not reconstruct flow{Colors.ENDC}")
            return
        
        # Generate embedding
        embedding = self.model.embed_flow(flow_obj)
        
        # Search for similar flows
        results = self.store.search(embedding, k=k+1)  # +1 to exclude self
        
        print(f"\n{Colors.BOLD}Top {k} most similar flows:{Colors.ENDC}")
        
        count = 0
        for result in results:
            result_idx = result['id']
            # Skip the reference flow itself
            if result_idx == index:
                continue
            
            count += 1
            if count > k:
                break
            
            flow_data = result['metadata']
            distance = result['distance']
            self.print_flow(flow_data, index=result_idx, distance=distance)
    
    def find_anomalies(self, threshold: float = 2.0, k: int = 10):
        """Find potential anomalies (flows far from their neighbors)"""
        if not self.model:
            print(f"{Colors.RED}Error: Model required for anomaly detection. "
                  f"Provide --checkpoint argument.{Colors.ENDC}")
            return
        
        print(f"\n{Colors.BOLD}Finding anomalies (distance threshold: {threshold})...{Colors.ENDC}\n")
        
        anomalies = []
        sample_size = min(100, self.store.total_flows)
        
        print(f"Analyzing {sample_size} flows...")
        
        for idx in range(sample_size):
            flow_data = self.store.metadata[idx]
            flow_obj = self._reconstruct_flow(flow_data)
            
            if not flow_obj:
                continue
            
            # Get embedding and find nearest neighbor
            embedding = self.model.embed_flow(flow_obj)
            results = self.store.search(embedding, k=2)  # Self + 1 neighbor
            
            # Skip self and get distance to nearest neighbor
            for result in results:
                if result['id'] != idx:
                    distance = result['distance']
                    if distance > threshold:
                        anomalies.append((idx, flow_data, distance))
                    break
        
        if not anomalies:
            print(f"{Colors.GREEN}No anomalies found with threshold {threshold}{Colors.ENDC}")
            return
        
        # Sort by distance (most anomalous first)
        anomalies.sort(key=lambda x: x[2], reverse=True)
        
        print(f"\n{Colors.BOLD}Found {len(anomalies)} potential anomalies:{Colors.ENDC}")
        
        for idx, flow_data, distance in anomalies[:k]:
            self.print_flow(flow_data, index=idx, distance=distance)
    
    def show_statistics(self):
        """Show statistics about the vector store"""
        stats = self.store.get_statistics()
        
        print(f"\n{Colors.BOLD}{'='*70}")
        print("Vector Store Statistics")
        print(f"{'='*70}{Colors.ENDC}\n")
        
        print(f"{Colors.BOLD}General:{Colors.ENDC}")
        print(f"  Total flows:      {stats['total_flows']:,}")
        print(f"  Index type:       {stats['index_type']}")
        print(f"  Embedding dim:    {stats['dimension']}")
        
        print(f"\n{Colors.BOLD}Protocol Distribution:{Colors.ENDC}")
        proto_dist = stats['protocol_distribution']
        total = stats['total_flows']
        
        for proto, count in sorted(proto_dist.items()):
            percentage = (count / total * 100) if total > 0 else 0
            bar_length = int(percentage / 2)
            bar = '█' * bar_length + '░' * (50 - bar_length)
            print(f"  {proto:8s}: {bar} {percentage:5.1f}% ({count:,})")
        
        print(f"\n{Colors.BOLD}Application Protocol Distribution:{Colors.ENDC}")
        app_dist = stats['app_protocol_distribution']
        
        for app_proto, count in sorted(app_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
            percentage = (count / total * 100) if total > 0 else 0
            bar_length = int(percentage / 2)
            bar = '█' * bar_length + '░' * (50 - bar_length)
            print(f"  {app_proto:8s}: {bar} {percentage:5.1f}% ({count:,})")
        
        # Calculate additional statistics
        print(f"\n{Colors.BOLD}Flow Statistics:{Colors.ENDC}")
        
        total_bytes = sum(f.get('statistical', {}).get('total_bytes', 0) 
                         for f in self.store.metadata)
        total_packets = sum(f.get('statistical', {}).get('packet_count', 0) 
                           for f in self.store.metadata)
        
        print(f"  Total bytes:      {total_bytes:,.0f}")
        print(f"  Total packets:    {total_packets:,.0f}")
        print(f"  Avg bytes/flow:   {total_bytes/total:,.1f}" if total > 0 else "  Avg bytes/flow:   N/A")
        print(f"  Avg pkts/flow:    {total_packets/total:,.1f}" if total > 0 else "  Avg pkts/flow:    N/A")
        
        print(f"\n{Colors.BOLD}{'='*70}{Colors.ENDC}\n")
    
    def show_temporal_features(self, index: int):
        """Show detailed temporal features for a specific flow"""
        if index < 0 or index >= self.store.total_flows:
            print(f"{Colors.RED}Error: Flow index out of range (0-{self.store.total_flows-1}){Colors.ENDC}")
            return
        
        flow_data = self.store.metadata[index]
        temporal = flow_data.get('temporal', {})
        
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'─'*70}")
        print(f"Temporal Features - Flow #{index}")
        print(f"{'─'*70}{Colors.ENDC}\n")
        
        for feature_type in ['Δt', 'size', 'direction', 'entropy']:
            values = temporal.get(feature_type, [])
            if values:
                print(f"{Colors.BOLD}{feature_type}:{Colors.ENDC}")
                print(f"  Values: {values}")
                print(f"  Count:  {len(values)}")
                if values:
                    print(f"  First:  {values[0]}")
                    print(f"  Last:   {values[-1]}")
                print()
        
        if not temporal:
            print(f"{Colors.YELLOW}No temporal features available{Colors.ENDC}")
    
    def show_full_metadata(self, index: int):
        """Show complete metadata for a specific flow"""
        if index < 0 or index >= self.store.total_flows:
            print(f"{Colors.RED}Error: Flow index out of range (0-{self.store.total_flows-1}){Colors.ENDC}")
            return
        
        flow_data = self.store.metadata[index]
        
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'─'*70}")
        print(f"Complete Metadata - Flow #{index}")
        print(f"{'─'*70}{Colors.ENDC}\n")
        
        print(f"{Colors.BOLD}Flow Identification:{Colors.ENDC}")
        flow_id = flow_data.get('flow_id', {})
        for key, value in flow_id.items():
            print(f"  {key}: {value}")
        
        print(f"\n{Colors.BOLD}Temporal Features:{Colors.ENDC}")
        temporal = flow_data.get('temporal', {})
        for key, values in temporal.items():
            if values:
                print(f"  {key}: {values[:5]}..." if len(values) > 5 else f"  {key}: {values}")
        
        print(f"\n{Colors.BOLD}Statistical Features:{Colors.ENDC}")
        stats = flow_data.get('statistical', {})
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print(f"\n{Colors.BOLD}Protocol Information:{Colors.ENDC}")
        proto = flow_data.get('protocol', {})
        for key, value in proto.items():
            print(f"  {key}: {value}")
        
        print(f"\n{Colors.BOLD}System Metadata:{Colors.ENDC}")
        print(f"  _id: {flow_data.get('_id', 'N/A')}")
        print(f"  _indexed_at: {flow_data.get('_indexed_at', 'N/A')}")
        print(f"  timestamp: {flow_data.get('timestamp', 'N/A')}")
    
    def _reconstruct_flow(self, flow_data: Dict) -> Optional[Flow]:
        """Reconstruct a Flow object from stored metadata"""
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
            print(f"{Colors.RED}Error reconstructing flow: {e}{Colors.ENDC}")
            return None
    
    def interactive_mode(self):
        """Run interactive query mode"""
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}")
        print("DeepTrace Interactive Query Engine")
        print(f"{'='*70}{Colors.ENDC}\n")
        
        print("Commands:")
        print("  list [N]              - List all flows (limit N, default 10)")
        print("  show <index>          - Show detailed flow information")
        print("  verbose <index>       - Show flow with temporal features")
        print("  temporal <index>      - Show detailed temporal features")
        print("  metadata <index>      - Show complete metadata")
        print("  ip <address>          - Search by IP address")
        print("  port <number>         - Search by port number")
        print("  protocol <name>       - Search by protocol (TCP/UDP/HTTPS/etc)")
        print("  similar <index> [k]   - Find k similar flows (requires --checkpoint)")
        print("  anomalies [thresh] [k]- Find anomalies (requires --checkpoint)")
        print("  stats                 - Show store statistics")
        print("  help                  - Show this help")
        print("  exit, quit            - Exit\n")
        
        while True:
            try:
                cmd = input(f"{Colors.GREEN}query>{Colors.ENDC} ").strip()
                
                if not cmd:
                    continue
                
                parts = cmd.split()
                command = parts[0].lower()
                
                if command in ['exit', 'quit']:
                    print("Goodbye!")
                    break
                
                elif command == 'help':
                    print("\nCommands:")
                    print("  list [N]              - List flows")
                    print("  show <index>          - Show flow details")
                    print("  verbose <index>       - Show flow with temporal features")
                    print("  temporal <index>      - Show detailed temporal features")
                    print("  metadata <index>      - Show complete metadata")
                    print("  ip <address>          - Search by IP")
                    print("  port <number>         - Search by port")
                    print("  protocol <name>       - Search by protocol")
                    print("  similar <index> [k]   - Find similar flows")
                    print("  anomalies [thresh] [k]- Find anomalies")
                    print("  stats                 - Show statistics")
                    print("  help                  - Show help")
                    print("  exit, quit            - Exit\n")
                
                elif command == 'list':
                    limit = int(parts[1]) if len(parts) > 1 else 10
                    self.list_all(limit)
                
                elif command == 'show':
                    if len(parts) < 2:
                        print(f"{Colors.RED}Usage: show <index>{Colors.ENDC}")
                    else:
                        self.show_flow(int(parts[1]))
                
                elif command == 'ip':
                    if len(parts) < 2:
                        print(f"{Colors.RED}Usage: ip <address>{Colors.ENDC}")
                    else:
                        k = int(parts[2]) if len(parts) > 2 else 10
                        self.search_by_ip(parts[1], k)
                
                elif command == 'port':
                    if len(parts) < 2:
                        print(f"{Colors.RED}Usage: port <number>{Colors.ENDC}")
                    else:
                        k = int(parts[2]) if len(parts) > 2 else 10
                        self.search_by_port(int(parts[1]), k)
                
                elif command == 'protocol':
                    if len(parts) < 2:
                        print(f"{Colors.RED}Usage: protocol <name>{Colors.ENDC}")
                    else:
                        k = int(parts[2]) if len(parts) > 2 else 10
                        self.search_by_protocol(parts[1], k)
                
                elif command == 'similar':
                    if len(parts) < 2:
                        print(f"{Colors.RED}Usage: similar <index> [k]{Colors.ENDC}")
                    else:
                        k = int(parts[2]) if len(parts) > 2 else 5
                        self.find_similar(int(parts[1]), k)
                
                elif command == 'anomalies':
                    threshold = float(parts[1]) if len(parts) > 1 else 2.0
                    k = int(parts[2]) if len(parts) > 2 else 10
                    self.find_anomalies(threshold, k)
                
                elif command == 'stats':
                    self.show_statistics()
                
                elif command == 'verbose':
                    if len(parts) < 2:
                        print(f"{Colors.RED}Usage: verbose <index>{Colors.ENDC}")
                    else:
                        self.show_flow(int(parts[1]), verbose=True)
                
                elif command == 'temporal':
                    if len(parts) < 2:
                        print(f"{Colors.RED}Usage: temporal <index>{Colors.ENDC}")
                    else:
                        self.show_temporal_features(int(parts[1]))
                
                elif command == 'metadata':
                    if len(parts) < 2:
                        print(f"{Colors.RED}Usage: metadata <index>{Colors.ENDC}")
                    else:
                        self.show_full_metadata(int(parts[1]))
                
                else:
                    print(f"{Colors.RED}Unknown command: {command}{Colors.ENDC}")
                    print("Type 'help' for available commands")
            
            except KeyboardInterrupt:
                print("\nUse 'exit' or 'quit' to exit")
            except Exception as e:
                print(f"{Colors.RED}Error: {e}{Colors.ENDC}")


def main():
    parser = argparse.ArgumentParser(
        description="DeepTrace Vector Store Query Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python3 query_store.py -s ./vector_store

  # With model for similarity search
  python3 query_store.py -s ./vector_store -c models/checkpoints/model_epoch_50.pth

  # Show statistics
  python3 query_store.py -s ./vector_store --stats

  # Show flow details
  python3 query_store.py -s ./vector_store --show 5

  # Show flow with temporal features
  python3 query_store.py -s ./vector_store --verbose 5

  # Show complete metadata
  python3 query_store.py -s ./vector_store --metadata 5

  # Search by IP
  python3 query_store.py -s ./vector_store --ip 192.168.1.100

  # Find similar flows
  python3 query_store.py -s ./vector_store -c models/checkpoints/model_epoch_50.pth --similar 0
        """
    )
    
    parser.add_argument(
        '-s', '--store',
        required=True,
        help='Path to vector store directory'
    )
    parser.add_argument(
        '-c', '--checkpoint',
        help='Path to model checkpoint (required for similarity search and anomaly detection)'
    )
    
    # Query options
    parser.add_argument('--stats', action='store_true', help='Show statistics and exit')
    parser.add_argument('--list', type=int, metavar='N', help='List N flows and exit')
    parser.add_argument('--show', type=int, metavar='INDEX', help='Show flow at index')
    parser.add_argument('--verbose', type=int, metavar='INDEX', help='Show flow with temporal features')
    parser.add_argument('--temporal', type=int, metavar='INDEX', help='Show detailed temporal features')
    parser.add_argument('--metadata', type=int, metavar='INDEX', help='Show complete metadata')
    parser.add_argument('--ip', metavar='ADDRESS', help='Search by IP address')
    parser.add_argument('--port', type=int, metavar='PORT', help='Search by port')
    parser.add_argument('--protocol', metavar='NAME', help='Search by protocol')
    parser.add_argument('--similar', type=int, metavar='INDEX', help='Find similar flows')
    parser.add_argument('--anomalies', action='store_true', help='Find anomalies')
    parser.add_argument('-k', type=int, default=10, help='Number of results (default: 10)')
    
    args = parser.parse_args()
    
    try:
        engine = QueryEngine(args.store, args.checkpoint)
        
        # If specific query provided, execute and exit
        if args.stats:
            engine.show_statistics()
        elif args.list:
            engine.list_all(args.list)
        elif args.show is not None:
            engine.show_flow(args.show)
        elif args.verbose is not None:
            engine.show_flow(args.verbose, verbose=True)
        elif args.temporal is not None:
            engine.show_temporal_features(args.temporal)
        elif args.metadata is not None:
            engine.show_full_metadata(args.metadata)
        elif args.ip:
            engine.search_by_ip(args.ip, args.k)
        elif args.port:
            engine.search_by_port(args.port, args.k)
        elif args.protocol:
            engine.search_by_protocol(args.protocol, args.k)
        elif args.similar is not None:
            engine.find_similar(args.similar, args.k)
        elif args.anomalies:
            engine.find_anomalies(k=args.k)
        else:
            # No specific query, enter interactive mode
            engine.interactive_mode()
    
    except FileNotFoundError as e:
        print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
        print(f"Make sure the vector store exists at: {args.store}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()