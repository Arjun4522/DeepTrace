#!/usr/bin/env python3
"""
DeepTrace: Network packet capture, flow extraction, and realtime embedding pipeline
"""

from deeptrace.capture import PacketCapture
from deeptrace.features import FlowExtractor
import json
import argparse
import os
import datetime
import sys
import signal
import time
import threading
from queue import Queue


def basic_mode(interface, packet_count=0):
    """
    Basic mode: Capture packets once, extract flows, and save to JSONL.
    This is the original functionality.
    """
    print(f"Running in BASIC mode on interface: {interface}")
    
    try:
        # 1. Capture packets
        packet_capturer = PacketCapture(interface=interface)
        packets = packet_capturer.capture_packets(count=packet_count)

        if not packets:
            print("No packets were captured. Make sure you have traffic on the interface.")
            return

        print(f"Captured {len(packets)} packets.")

        # 2. Extract flows and features
        flow_extractor = FlowExtractor()
        flows = flow_extractor.process_packets(packets)

        if not flows:
            print("No flows were extracted.")
            return

        print(f"Extracted {len(flows)} flows.")

        # 3. Save dataset
        output_dir = "train"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        outfile = os.path.join(output_dir, f"flows_{timestamp}.jsonl")

        with open(outfile, "a") as f:
            for flow in flows:
                flow_dict = {
                    "flow_id": flow.flow_id,
                    "temporal": flow.temporal,
                    "statistical": flow.statistical,
                    "protocol": flow.protocol,
                    "timestamp": flow.timestamp
                }
                json.dump(flow_dict, f)
                f.write("\n")

        print(f"\n[+] Saved {len(flows)} flows to {outfile}")

        # 4. Print extracted features
        for i, flow in enumerate(flows):
            print(f"\n--- Flow {i+1} ---")
            flow_dict = {
                "flow_id": flow.flow_id,
                "temporal": flow.temporal,
                "statistical": flow.statistical,
                "protocol": flow.protocol,
                "timestamp": flow.timestamp
            }
            print(json.dumps(flow_dict, indent=2))

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you are running this script with sufficient privileges (e.g., using sudo).")
        print("Also, make sure the specified interface exists and is active.")


def streaming_mode(interface, checkpoint_path, storage_path, capture_interval, batch_size, device, load_existing):
    """
    Streaming mode: Continuously capture packets, extract flows, generate embeddings,
    and store in vector database for RAG applications.
    """
    print(f"Running in STREAMING mode on interface: {interface}")
    
    try:
        from deeptrace.model import ModelInference
        from deeptrace.storage import StreamingVectorStore, FlowQueryEngine
    except ImportError as e:
        print(f"Error importing streaming components: {e}")
        print("Make sure model.py and storage.py are properly implemented.")
        return
    
    # Initialize pipeline components
    class StreamingPipeline:
        def __init__(self):
            print("\nInitializing streaming pipeline components...")
            self.interface = interface
            self.checkpoint_path = checkpoint_path
            self.storage_path = storage_path
            self.capture_interval = capture_interval
            self.batch_size = batch_size
            self.device = device
            
            # Initialize components
            self.packet_capturer = PacketCapture(interface=interface)
            self.model = ModelInference(checkpoint_path, device=device)
            
            # Get actual embedding dimension from loaded model
            embedding_dim = getattr(self.model, 'embedding_dim', 64)
            
            self.vector_store = StreamingVectorStore(
                dimension=embedding_dim,
                index_type="flatl2",
                buffer_size=batch_size
            )
            self.query_engine = FlowQueryEngine(self.vector_store, self.model)
            
            # Load existing store if requested
            if load_existing:
                try:
                    self.vector_store.load(storage_path)
                    print(f"✓ Loaded existing vector store: {self.vector_store.total_flows} flows")
                except FileNotFoundError:
                    print(f"No existing store found at {storage_path}, starting fresh")
            
            # Pipeline state
            self.running = False
            self.packet_queue = Queue(maxsize=10000)
            self.start_time = None
            
            # Statistics
            self.stats = {
                'total_packets': 0,
                'total_flows': 0,
                'total_embeddings': 0,
                'last_batch_time': None
            }
        
        def start(self):
            """Start the streaming pipeline"""
            self.running = True
            self.start_time = time.time()
            
            print(f"\n{'='*70}")
            print(f"🚀 DeepTrace Realtime Embedding Pipeline Started")
            print(f"{'='*70}")
            print(f"  Interface:        {self.interface}")
            print(f"  Model:            {self.checkpoint_path}")
            print(f"  Storage:          {self.storage_path}")
            print(f"  Capture interval: {self.capture_interval}s")
            print(f"  Batch size:       {self.batch_size}")
            print(f"  Device:           {self.device}")
            print(f"{'='*70}\n")
            
            # Start packet capture thread
            capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            capture_thread.start()
            
            # Start processing loop
            try:
                self._processing_loop()
            except KeyboardInterrupt:
                print("\n⚠️  Received interrupt signal. Shutting down...")
                self.stop()
        
        def _capture_loop(self):
            """Continuously capture packets"""
            from scapy.all import sniff
            
            def packet_handler(packet):
                if self.running:
                    try:
                        self.packet_queue.put(packet, timeout=1)
                        self.stats['total_packets'] += 1
                    except:
                        pass  # Queue full, drop packet
            
            while self.running:
                try:
                    sniff(
                        iface=self.interface,
                        prn=packet_handler,
                        store=False,
                        timeout=self.capture_interval
                    )
                except Exception as e:
                    if self.running:
                        print(f"❌ Capture error: {e}")
                        time.sleep(1)
        
        def _processing_loop(self):
            """Process captured packets in intervals"""
            last_process_time = time.time()
            
            while self.running:
                current_time = time.time()
                
                if current_time - last_process_time >= self.capture_interval:
                    self._process_batch()
                    last_process_time = current_time
                    self._print_stats()
                
                time.sleep(0.1)
        
        def _process_batch(self):
            """Extract flows, generate embeddings, and store"""
            # Collect packets from queue
            packets = []
            while not self.packet_queue.empty() and len(packets) < 10000:
                try:
                    packets.append(self.packet_queue.get_nowait())
                except:
                    break
            
            if not packets:
                timestamp = datetime.datetime.now().strftime('%H:%M:%S')
                print(f"[{timestamp}] ⏸️  No packets in this interval")
                return
            
            timestamp = datetime.datetime.now().strftime('%H:%M:%S')
            print(f"\n[{timestamp}] 📦 Processing {len(packets)} packets...")
            
            # Extract flows
            flow_extractor = FlowExtractor()
            flows = flow_extractor.process_packets(packets)
            
            if not flows:
                print(f"  ⚠️  No flows extracted")
                return
            
            self.stats['total_flows'] += len(flows)
            print(f"  ✓ Extracted {len(flows)} flows")
            
            # Generate embeddings
            print(f"  🧠 Generating embeddings...")
            start_time = time.time()
            embeddings = self.model.embed_flows_batch(flows)
            embed_time = time.time() - start_time
            
            self.stats['total_embeddings'] += len(embeddings)
            self.stats['last_batch_time'] = datetime.datetime.now().isoformat()
            
            # Store in vector database
            flow_data_list = []
            for flow in flows:
                flow_data = {
                    'flow_id': flow.flow_id,
                    'temporal': {k: v[:10] for k, v in flow.temporal.items()},
                    'statistical': flow.statistical,
                    'protocol': flow.protocol,
                    'timestamp': flow.timestamp
                }
                flow_data_list.append(flow_data)
            
            for emb, flow_data in zip(embeddings, flow_data_list):
                self.vector_store.buffer_flow(emb, flow_data)
            
            print(f"  ✓ Generated {len(embeddings)} embeddings in {embed_time:.2f}s")
            print(f"    ({len(embeddings)/embed_time:.1f} flows/sec)")
            
            # Periodically save checkpoint
            if self.stats['total_embeddings'] % 500 < len(embeddings):
                self._save_checkpoint()
        
        def _print_stats(self):
            """Print pipeline statistics"""
            uptime = time.time() - self.start_time
            hours = int(uptime // 3600)
            minutes = int((uptime % 3600) // 60)
            seconds = int(uptime % 60)
            
            print(f"\n{'─'*70}")
            print(f"📊 Statistics (Uptime: {hours:02d}:{minutes:02d}:{seconds:02d})")
            print(f"{'─'*70}")
            print(f"  Packets captured:     {self.stats['total_packets']:,}")
            print(f"  Flows extracted:      {self.stats['total_flows']:,}")
            print(f"  Embeddings generated: {self.stats['total_embeddings']:,}")
            print(f"  Vector store size:    {self.vector_store.total_flows:,}")
            
            # Protocol distribution
            if self.vector_store.total_flows > 0:
                store_stats = self.vector_store.get_statistics()
                print(f"\n  Protocol distribution:")
                for proto, count in sorted(store_stats['protocol_distribution'].items()):
                    percentage = (count / store_stats['total_flows'] * 100)
                    bar_length = int(percentage / 2)
                    bar = '█' * bar_length + '░' * (50 - bar_length)
                    print(f"    {proto:8s}: {bar} {percentage:5.1f}% ({count:,})")
            
            print(f"{'─'*70}")
        
        def _save_checkpoint(self):
            """Save vector store to disk"""
            timestamp = datetime.datetime.now().strftime('%H:%M:%S')
            print(f"\n[{timestamp}] 💾 Saving checkpoint...")
            self.vector_store.flush()
            self.vector_store.save(self.storage_path)
            print(f"  ✓ Saved to {self.storage_path}")
        
        def stop(self):
            """Stop the pipeline gracefully"""
            self.running = False
            
            print("\n🔄 Flushing buffers and saving final state...")
            self.vector_store.flush()
            self._save_checkpoint()
            
            print("\n✅ Pipeline stopped successfully!")
            self._print_stats()
    
    # Setup signal handlers for graceful shutdown
    pipeline = StreamingPipeline()
    
    def signal_handler(sig, frame):
        print("\n⚠️  Received shutdown signal...")
        pipeline.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start pipeline
    pipeline.start()


def main():
    parser = argparse.ArgumentParser(
        description="DeepTrace: Network packet capture, flow extraction, and realtime embedding pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic mode: Capture 100 packets and save flows
  sudo python3 main.py -i wlo1 --basic --count 100

  # Streaming mode: Continuous capture with embeddings
  sudo python3 main.py -i wlo1 --stream -c models/checkpoints/model_epoch_50.pth

  # Streaming mode with custom settings
  sudo python3 main.py -i wlo1 --stream -c models/checkpoints/model_epoch_50.pth \\
      -s ./my_vector_store -t 5 -b 100 --device cuda --load
        """
    )
    
    # Common arguments
    parser.add_argument(
        "-i", "--interface",
        required=True,
        help="Network interface to capture from (e.g., eth0, wlo1)"
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--basic",
        action="store_true",
        help="Run in basic mode (capture once and save)"
    )
    mode_group.add_argument(
        "--stream",
        action="store_true",
        help="Run in streaming mode (continuous capture with embeddings)"
    )
    
    # Basic mode arguments
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Number of packets to capture in basic mode (0 = until Ctrl+C)"
    )
    
    # Streaming mode arguments
    parser.add_argument(
        "-c", "--checkpoint",
        help="Path to model checkpoint (required for streaming mode)"
    )
    parser.add_argument(
        "-s", "--storage",
        default="./vector_store",
        help="Path to vector store directory (default: ./vector_store)"
    )
    parser.add_argument(
        "-t", "--interval",
        type=int,
        default=10,
        help="Capture interval in seconds for streaming mode (default: 10)"
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=50,
        help="Batch size for embedding generation (default: 50)"
    )
    parser.add_argument(
        "-d", "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for model inference (default: cpu)"
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load existing vector store if available (streaming mode)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.stream and not args.checkpoint:
        parser.error("--checkpoint is required when using --stream mode")
    
    # Run appropriate mode
    if args.basic:
        basic_mode(args.interface, args.count)
    else:  # streaming mode
        streaming_mode(
            interface=args.interface,
            checkpoint_path=args.checkpoint,
            storage_path=args.storage,
            capture_interval=args.interval,
            batch_size=args.batch_size,
            device=args.device,
            load_existing=args.load
        )


if __name__ == "__main__":
    main()