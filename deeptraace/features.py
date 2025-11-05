
import numpy as np
from collections import defaultdict
from scapy.all import IP, TCP, UDP

class Flow:
    def __init__(self, flow_id):
        self.flow_id = flow_id
        self.packets = []
        self.features = {}

    def add_packet(self, packet, direction):
        self.packets.append((packet, direction))

    def calculate_features(self):
        if not self.packets:
            return

        # Timestamps
        timestamps = [p.time for p, d in self.packets]
        deltas = np.diff(timestamps)
        self.features['dt'] = deltas

        # Packet sizes
        sizes = [len(p) for p, d in self.packets]
        self.features['pkt_size'] = sizes

        # Direction
        directions = [d for p, d in self.packets]
        self.features['direction'] = directions

        # Protocol
        self.features['proto'] = self.packets[0][0][IP].proto

        # Flags
        if self.features['proto'] == 6: # TCP
            flags = [p[TCP].flags for p, d in self.packets if TCP in p]
            self.features['flags'] = flags

        # Entropy
        payloads = [bytes(p[IP].payload) for p, d in self.packets]
        self.features['entropy'] = [self.calculate_entropy(p) for p in payloads]

    @staticmethod
    def calculate_entropy(payload):
        if not payload:
            return 0
        byte_counts = defaultdict(int)
        for byte in payload:
            byte_counts[byte] += 1
        
        entropy = 0
        for count in byte_counts.values():
            p = count / len(payload)
            entropy -= p * np.log2(p)
        return entropy

class FlowExtractor:
    def __init__(self):
        self.flows = defaultdict(Flow)

    def process_packets(self, packets):
        for packet in packets:
            if IP in packet:
                flow_id = self.get_flow_id(packet)
                direction = self.get_direction(packet)
                if flow_id not in self.flows:
                    self.flows[flow_id] = Flow(flow_id)
                self.flows[flow_id].add_packet(packet, direction)
        
        for flow in self.flows.values():
            flow.calculate_features()
            
        return list(self.flows.values())

    def get_flow_id(self, packet):
        if IP in packet:
            proto = packet[IP].proto
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            if TCP in packet or UDP in packet:
                src_port = packet.sport
                dst_port = packet.dport
                return (src_ip, src_port, dst_ip, dst_port, proto)
            else:
                return (src_ip, None, dst_ip, None, proto)
        return None

    def get_direction(self, packet):
        # Simple direction heuristic: assume first packet determines forward direction
        flow_id = self.get_flow_id(packet)
        if flow_id not in self.flows:
            return 0 # Forward
        else:
            return 1 # Backward
