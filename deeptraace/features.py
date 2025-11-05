import numpy as np
from collections import defaultdict
from scapy.all import IP, TCP, UDP, ICMP
from datetime import datetime

class Flow:
    def __init__(self, flow_id_tuple):
        self.flow_id = {
            "src_ip": flow_id_tuple[0],
            "src_port": flow_id_tuple[1],
            "dst_ip": flow_id_tuple[2],
            "dst_port": flow_id_tuple[3],
            "protocol": flow_id_tuple[4]
        }
        
        # Tier 1: Temporal Features (raw values, per-packet)
        self.temporal = {
            "dt": [],
            "pkt_size": [],
            "direction": [], # +1 for forward, -1 for backward
            "entropy": []
        }

        # Tier 2: Statistical Features (aggregated, per-flow)
        self.statistical = {}

        # Tier 3: Protocol Features (categorical, per-flow)
        self.protocol = {
            "proto": self._get_protocol_name(self.flow_id["protocol"]),
            "app_proto": "UNKNOWN",
            "flags_summary": "NONE" # e.g., "S", "SA", "S_A_F"
        }

        self.timestamp = datetime.now().isoformat() + 'Z'

        # Internal state for calculations
        self._first_packet_time = None
        self._last_packet_time = None
        self._total_bytes_forward = 0
        self._total_bytes_backward = 0
        self._all_tcp_flags_raw = []

    def add_packet(self, packet, direction):
        current_time = packet.time
        current_size = len(packet)

        if self._first_packet_time is None:
            self._first_packet_time = current_time
            self.timestamp = datetime.fromtimestamp(current_time).isoformat() + 'Z'

        delta_t = 0.0
        if self._last_packet_time is not None:
            delta_t = current_time - self._last_packet_time

        entropy = self.calculate_entropy(bytes(packet[IP].payload)) if IP in packet and hasattr(packet[IP], 'payload') else 0.0
        
        # Populate temporal features
        self.temporal["dt"].append(delta_t)
        self.temporal["pkt_size"].append(float(current_size))
        self.temporal["direction"].append(1.0 if direction == 0 else -1.0)
        self.temporal["entropy"].append(entropy)

        # Update internal state
        self._last_packet_time = current_time
        if direction == 0: # Forward
            self._total_bytes_forward += current_size
        else: # Backward
            self._total_bytes_backward += current_size
        
        if TCP in packet:
            self._all_tcp_flags_raw.extend(self._get_tcp_flags_raw(packet))

    def calculate_final_features(self):
        if not self.temporal["pkt_size"]:
            return

        # Calculate Statistical Features
        packet_count = len(self.temporal["pkt_size"])
        total_bytes = sum(self.temporal["pkt_size"])
        flow_duration = self._last_packet_time - self._first_packet_time if self._last_packet_time and self._first_packet_time else 0.0
        
        valid_dts = [dt for dt in self.temporal["dt"] if dt > 0]
        mean_dt = np.mean(valid_dts) if valid_dts else 0.0
        std_dt = np.std(valid_dts) if len(valid_dts) > 1 else 0.0

        self.statistical = {
            "flow_duration": flow_duration,
            "total_bytes": total_bytes,
            "packet_count": float(packet_count),
            "mean_pkt_size": np.mean(self.temporal["pkt_size"]),
            "std_pkt_size": np.std(self.temporal["pkt_size"]) if packet_count > 1 else 0.0,
            "mean_dt": mean_dt,
            "std_dt": std_dt,
            "entropy_mean": np.mean(self.temporal["entropy"]),
            "entropy_std": np.std(self.temporal["entropy"]) if packet_count > 1 else 0.0,
            "byte_ratio": self._total_bytes_forward / total_bytes if total_bytes > 0 else 0.0
        }

        # Calculate Protocol Features
        self.protocol["app_proto"] = self._get_application_protocol(self.flow_id["src_port"], self.flow_id["dst_port"], self.flow_id["protocol"])
        self.protocol["flags_summary"] = self._summarize_tcp_flags(self._all_tcp_flags_raw)

    @staticmethod
    def calculate_entropy(payload):
        if not payload:
            return 0.0
        byte_counts = defaultdict(int)
        for byte in payload:
            byte_counts[byte] += 1
        
        entropy = 0.0
        for count in byte_counts.values():
            p = count / len(payload)
            entropy -= p * np.log2(p)
        return entropy

    def _get_tcp_flags_raw(self, packet):
        flags_list = []
        if TCP in packet:
            # Scapy flags can be combined, e.g., "SA" for SYN-ACK. We handle them individually.
            flags = str(packet[TCP].flags)
            if 'S' in flags: flags_list.append("S")
            if 'A' in flags: flags_list.append("A")
            if 'F' in flags: flags_list.append("F")
            if 'R' in flags: flags_list.append("R")
            if 'P' in flags: flags_list.append("P")
            if 'U' in flags: flags_list.append("U")
        return flags_list

    def _summarize_tcp_flags(self, flags_list):
        if not flags_list:
            return "NONE"
        # Create a summary of unique flags in the order they appeared.
        unique_ordered_flags = list(dict.fromkeys(flags_list))
        return "_".join(unique_ordered_flags)

    def _get_protocol_name(self, proto_id):
        if proto_id == 6: return "TCP"
        if proto_id == 17: return "UDP"
        if proto_id == 1: return "ICMP"
        return "OTHER"

    def _get_application_protocol(self, src_port, dst_port, proto):
        # Simple port-based mapping
        port_map = {
            80: "HTTP", 443: "HTTPS", 22: "SSH", 21: "FTP", 25: "SMTP",
            53: "DNS", 67: "DHCP", 68: "DHCP", 123: "NTP"
        }
        if proto == 6 or proto == 17: # TCP or UDP
            for port in [src_port, dst_port]:
                if port is not None and port in port_map:
                    return port_map[port]
        return "UNKNOWN"

class FlowExtractor:
    def __init__(self):
        self.flows = {}

    def process_packets(self, packets):
        for packet in packets:
            if IP in packet:
                flow_id_tuple = self.get_flow_id(packet)
                if flow_id_tuple is None:
                    continue

                canonical_flow_id, direction = self._canonicalize_flow_id(flow_id_tuple)

                if canonical_flow_id not in self.flows:
                    self.flows[canonical_flow_id] = Flow(canonical_flow_id)

                self.flows[canonical_flow_id].add_packet(packet, direction)
        
        for flow in self.flows.values():
            flow.calculate_final_features()
            
        return list(self.flows.values())

    def get_flow_id(self, packet):
        if IP in packet:
            proto = int(packet[IP].proto)
            src_ip = str(packet[IP].src)
            dst_ip = str(packet[IP].dst)
            src_port = None
            dst_port = None

            if TCP in packet:
                src_port = int(packet[TCP].sport)
                dst_port = int(packet[TCP].dport)
            elif UDP in packet:
                src_port = int(packet[UDP].sport)
                dst_port = int(packet[UDP].dport)
            
            return (src_ip, src_port, dst_ip, dst_port, proto)
        return None

    def _canonicalize_flow_id(self, flow_id_tuple):
        """
        Canonicalize flow ID to handle bidirectional flows.
        Returns (canonical_flow_id, direction) where direction is 0 (forward) or 1 (backward)
        """
        src_ip, src_port, dst_ip, dst_port, proto = flow_id_tuple

        # Convert None ports to a comparable value for sorting
        src_port_cmp = src_port if src_port is not None else 0
        dst_port_cmp = dst_port if dst_port is not None else 0

        # Canonicalize by sorting (IP, Port) pairs to handle bidirectional flows
        # Compare tuples: (ip, port)
        if (src_ip, src_port_cmp) <= (dst_ip, dst_port_cmp):
            canonical_id = (src_ip, src_port, dst_ip, dst_port, proto)
            direction = 0  # Forward
        else:
            canonical_id = (dst_ip, dst_port, src_ip, src_port, proto)
            direction = 1  # Backward
        
        return canonical_id, direction