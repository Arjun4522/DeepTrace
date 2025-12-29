
from scapy.all import sniff, rdpcap
from scapy.packet import Packet
import json
import time
from typing import Optional
import redis


class PacketCapture:
    def __init__(self, interface=None, pcap_file=None, redis_host=None, redis_port=6379, redis_stream='packets:raw'):
        self.interface = interface
        self.pcap_file = pcap_file
        self.redis_enabled = redis_host is not None
        
        if self.redis_enabled:
            try:
                import redis
                self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
                self.redis_stream = redis_stream
                self.redis_client.ping()
                print(f"✓ Redis enabled: {redis_host}:{redis_port} → {redis_stream}")
            except ImportError:
                print("⚠️  Redis not available - falling back to local capture")
                self.redis_enabled = False
            except Exception as e:
                print(f"⚠️  Redis connection failed: {e} - falling back to local capture")
                self.redis_enabled = False

    def capture_packets(self, count=0):
        if self.pcap_file:
            return self.capture_pcap(count)
        elif self.interface:
            return self.capture_live(count)
        else:
            raise ValueError("Either interface or pcap_file must be specified.")

    def capture_live(self, count=0):
        """
        Capture packets live from the specified interface.
        """
        if not self.interface:
            raise ValueError("Interface must be specified for live capture.")
        
        if self.redis_enabled:
            return self._capture_live_to_redis(count)
        else:
            print(f"Capturing packets on interface {self.interface}...")
            packets = sniff(iface=self.interface, count=count)
            return packets
    
    def _capture_live_to_redis(self, count=0):
        """Capture packets and write to Redis stream"""
        print(f"Capturing packets on {self.interface} → Redis stream '{self.redis_stream}'")
        
        packet_count = 0
        
        def packet_handler(packet):
            nonlocal packet_count
            
            # Convert packet to serializable format
            packet_data = self._packet_to_dict(packet)
            
            # Write to Redis stream
            try:
                self.redis_client.xadd(
                    self.redis_stream,
                    {'packet': json.dumps(packet_data)},
                    maxlen=10000
                )
                
                packet_count += 1
                if packet_count % 100 == 0:
                    print(f"Captured {packet_count} packets...")
                    
            except Exception as e:
                print(f"Error writing to Redis: {e}")
        
        # Start capture
        try:
            sniff(
                iface=self.interface,
                prn=packet_handler,
                store=False,
                count=count
            )
        except KeyboardInterrupt:
            print("\nCapture stopped by user")
        except Exception as e:
            print(f"Capture error: {e}")
        
        print(f"Total packets captured: {packet_count}")
        return []  # Return empty list since packets are in Redis
    
    def _packet_to_dict(self, packet):
        """Convert Scapy packet to serializable dictionary"""
        try:
            packet_dict = {
                'time': packet.time,
                'size': len(packet),
                'summary': packet.summary(),
                'has_ip': 'IP' in packet,
                'has_tcp': 'TCP' in packet,
                'has_udp': 'UDP' in packet
            }
            
            if 'IP' in packet:
                packet_dict.update({
                    'src_ip': packet['IP'].src,
                    'dst_ip': packet['IP'].dst,
                    'proto': packet['IP'].proto
                })
            
            if 'TCP' in packet:
                packet_dict.update({
                    'src_port': packet['TCP'].sport,
                    'dst_port': packet['TCP'].dport,
                    'tcp_flags': str(packet['TCP'].flags)
                })
            
            if 'UDP' in packet:
                packet_dict.update({
                    'src_port': packet['UDP'].sport,
                    'dst_port': packet['UDP'].dport
                })
                
            return packet_dict
        except Exception as e:
            print(f"Error converting packet: {e}")
            return {'error': str(e), 'summary': packet.summary()}

    def capture_pcap(self, count=0):
        """
        Read packets from a PCAP file.
        """
        if not self.pcap_file:
            raise ValueError("PCAP file must be specified for pcap capture.")

        if self.redis_enabled:
            return self._capture_pcap_to_redis(count)
        else:
            print(f"Reading packets from {self.pcap_file}...")
            if count > 0:
                packets = rdpcap(self.pcap_file, count=count)
            else:
                packets = rdpcap(self.pcap_file)
            return packets
    
    def _capture_pcap_to_redis(self, count=0):
        """Read PCAP file and write packets to Redis stream"""
        print(f"Reading packets from {self.pcap_file} → Redis stream '{self.redis_stream}'")
        
        # Read packets from PCAP
        if count > 0:
            packets = rdpcap(self.pcap_file, count=count)
        else:
            packets = rdpcap(self.pcap_file)
        
        packet_count = 0
        
        for packet in packets:
            packet_data = self._packet_to_dict(packet)
            
            try:
                self.redis_client.xadd(
                    self.redis_stream,
                    {'packet': json.dumps(packet_data)},
                    maxlen=10000
                )
                packet_count += 1
                
                if packet_count % 100 == 0:
                    print(f"Processed {packet_count} packets...")
                    
            except Exception as e:
                print(f"Error writing packet {packet_count} to Redis: {e}")
        
        print(f"Total packets processed: {packet_count}")
        return []  # Return empty list since packets are in Redis
