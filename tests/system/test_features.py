
import unittest
from scapy.all import IP, TCP, Raw
from deeptrace.features import FlowExtractor

class TestFeaturesSystem(unittest.TestCase):

    def test_http_flow_extraction(self):
        # Simulate a simple HTTP GET request and response
        packets = [
            # Handshake
            IP(src="192.168.1.100", dst="8.8.8.8") / TCP(sport=12345, dport=80, flags="S"),
            IP(src="8.8.8.8", dst="192.168.1.100") / TCP(sport=80, dport=12345, flags="SA"),
            IP(src="192.168.1.100", dst="8.8.8.8") / TCP(sport=12345, dport=80, flags="A"),
            # Request
            IP(src="192.168.1.100", dst="8.8.8.8") / TCP(sport=12345, dport=80, flags="PA") / Raw(load="GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"),
            # Response
            IP(src="8.8.8.8", dst="192.168.1.100") / TCP(sport=80, dport=12345, flags="PA") / Raw(load="HTTP/1.1 200 OK\r\nContent-Length: 12\r\n\r\nHello, world!"),
            # Teardown
            IP(src="192.168.1.100", dst="8.8.8.8") / TCP(sport=12345, dport=80, flags="FA"),
            IP(src="8.8.8.8", dst="192.168.1.100") / TCP(sport=80, dport=12345, flags="FA"),
            IP(src="192.168.1.100", dst="8.8.8.8") / TCP(sport=12345, dport=80, flags="A"),
        ]

        # Assign timestamps
        base_time = 1672531200 # 2023-01-01 00:00:00 UTC
        for i, p in enumerate(packets):
            p.time = base_time + i * 0.1

        # Process packets
        extractor = FlowExtractor()
        flows = extractor.process_packets(packets)

        # Verification
        self.assertEqual(len(flows), 1, "Should extract a single flow")
        
        flow = flows[0]
        self.assertEqual(flow.flow_id["src_ip"], "192.168.1.100")
        self.assertEqual(flow.flow_id["dst_ip"], "8.8.8.8")
        self.assertEqual(flow.flow_id["src_port"], 12345)
        self.assertEqual(flow.flow_id["dst_port"], 80)
        self.assertEqual(flow.protocol["proto"], "TCP")
        self.assertEqual(flow.protocol["app_proto"], "HTTP")
        self.assertEqual(flow.protocol["flags_summary"], "S_A_P_F")

        self.assertEqual(flow.statistical["packet_count"], 8)
        self.assertAlmostEqual(flow.statistical["flow_duration"], 0.7, places=5)
        self.assertGreater(flow.statistical["total_bytes"], 0)
        self.assertGreater(flow.statistical["mean_pkt_size"], 0)
        self.assertGreater(flow.statistical["mean_dt"], 0)

if __name__ == "__main__":
    unittest.main()
