
import unittest
from scapy.all import IP, TCP, UDP, ICMP, Raw
from deeptrace.features import Flow, FlowExtractor

class TestFlow(unittest.TestCase):

    def test_flow_initialization(self):
        flow = Flow(("1.1.1.1", 1234, "2.2.2.2", 80, 6))
        self.assertEqual(flow.flow_id["src_ip"], "1.1.1.1")
        self.assertEqual(flow.protocol["proto"], "TCP")

    def test_add_packet(self):
        flow = Flow(("1.1.1.1", 1234, "2.2.2.2", 80, 6))
        packet = IP(src="1.1.1.1", dst="2.2.2.2") / TCP(sport=1234, dport=80) / Raw(load="test")
        packet.time = 100
        flow.add_packet(packet, 0)

        self.assertEqual(len(flow.temporal["pkt_size"]), 1)
        self.assertEqual(flow.temporal["pkt_size"][0], len(packet))
        self.assertEqual(flow.temporal["direction"][0], 1.0)
        self.assertNotEqual(flow.temporal["entropy"][0], 0.0)

    def test_calculate_final_features(self):
        flow = Flow(("1.1.1.1", 1234, "2.2.2.2", 80, 6))
        packet1 = IP(src="1.1.1.1", dst="2.2.2.2") / TCP(sport=1234, dport=80) / Raw(load="test1")
        packet1.time = 100
        flow.add_packet(packet1, 0)

        packet2 = IP(src="2.2.2.2", dst="1.1.1.1") / TCP(sport=80, dport=1234) / Raw(load="test2")
        packet2.time = 101
        flow.add_packet(packet2, 1)

        flow.calculate_final_features()

        self.assertEqual(flow.statistical["packet_count"], 2)
        self.assertAlmostEqual(flow.statistical["flow_duration"], 1.0)
        self.assertEqual(flow.protocol["app_proto"], "HTTP")

    def test_calculate_entropy(self):
        entropy = Flow.calculate_entropy(b"aabbcc")
        self.assertAlmostEqual(entropy, 1.58496, places=5)
        entropy_empty = Flow.calculate_entropy(b"")
        self.assertEqual(entropy_empty, 0.0)

    def test_get_protocol_name(self):
        flow = Flow(("1.1.1.1", 1234, "2.2.2.2", 80, 6))
        self.assertEqual(flow._get_protocol_name(6), "TCP")
        self.assertEqual(flow._get_protocol_name(17), "UDP")
        self.assertEqual(flow._get_protocol_name(1), "ICMP")
        self.assertEqual(flow._get_protocol_name(99), "OTHER")

    def test_get_application_protocol(self):
        flow = Flow(("1.1.1.1", 1234, "2.2.2.2", 80, 6))
        self.assertEqual(flow._get_application_protocol(1234, 80, 6), "HTTP")
        self.assertEqual(flow._get_application_protocol(1234, 53, 17), "DNS")
        self.assertEqual(flow._get_application_protocol(1234, 9999, 6), "UNKNOWN")

    def test_summarize_tcp_flags(self):
        flow = Flow(("1.1.1.1", 1234, "2.2.2.2", 80, 6))
        self.assertEqual(flow._summarize_tcp_flags(["S", "A", "S", "A"]), "S_A")
        self.assertEqual(flow._summarize_tcp_flags([]), "NONE")
        self.assertEqual(flow._summarize_tcp_flags(["F", "A"]), "F_A")

class TestFlowExtractor(unittest.TestCase):

    def test_get_flow_id(self):
        extractor = FlowExtractor()
        packet_tcp = IP(src="1.1.1.1", dst="2.2.2.2") / TCP(sport=1234, dport=80)
        self.assertEqual(extractor.get_flow_id(packet_tcp), ("1.1.1.1", 1234, "2.2.2.2", 80, 6))

        packet_udp = IP(src="1.1.1.1", dst="2.2.2.2") / UDP(sport=1234, dport=53)
        self.assertEqual(extractor.get_flow_id(packet_udp), ("1.1.1.1", 1234, "2.2.2.2", 53, 17))

        packet_icmp = IP(src="1.1.1.1", dst="2.2.2.2") / ICMP()
        self.assertEqual(extractor.get_flow_id(packet_icmp), ("1.1.1.1", None, "2.2.2.2", None, 1))

    def test_canonicalize_flow_id(self):
        extractor = FlowExtractor()
        flow_id = ("1.1.1.1", 1234, "2.2.2.2", 80, 6)
        canonical_id, direction = extractor._canonicalize_flow_id(flow_id)
        self.assertEqual(canonical_id, ("1.1.1.1", 1234, "2.2.2.2", 80, 6))
        self.assertEqual(direction, 0)

        flow_id_rev = ("2.2.2.2", 80, "1.1.1.1", 1234, 6)
        canonical_id_rev, direction_rev = extractor._canonicalize_flow_id(flow_id_rev)
        self.assertEqual(canonical_id_rev, ("1.1.1.1", 1234, "2.2.2.2", 80, 6))
        self.assertEqual(direction_rev, 1)

    def test_process_packets(self):
        extractor = FlowExtractor()
        packets = [
            IP(src="1.1.1.1", dst="2.2.2.2", proto=6) / TCP(sport=1234, dport=80),
            IP(src="2.2.2.2", dst="1.1.1.1", proto=6) / TCP(sport=80, dport=1234),
            IP(src="3.3.3.3", dst="4.4.4.4", proto=17) / UDP(sport=53, dport=5353)
        ]
        for p in packets:
            p.time = 100

        flows = extractor.process_packets(packets)
        self.assertEqual(len(flows), 2)

if __name__ == "__main__":
    unittest.main()
