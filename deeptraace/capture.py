
from scapy.all import sniff, rdpcap

class PacketCapture:
    def __init__(self, interface=None, pcap_file=None):
        self.interface = interface
        self.pcap_file = pcap_file

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
        
        print(f"Capturing packets on interface {self.interface}...")
        packets = sniff(iface=self.interface, count=count)
        return packets

    def capture_pcap(self, count=0):
        """
        Read packets from a PCAP file.
        """
        if not self.pcap_file:
            raise ValueError("PCAP file must be specified for pcap capture.")

        print(f"Reading packets from {self.pcap_file}...")
        if count > 0:
            packets = rdpcap(self.pcap_file, count=count)
        else:
            packets = rdpcap(self.pcap_file)
        return packets
