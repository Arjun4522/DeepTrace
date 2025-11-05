
from deeptraace.capture import PacketCapture
from deeptraace.features import FlowExtractor

def main():
    # NOTE: You may need to change the interface name to match your system.
    # Common interface names include 'eth0', 'enp0s3', 'wlan0', etc.
    # You can find your interface names by running `ip addr` or `ifconfig` in your terminal.
    interface = "wlo1"
    
    print(f"Testing live packet capture on interface: {interface}")
    
    try:
        # 1. Capture live packets
        packet_capturer = PacketCapture(interface=interface)
        packets = packet_capturer.capture_packets()
        
        if not packets:
            print("No packets were captured. Make sure you have traffic on the interface.")
            return

        print(f"Captured {len(packets)} packets.")

        # 2. Extract flows and features
        flow_extractor = FlowExtractor()
        flows = flow_extractor.process_packets(packets)

        print(f"Extracted {len(flows)} flows.")

        # 3. Print extracted features for each flow
        for i, flow in enumerate(flows):
            print(f"\n--- Flow {i+1} ---")
            print(f"Flow ID: {flow.flow_id}")
            print(f"Features:")
            for feature, value in flow.features.items():
                print(f"  {feature}: {value}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you are running this script with sufficient privileges (e.g., using sudo).")
        print("Also, make sure the specified interface exists and is active.")

if __name__ == "__main__":
    main()
