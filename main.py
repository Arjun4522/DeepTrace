from deeptraace.capture import PacketCapture
from deeptraace.features import FlowExtractor
import json

def main():
    interface = "wlo1" # Changed to wlo1 based on user's output
    
    print(f"Testing live packet capture on interface: {interface}")
    
    try:
        # 1. Capture live packets
        packet_capturer = PacketCapture(interface=interface)
        packets = packet_capturer.capture_packets() # Capture indefinitely until Ctrl+C

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

        # 3. Print extracted features for each flow in the new structured format
        for i, flow in enumerate(flows):
            print(f"\n--- Flow {i+1} ---")
            # Convert flow object to a dictionary for JSON-like printing
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

if __name__ == "__main__":
    main()