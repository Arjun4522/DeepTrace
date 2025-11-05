from deeptrace.capture import PacketCapture
from deeptrace.features import FlowExtractor
import json
import argparse
import os, datetime

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="DeepTrace: Live packet capture and feature extraction.")
    parser.add_argument("-i", "--interface", required=True, help="Network interface to capture packets from (e.g., eth0, wlo1)")
    args = parser.parse_args()

    interface = args.interface # Get interface from arguments
    
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

        # --- Save dataset ---
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
