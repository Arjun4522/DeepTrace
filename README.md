# DeepTrace: Deep Packet Inspection–based Behavioral Flow Embedding System

DeepTrace is a Python-based system for deep packet inspection (DPI) and behavioral flow embedding. It captures network packets, extracts detailed features from the flows, and is designed to generate numerical embeddings for anomaly detection, traffic analysis, and other network security applications.

## Features

*   **Live Packet Capture:** Captures live network traffic from a specified interface using Scapy.
*   **PCAP File Reading:** Can also process packets from existing PCAP files.
*   **Advanced Feature Extraction:** Extracts a rich set of features from network flows, categorized into three tiers:
    *   **Temporal Features:** Per-packet dynamics like inter-arrival time (`Δt`), packet size, direction, and entropy.
    *   **Statistical Features:** Aggregated flow-level statistics such as flow duration, total bytes, mean/std of packet sizes and inter-arrival times, and more.
    *   **Protocol Features:** Identification of transport-layer (TCP, UDP, ICMP) and application-layer (HTTP, TLS, DNS) protocols, along with TCP flag summaries.
*   **Transformer-based Embeddings (WIP):** A Transformer-based model is being developed to generate 128-D embeddings from the extracted flow features.
*   **Extensible Architecture:** The modular design allows for easy extension and customization.

## Architecture

The DeepTrace system is composed of the following key components:

1.  **Packet Capture (`capture.py`):** Responsible for capturing packets from a live network interface or reading them from a PCAP file.
2.  **Feature Extractor (`features.py`):** The core component that processes raw packets, groups them into bidirectional flows, and extracts the three tiers of features (Temporal, Statistical, and Protocol).
3.  **Embedding Model (`model.py`):** A Transformer-based neural network (currently under development) that takes the extracted features and generates a fixed-size numerical embedding for each flow.
4.  **Storage (`storage.py`):** (Planned) A module for storing the extracted features and generated embeddings in a database like Redis or a vector database like FAISS.
5.  **CLI (`cli.py`):** (Planned) A command-line interface for interacting with the DeepTrace system, allowing for querying, summarization, and anomaly detection.

## Setup and Installation

This project is developed on NixOS and uses a `shell.nix` file to provide a consistent development environment.

### Prerequisites

*   [Nix](https://nixos.org/download.html) package manager.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Arjun4522/DeepTrace.git
    cd DeepTrace
    ```

2.  **Enter the Nix shell:**
    This command will automatically download and configure all the necessary dependencies, including Python, Scapy, and other required libraries.
    ```bash
    nix-shell
    ```

## Usage

The `main.py` script is the main entry point for running the packet capture and feature extraction process.

### Live Capture

To capture packets from a live network interface, run the following command from within the `nix-shell`:

```bash
sudo PYTHONPATH=$PYTHONPATH python3 main.py -i wlo1
```

*   You may need to change the `interface` variable in `main.py` to match your system's network interface (e.g., `eth0`, `enp0s3`).
*   The script will capture packets indefinitely. Press `Ctrl+C` to stop the capture and see the extracted flow features printed to the console in JSON format.

## Project Structure

```
deeptraace/
├── deeptraace/
│   ├── __init__.py
│   ├── capture.py        # Packet capture logic
│   ├── features.py       # Flow and feature extraction logic
│   ├── model.py          # Transformer embedding model (WIP)
│   ├── storage.py        # Storage logic (planned)
│   └── cli.py            # CLI logic (planned)
├── tests/
│   └── ...               # Unit tests
├── main.py               # Main script for running the system
├── requirements.txt      # Python dependencies (for reference)
├── shell.nix             # Nix shell environment definition
└── README.md             # This file
```

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
