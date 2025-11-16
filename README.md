# DeepTrace: Deep Packet Inspection–based Behavioral Flow Embedding System

DeepTrace is a system for deep packet inspection (DPI) and behavioral flow embedding. It captures network packets, extracts detailed features from the flows, and generates numerical embeddings for anomaly detection, traffic analysis, and other network security applications.

## Features

*   **Live Packet Capture:** Captures live network traffic from a specified interface using Scapy.
*   **PCAP File Reading:** Can also process packets from existing PCAP files.
*   **Advanced Feature Extraction:** Extracts a rich set of features from network flows, categorized into three tiers:
    *   **Temporal Features:** Per-packet dynamics like inter-arrival time (`Δt`), packet size, direction, and entropy.
    *   **Statistical Features:** Aggregated flow-level statistics such as flow duration, total bytes, mean/std of packet sizes and inter-arrival times, and more.
    *   **Protocol Features:** Identification of transport-layer (TCP, UDP, ICMP) and application-layer (HTTP, TLS, DNS) protocols, along with TCP flag summaries.
*   **Transformer-based Embeddings:** A fully implemented Transformer-based model that generates 64-D embeddings from the extracted flow features for clustering and anomaly detection.
*   **Docker-based Traffic Generation Lab:** Complete lab environment for generating diverse network traffic patterns for model training.
*   **Extensible Architecture:** The modular design allows for easy extension and customization.

## Architecture

The DeepTrace system is composed of the following key components:

1.  **Packet Capture (`capture.py`):** Responsible for capturing packets from a live network interface or reading them from a PCAP file.
2.  **Feature Extractor (`features.py`):** The core component that processes raw packets, groups them into bidirectional flows, and extracts the three tiers of features (Temporal, Statistical, and Protocol).
3.  **Embedding Model (`models/deeptrace.py`):** A fully implemented Transformer-based neural network that takes the extracted features and generates fixed-size numerical embeddings for each flow.
4.  **Traffic Generation Lab (`traffic_lab/`):** Docker-based environment with containers and scripts for generating diverse network traffic patterns.
5.  **Storage (`storage.py`):** Module for storing the extracted features and generated embeddings in JSONL format.
6.  **CLI (`cli.py`):** Command-line interface for interacting with the DeepTrace system.

## Setup and Installation

This project is developed on NixOS and uses a `shell.nix` file to provide a consistent development environment.

### Prerequisites

*   [Nix](https://nixos.org/download.html) package manager.
*   Docker (for traffic generation lab)

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

3.  **Install Docker:**
    Follow Docker installation instructions for your system.

## Running Tests

To run the unit and system tests, execute the following command from the root directory of the project within the `nix-shell`:

```bash
python3 -m unittest discover -s tests
```

This command will discover and run all test cases located in the `tests/unit` and `tests/system` directories.

## Usage

The `main.py` script is the main entry point for running the packet capture and feature extraction process.

### Live Capture

To capture packets from a live network interface, run the following command from within the `nix-shell`:

```bash
sudo PYTHONPATH=$PYTHONPATH python3 main.py -i docker0
```

*   Replace `docker0` with your desired network interface (e.g., `eth0`, `wlan0`).
*   The script will capture packets indefinitely. Press `Ctrl+C` to stop the capture and see the extracted flow features printed to the console in JSON format.

### Traffic Generation Lab

To use the Docker-based traffic generation lab:

1.  Navigate to the traffic lab directory:
    ```bash
    cd traffic_lab
    ```

2.  Start the Docker containers:
    ```bash
    docker-compose up -d
    ```

3.  Run all traffic generation scripts:
    ```bash
    cd scripts
    sudo ./generate_all.sh
    ```

4.  Capture traffic in another terminal:
    ```bash
    sudo PYTHONPATH=$PYTHONPATH python3 ../main.py -i docker0
    ```

5.  Stop containers when finished:
    ```bash
    docker-compose down
    ```

### Model Training

To train the embedding model:

1.  Ensure you have generated training data in the `train/` directory
2.  Navigate to the models directory:
    ```bash
    cd models
    ```
3.  Run the training script:
    ```bash
    python3 deeptrace.py
    ```

### Embedding Model Architecture

The DeepTrace embedding model is a Transformer-based neural network specifically designed for generating semantic embeddings from network flow features. It transforms 12-dimensional flow feature vectors into 64-dimensional embeddings that capture behavioral patterns for anomaly detection and clustering.

#### Model Design

##### Input Layer
- **12-D Feature Vector**: Each flow is represented by 12 key features:
  - Flow duration, total bytes, packet count
  - Mean/std of packet sizes and inter-arrival times
  - Entropy statistics
  - Byte ratio
  - Protocol indicators (TCP/UDP)

##### Transformer Encoder
- **Linear Projection**: Maps 12-D input to 64-D hidden representation
- **2-Layer Transformer**: With 4 attention heads per layer
- **Pre-Layer Normalization**: For training stability
- **GELU Activation**: For non-linear transformations
- **Dropout (0.1)**: For regularization

##### Output Layer
- **Linear Projection**: 64-D transformer output to 64-D embedding
- **Layer Normalization**: For stable embeddings
- **L2 Normalization**: Ensures unit-length embeddings for cosine similarity

#### Training Process

##### Two-Stage Approach
1. **Pre-training (50 epochs)**: Contrastive learning with AdamW optimizer
2. **Fine-tuning**: Triplet loss for refined clustering (planned)

##### Key Features
- **Mixed Precision Training**: FP16 for efficiency on GPU
- **Gradient Scaling**: For stable training
- **Checkpointing**: Saves every 5 epochs
- **Batch Size**: 32 for pre-training, 16 for fine-tuning

#### Technical Details

##### Hyperparameters
- Hidden dimension: 64
- Attention heads: 4
- Transformer layers: 2
- Learning rates: 1e-4 (pre-train), 5e-5 (fine-tune)
- Margin for contrastive loss: 0.5

##### Evaluation Methods
- **t-SNE Visualization**: 64-D to 2-D embedding projection
- **Silhouette Score**: Clustering quality metric
- **K-Means Clustering**: Unsupervised grouping validation
- **Davies-Bouldin Index**: Cluster separation measurement

#### Deployment

The model can be used for:
- Real-time flow embedding generation
- Anomaly detection through embedding distance metrics
- Behavioral clustering of network traffic
- Similarity search between network flows

Trained checkpoints are available in `models/checkpoints.zip` with the final model at epoch 50 representing the best performing version.

## Project Structure

```
deeptrace/
├── deeptrace/
│   ├── __init__.py
│   ├── capture.py        # Packet capture logic
│   ├── features.py       # Flow and feature extraction logic
│   ├── model.py          # Core model interface
│   ├── storage.py        # Storage logic
│   └── cli.py            # CLI logic
├── models/
│   ├── deeptrace.py      # Transformer embedding model implementation
│   ├── deeptrace.ipynb   # Jupyter notebook with training code
│   └── checkpoints.zip   # Pre-trained model checkpoints
├── traffic_lab/
│   ├── docker-compose.yml     # Docker container definitions
│   └── scripts/               # Traffic generation scripts
│       ├── gen_http.sh
│       ├── gen_dns.sh
│       ├── gen_ssh.sh
│       ├── gen_iperf.sh
│       ├── gen_scan.sh
│       ├── gen_exfil.sh
│       └── generate_all.sh
├── train/                     # Training data directory
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
