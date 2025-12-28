# DeepTrace: Deep Packet Inspection–based Behavioral Flow Embedding System

DeepTrace is a system for deep packet inspection (DPI) and behavioral flow embedding. It captures network packets, extracts detailed features from the flows, and is designed to generate numerical embeddings for anomaly detection, traffic analysis, and other network security applications.

## Features

*   **Live Packet Capture:** Captures live network traffic from a specified interface using Scapy.
*   **PCAP File Reading:** Can also process packets from existing PCAP files.
*   **Advanced Feature Extraction:** Extracts a rich set of features from network flows, categorized into three tiers:
    *   **Temporal Features:** Per-packet dynamics like inter-arrival time (`Δt`), packet size, direction, and entropy.
    *   **Statistical Features:** Aggregated flow-level statistics such as flow duration, total bytes, mean/std of packet sizes and inter-arrival times, and more.
    *   **Protocol Features:** Identification of transport-layer (TCP, UDP, ICMP) and application-layer (HTTP, TLS, DNS) protocols, along with TCP flag summaries.
*   **Transformer-based Embeddings:** A fully implemented Transformer-based model that generates 64-D embeddings from the extracted flow features.
*   **Real-time Vector Database:** FAISS-based vector storage for fast similarity search and retrieval of network flows.
*   **Streaming Pipeline:** Continuous packet capture with real-time embedding generation and indexing.
*   **Docker-based Traffic Generation Lab:** Complete lab environment for generating diverse network traffic patterns for model training.
*   **Web Dashboard:** Modern React-based frontend for real-time monitoring and visualization.
*   **Extensible Architecture:** The modular design allows for easy extension and customization.

## Architecture

The DeepTrace system is composed of the following key components:

1.  **Packet Capture (`capture.py`):** Responsible for capturing packets from a live network interface or reading them from a PCAP file.
2.  **Feature Extractor (`features.py`):** The core component that processes raw packets, groups them into bidirectional flows, and extracts the three tiers of features (Temporal, Statistical, and Protocol).
3.  **Embedding Model (`model.py`):** A Transformer-based neural network that takes the extracted features and generates fixed-size 64-dimensional numerical embeddings for each flow.
4.  **Vector Storage (`storage.py`):** FAISS-based vector database for storing flow embeddings with metadata, enabling fast similarity search and retrieval.
5.  **Streaming Pipeline (`main.py`):** Real-time pipeline that continuously captures packets, generates embeddings, and updates the vector store.
6.  **Traffic Generation Lab (`traffic_lab/`):** Docker-based environment with containers and scripts for generating diverse network traffic patterns.
7.  **Web Dashboard (`web/`):** Modern React/Vite frontend with real-time monitoring, analytics, and flow visualization.
8.  **CLI (`cli.py`):** Command-line interface for interacting with the DeepTrace system.

## Setup and Installation

This project is developed on NixOS and uses a `shell.nix` file to provide a consistent development environment.

### Prerequisites

*   [Nix](https://nixos.org/download.html) package manager, or
*   Python 3.8+ with pip
*   Docker (for traffic generation lab)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Arjun4522/DeepTrace.git
    cd DeepTrace
    ```

2.  **Option A: Using Nix (Recommended)**
    
    Enter the Nix shell to automatically configure all dependencies:
    ```bash
    nix-shell
    ```

3.  **Option B: Using pip**
    
    Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Web Dashboard Setup**
    
    Navigate to the web directory and install Node.js dependencies:
    ```bash
    cd web
    npm install
    ```
    
    Start the development server:
    ```bash
    npm run dev
    ```

## Usage

DeepTrace has two main modes of operation:

### Basic Mode: Single Capture

Capture packets once, extract flows, and save to JSONL files:

```bash
# Capture 100 packets
sudo PYTHONPATH=$PYTHONPATH python3 main.py -i wlo1 --basic --count 100

# Capture continuously until Ctrl+C
sudo PYTHONPATH=$PYTHONPATH python3 main.py -i wlo1 --basic
```

*   Replace `wlo1` with your network interface (e.g., `eth0`, `docker0`).
*   Flows are saved to `train/flows_TIMESTAMP.jsonl`.

### CLI Mode: Interactive Vector Store Query

Query and analyze captured flows stored in the vector store:

```bash
# Interactive mode with vector store
python3 -m deeptrace.cli -s ./vector_store

# With model checkpoint for similarity search
python3 -m deeptrace.cli -s ./vector_store -c models/checkpoints/model_epoch_50.pth

# Show statistics only
python3 -m deeptrace.cli -s ./vector_store --stats

# Search by IP address
python3 -m deeptrace.cli -s ./vector_store --ip 192.168.1.100

# Find similar flows
python3 -m deeptrace.cli -s ./vector_store -c models/checkpoints/model_epoch_50.pth --similar 0

# List first 20 flows
python3 -m deeptrace.cli -s ./vector_store --list 20
```

**CLI Commands:**
- `list [N]` - List all flows (limit N, default 10)
- `show <index>` - Show detailed flow information
- `ip <address>` - Search by IP address
- `port <number>` - Search by port number
- `protocol <name>` - Search by protocol (TCP/UDP/HTTPS/etc)
- `similar <index> [k]` - Find k similar flows (requires --checkpoint)
- `anomalies [thresh] [k]` - Find anomalies (requires --checkpoint)
- `stats` - Show store statistics
- `help` - Show help
- `exit`, `quit` - Exit

### Streaming Mode: Real-time Embedding Pipeline

Continuously capture packets, generate embeddings, and store in vector database:

```bash
# Basic streaming with default settings
sudo PYTHONPATH=$PYTHONPATH python3 main.py -i wlo1 --stream \
    -c models/checkpoints/model_epoch_50.pth

# Advanced: Custom capture interval and batch size
sudo PYTHONPATH=$PYTHONPATH python3 main.py -i wlo1 --stream \
    -c models/checkpoints/model_epoch_50.pth \
    -t 5 \
    -b 100 \
    -s ./my_vector_store

# Resume from existing vector store
sudo PYTHONPATH=$PYTHONPATH python3 main.py -i wlo1 --stream \
    -c models/checkpoints/model_epoch_50.pth \
    --load

# Start web dashboard alongside streaming
sudo PYTHONPATH=$PYTHONPATH python3 main.py -i wlo1 --stream \
    -c models/checkpoints/model_epoch_50.pth &
cd web && npm run dev
```

**Options:**
*   `-i, --interface`: Network interface to capture from
*   `-c, --checkpoint`: Path to model checkpoint (required for streaming)
*   `-t, --interval`: Capture interval in seconds (default: 10)
*   `-b, --batch-size`: Batch size for embeddings (default: 50)
*   `-s, --storage`: Path to vector store directory (default: ./vector_store)
*   `-d, --device`: Device for inference: cpu or cuda (default: cpu)
*   `--load`: Load existing vector store if available

**Example Output:**
```
======================================================================
🚀 DeepTrace Realtime Embedding Pipeline Started
======================================================================
  Interface:        wlo1
  Model:            models/checkpoints/model_epoch_50.pth
  Storage:          ./vector_store
  Capture interval: 10s
  Batch size:       50
  Device:           cpu
======================================================================

[03:02:10] 📦 Processing 137 packets...
  ✓ Extracted 16 flows
  🧠 Generating embeddings...
  ✓ Generated 16 embeddings in 0.00s
    (9244.9 flows/sec)

──────────────────────────────────────────────────────────────────────
📊 Statistics (Uptime: 00:00:10)
──────────────────────────────────────────────────────────────────────
  Packets captured:     137
  Flows extracted:      16
  Embeddings generated: 16
  Vector store size:    16

  Protocol distribution:
    TCP     : █████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░  42.0% (21)
    UDP     : ████████████████████████████░░░░░░░░░░░░░░░░░░░░  58.0% (29)
──────────────────────────────────────────────────────────────────────
```

## Traffic Generation Lab

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
    sudo python3 ../main.py -i docker0 --stream \
        -c ../models/checkpoints/model_epoch_50.pth
    ```

5.  Monitor traffic in the web dashboard:
    ```bash
    cd ../web
    npm run dev
    ```
    Open http://localhost:3000 to view real-time analytics

6.  Stop containers when finished:
    ```bash
    docker-compose down
    ```

## Model Training

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
  - Entropy statistics (mean and std)
  - Byte ratio (forward/total)
  - Protocol indicators (TCP/UDP one-hot encoding)

##### Transformer Encoder
- **Input Projection**: Linear layer mapping 12-D input to 64-D hidden representation
- **Input Normalization**: Layer normalization for stable training
- **2-Layer Transformer Encoder**: Each layer with:
  - 4 attention heads for multi-scale pattern recognition
  - 128-D feedforward dimension (2× hidden dimension)
  - Dropout (0.1) for regularization
  - Pre-layer normalization for training stability
- **Batch-first format**: Efficient processing of batched inputs

##### Output Layer
- **Global Average Pooling**: Aggregates sequence information
- **Output Normalization**: Layer normalization before final projection
- **Linear Projection**: Maps 64-D representation to 64-D embedding space
- **L2 Normalized Embeddings**: Unit-length vectors for cosine similarity

#### Training Process

##### Two-Stage Approach
1. **Pre-training (50 epochs)**: Contrastive learning with AdamW optimizer
2. **Fine-tuning**: Triplet loss for refined clustering (optional)

##### Key Features
- **Mixed Precision Training**: FP16 for GPU efficiency
- **Gradient Scaling**: For stable mixed precision training
- **Checkpointing**: Model saved every 5 epochs
- **Batch Sizes**: 32 for pre-training, 16 for fine-tuning

#### Technical Details

##### Hyperparameters
- Input dimension: 12 features
- Hidden dimension: 64
- Output dimension: 64 (embedding size)
- Transformer layers: 2
- Attention heads: 4 per layer
- Feedforward dimension: 128
- Dropout rate: 0.1
- Learning rates: 1e-4 (pre-train), 5e-5 (fine-tune)
- Contrastive loss margin: 0.5

##### Evaluation Methods
- **t-SNE Visualization**: 64-D to 2-D projection for cluster visualization
- **Silhouette Score**: Measures clustering quality (range: -1 to 1)
- **K-Means Clustering**: Unsupervised grouping validation
- **Davies-Bouldin Index**: Lower values indicate better cluster separation

#### Model Performance
- **Inference Speed**: ~10,000-17,000 flows/second on CPU
- **Model Size**: 72,192 parameters
- **Memory Footprint**: ~280 KB (model weights)
- **Embedding Dimension**: 64-D (suitable for FAISS indexing)

#### Deployment

The trained model supports:
- **Real-time Embedding Generation**: Fast inference for streaming traffic
- **Batch Processing**: Efficient parallel processing of multiple flows
- **Anomaly Detection**: Distance-based outlier identification
- **Behavioral Clustering**: Automatic traffic pattern grouping
- **Similarity Search**: Find flows with similar behavior
- **RAG Applications**: Context retrieval for LLM-powered analysis

Trained checkpoints are available in `models/checkpoints/` with `model_epoch_50.pth` being the recommended production model.

## Web Dashboard

The DeepTrace web dashboard provides a modern interface for real-time network monitoring and analysis.

### Dashboard Features

- **Real-time Statistics**: Live updates of total flows, packets, bytes, and average flow size
- **Protocol Distribution**: Visual breakdown of network protocols with progress bars
- **Application Protocol Analysis**: Top application protocols with usage statistics
- **Recent Flows Table**: Detailed view of recent network flows with source/destination information
- **Live Indicators**: Real-time updates with live status indicators
- **Responsive Design**: Works on desktop and mobile devices

### Web Architecture

- **Frontend**: React 18 with Vite for fast development
- **Styling**: Tailwind CSS for responsive design
- **State Management**: React Query for data fetching and caching
- **Icons**: Lucide React for consistent iconography
- **Real-time Updates**: WebSocket connections for live data
- **API Integration**: RESTful endpoints for statistics and flow data

### Getting Started with Web Dashboard

1. Navigate to the web directory:
   ```bash
   cd web
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open your browser to http://localhost:3000

5. Ensure the backend is running to provide data to the dashboard

## Vector Store and Similarity Search

DeepTrace uses FAISS (Facebook AI Similarity Search) for efficient storage and retrieval of flow embeddings.

### Vector Store Features

- **Fast Similarity Search**: L2 distance-based nearest neighbor search
- **Metadata Storage**: Full flow details stored alongside embeddings
- **Persistent Storage**: Save/load vector stores to disk
- **Streaming Buffer**: Batch optimization for real-time ingestion
- **Protocol Filtering**: Query by protocol, IP, or custom filters

## Project Structure

```
deeptrace/
├── deeptrace/
│   ├── __init__.py
│   ├── capture.py        # Packet capture logic
│   ├── features.py       # Flow and feature extraction
│   ├── model.py          # Embedding model and inference
│   ├── storage.py        # FAISS vector store
│   └── cli.py            # CLI interface for vector store query
├── models/
│   ├── deeptrace.py      # Model training script
│   ├── deeptrace.ipynb   # Training notebook
│   └── checkpoints/      # Pre-trained model weights
│       └── model_epoch_50.pth
├── traffic_lab/
│   ├── docker-compose.yml     # Container definitions
│   └── scripts/               # Traffic generation
│       ├── gen_http.sh
│       ├── gen_dns.sh
│       ├── gen_ssh.sh
│       ├── gen_iperf.sh
│       ├── gen_scan.sh
│       ├── gen_exfil.sh
│       └── generate_all.sh
├── web/                     # React web dashboard
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── contexts/       # React contexts
│   │   ├── pages/         # Page components
│   │   ├── App.jsx         # Main app component
│   │   └── main.jsx        # Entry point
│   ├── app.py              # FastAPI backend
│   ├── package.json        # Node.js dependencies
│   ├── vite.config.js      # Vite configuration
│   └── tailwind.config.js  # Tailwind CSS config
├── train/                     # Training data (JSONL)
├── vector_store/              # FAISS index and metadata
│   ├── faiss.index
│   └── metadata.pkl
├── tests/
│   └── ...               # Unit tests
├── main.py               # Main entry point
├── requirements.txt      # Python dependencies
├── shell.nix             # Nix environment
└── README.md             # This file
```

## Performance Benchmarks

| Metric | Value |
|--------|-------|
| Embedding Generation | ~10-17k flows/sec (CPU) |
| Model Parameters | 72,192 |
| Embedding Dimension | 64-D |
| Vector Store Indexing | Real-time (<1ms per flow) |
| Memory per Flow | ~1 KB (with metadata) |
| Capture Throughput | Handles Gbps networks |

## Use Cases

### Network Security
- Real-time anomaly detection
- Behavioral clustering of traffic
- Zero-day attack detection via embedding distance
- Lateral movement detection

### Traffic Analysis
- Protocol identification and classification
- Application fingerprinting
- Performance monitoring
- Capacity planning

### Research
- Network traffic datasets with embeddings
- Comparative analysis of traffic patterns
- Embedding space visualization
- Transfer learning for specialized networks

## Troubleshooting

### Permission Issues
```bash
# DeepTrace requires root for packet capture
sudo python3 main.py -i wlo1 --stream -c models/checkpoints/model_epoch_50.pth
```

### Interface Not Found
```bash
# List available interfaces
ip link show
# or
ifconfig
```

### No Packets Captured
```bash
# Generate traffic in another terminal
ping 8.8.8.8
curl https://google.com

# Or use the traffic generation lab
cd traffic_lab && docker-compose up -d
```

### Model Loading Issues
```bash
# Ensure checkpoint exists
ls -lh models/checkpoints/model_epoch_50.pth

# Check Python environment has all dependencies
pip install torch scapy numpy faiss-cpu
```

## Future Enhancements

- [ ] RAG-based AI assistant for natural language queries
- [ ] Advanced anomaly detection algorithms
- [ ] Multi-model ensemble for improved accuracy
- [ ] Distributed deployment for high-throughput networks
- [ ] Integration with SIEM systems
- [ ] Export to multiple formats (Parquet, Arrow, etc.)
- [ ] Real-time alerting system
- [ ] Historical data analysis
- [ ] Custom dashboard widgets
- [ ] User authentication and authorization


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with PyTorch, Scapy, and FAISS
- Inspired by modern network security research
- Thanks to the open-source community

---

**Note**: DeepTrace is under active development. For the latest updates and documentation, visit the [GitHub repository](https://github.com/Arjun4522/DeepTrace).