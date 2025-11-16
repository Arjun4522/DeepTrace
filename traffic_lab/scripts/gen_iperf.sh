#!/bin/bash
# Generate streaming/bursty traffic between traffic_gen1 and traffic_gen2

echo "Generating streaming/bursty traffic..."

# Run iperf3 client in infinite loop with various modes
docker exec traffic_gen2 sh -c "
while true; do
  iperf3 -c traffic_gen1 -t 10 -b 1M
  iperf3 -c traffic_gen1 -t 10 -b 10M
  iperf3 -c traffic_gen1 -t 10 -R
  sleep 5
done
" &

echo "Streaming/bursty traffic generation started."