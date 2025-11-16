#!/bin/bash
# Generate scanning traffic from scanner to web_server

echo "Generating scanning traffic..."

# Run nmap scans in infinite loop
docker exec scanner sh -c "
while true; do
  nmap -sS web_server
  sleep 5
done
" &

echo "Scanning traffic generation started."