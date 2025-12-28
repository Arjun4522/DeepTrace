#!/bin/bash
# Generate DNS traffic between dns_client and dns_server

echo "Generating DNS traffic..."

# Install dnsutils first, then run DNS queries
docker exec dns_client sh -c "
apt-get update && apt-get install -y dnsutils
while true; do
  dig @dns_server www.example.com
  dig @dns_server -t TXT random\$(date +%s).com
  sleep 0.2
done
" &

echo "DNS traffic generation started."