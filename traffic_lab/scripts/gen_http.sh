#!/bin/bash
# Generate HTTP traffic between web_client and web_server

echo "Generating HTTP traffic..."

# Run infinite loop of HTTP requests
docker exec web_client sh -c "
while true; do
  curl -s http://web_server/ > /dev/null
  sleep 0.5
done
" &

echo "HTTP traffic generation started."