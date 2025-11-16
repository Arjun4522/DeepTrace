#!/bin/bash
# Run all traffic generation scripts in parallel

echo "Starting all traffic generation scripts..."

# Make sure all scripts are executable
chmod +x gen_http.sh
chmod +x gen_dns.sh
chmod +x gen_ssh.sh
chmod +x gen_iperf.sh
chmod +x gen_scan.sh
chmod +x gen_exfil.sh

# Run all scripts
./gen_http.sh
./gen_dns.sh
./gen_ssh.sh
./gen_iperf.sh
./gen_scan.sh
./gen_exfil.sh

echo "All traffic generation scripts started."
echo "Use 'docker-compose down' to stop containers when finished."