#!/bin/bash
# Generate SSH/SFTP traffic between file_client and file_server

echo "Generating SSH/SFTP traffic..."

# Run infinite loop of file transfers
docker exec file_client sh -c "
while true; do
  echo 'test data' > /tmp/testfile
  sftp -o StrictHostKeyChecking=no -P 2222 foo@file_server <<< \$'put /tmp/testfile'
  rm /tmp/testfile
  sleep 1
done
" &

echo "SSH/SFTP traffic generation started."