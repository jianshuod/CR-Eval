#!/bin/bash

CPUS=$(nproc)

MEM_TOTAL_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')

MEM_TOTAL_MB=$((MEM_TOTAL_KB / 1024))

CPUS_ALLOC=$((CPUS * 80 / 100))
MEM_ALLOC_MB=$((MEM_TOTAL_MB * 80 / 100))

docker run -d \
  --name libreoffice \
  --security-opt seccomp=unconfined \
  -e PUID=1000 \
  -e PGID=1000 \
  -e TZ=Etc/UTC \
  -p 3000:3000 \
  -p 3001:3001 \
  --cpus=$CPUS_ALLOC \
  --memory="${MEM_ALLOC_MB}m" \
  --restart unless-stopped \
  my-libreoffice:latest
