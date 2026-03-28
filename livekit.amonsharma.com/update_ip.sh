#!/usr/bin/env bash
ip=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
if [ -z "$ip" ]; then
  ip=$(ip addr show | grep "inet " | grep -v 127.0.0. | head -1 | cut -d" " -f6 | cut -d/ -f1)
fi
if [ -n "$ip" ]; then
  sed -i "s/localhost:5349/$ip:5349/g" /opt/livekit/caddy.yaml
fi
