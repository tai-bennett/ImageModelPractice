#!/usr/bin/env bash

set -e

# exec > /var/log/setup.log 2>&1

echo "Updating ubuntu ==================="
apt update -y
apt upgrade -y

echo "Installing required programs ======"
apt install python3-pip -y
apt install python3-venv -y

echo "Running python venv setup ========"
./venv-setup.sh
