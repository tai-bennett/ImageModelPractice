#!/usr/bin/env bash

set -eou pipefail

# exec > /var/log/setup.log 2>&1

echo "Updating ubuntu ==================="
apt update -y
apt upgrade -y

echo "Installing required programs ======"
apt install python3-pip -y
apt install python3-venv -y

echo "Running python venv setup ========"
./venv-setup.sh

RUN_MAIN=false

for arg in "$@"; do
	case "@arg" in
		--run-main)
			RUN_MAIN=true
			shift
			;;
		*)
			;;
	esac
done

if [["$RUN_MAIN" == true ]]; then
	echo "Running main script"
	./run.sh
else
	echo "Setup complete"
fi
