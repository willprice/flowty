#!/usr/bin/env bash
set -ex

[[ $UID -eq 0 ]] || { echo "Run as root"; exit 1; }
CONTAINER_NAME="flowty-dev"
docker build -t flowty-dev --file Dockerfile.dev .

if docker ps -a | grep "$CONTAINER_NAME" 2>&1 >/dev/null; then
    CONTAINER_EXISTS=1
else
    CONTAINER_EXISTS=0
fi

if docker ps | grep "$CONTAINER_NAME" 2>&1 >/dev/null; then
    CONTAINER_IS_RUNNING=1
else
    CONTAINER_IS_RUNNING=0
fi

if [[ $CONTAINER_EXISTS -eq 0 ]]; then
    #   --cap-add=SYS_PTRACE
    # Supports the use of tools like GDB that use ptrace

    #    --security-opt seccomp=unconfined
    # Disable secure computing mode, as when enabled, it disables the availability
    # of syscalls used in gdb and the like

    docker run -it \
        --name $CONTAINER_NAME \
        --runtime=nvidia \
        --mount type=bind,source="$PWD",target="/app" \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        flowty-dev
else

    if [[ $CONTAINER_IS_RUNNING -eq 0 ]]; then
        docker start "$CONTAINER_NAME"
    fi
    docker attach "$CONTAINER_NAME"
fi

