#!/bin/bash
# 2. Loop through all local images
# We use -v "<none>" to avoid empty/dangling layers
for img in $(podman images --format "{{.Repository}}:{{.Tag}}" | grep -v "<none>"); do
    
    # Skip if the image is already correctly tagged for Docker Hub
    if [[ "$img" == "docker.io/$USER/"* ]]; then
        echo ">>> Skipping $img (already tagged correctly)"
        continue
    fi

    # Extract the base image name (e.g., 'my-app:latest')
    CLEAN_NAME=$(echo $img | awk -F'/' '{print $NF}')
    REMOTE_NAME="docker.io/$USER/$CLEAN_NAME"
    
    echo "----------------------------------------------------------"
    echo "Target: $REMOTE_NAME"
    
    # Create the Docker Hub tag
    podman tag $img $REMOTE_NAME
    
    # Push to Docker Hub
    echo "Pushing..."
    if podman push $REMOTE_NAME; then
        echo "Success! Removing old local tag: $img"
        # This removes the 'localhost' or untagged alias
        podman rmi $img
    else
        echo "FAILED to push $REMOTE_NAME. Keeping original tag."
    fi
done
