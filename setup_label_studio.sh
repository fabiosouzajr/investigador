#!/bin/bash
# Setup script for Label Studio Docker installation

echo "Setting up Label Studio data directory..."

# Create mydata directory if it doesn't exist
mkdir -p mydata

# Set proper ownership and permissions for UID 1001 (Label Studio container user)
sudo chown -R 1001:1001 mydata
sudo chmod 755 mydata

echo "Directory setup complete!"
echo ""
echo "You can now run Label Studio with:"
echo "  docker run -it -p 8080:8080 -v \$(pwd)/mydata:/label-studio/data heartexlabs/label-studio:latest"
echo ""
echo "Or if you need sudo:"
echo "  sudo docker run -it -p 8080:8080 -v \$(pwd)/mydata:/label-studio/data heartexlabs/label-studio:latest"

