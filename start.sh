#!/bin/bash

echo "=============================================="
echo "3D Generative Design for Additive Manufacturing"
echo "=============================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p checkpoints
mkdir -p data/shapenet
mkdir -p static
mkdir -p templates

# Check if dataset exists
if [ ! "$(ls -A data/shapenet)" ]; then
    echo ""
    echo "No dataset found. Creating sample synthetic dataset..."
    python download_shapenet.py --source sample --num_samples 100
fi

echo ""
echo "=============================================="
echo "Starting the web application..."
echo "=============================================="
echo ""
echo "Open your browser and go to:"
echo "http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the application
python app.py
