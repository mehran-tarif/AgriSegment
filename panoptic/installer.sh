#!/bin/bash
echo "🎭 Installing Mask2Former Plant Segmentation Web App"
echo "=================================================="

# Install required packages
echo "📦 Installing Python packages..."

# PyTorch and vision
pip install torch torchvision torchaudio

# Transformers for Mask2Former
echo "📦 Installing Transformers..."
pip install transformers datasets

# FastAPI web server
echo "📦 Installing FastAPI..."
pip install fastapi uvicorn jinja2 python-multipart

# Image processing
echo "📦 Installing image processing libraries..."
pip install pillow matplotlib

# Scientific computing
echo "📦 Installing NumPy and OpenCV..."
pip install numpy opencv-python scipy

# Create directories
echo "📁 Creating directories..."
mkdir -p templates uploads output

# Move index.html to templates directory if it exists
echo "📄 Setting up templates..."
if [ -f "index.html" ]; then
    mv index.html templates/
    echo "✅ Moved index.html to templates/"
fi

echo "✅ Installation complete!"
echo ""
echo "🚀 Quick Start:"
echo "   python3 main.py"
echo ""
echo "🌐 Then open: http://localhost:8003"
echo ""
echo "💡 Features:"
echo "   • Upload multiple images"
echo "   • Choose segmentation type (semantic/instance/panoptic)"
echo "   • Select model size (base/large)"
echo "   • Adjustable confidence threshold"
echo "   • 2x3 detailed visualization results"
echo "   • Download all results as ZIP"
echo ""
echo "🔧 Requirements:"
echo "   • Python 3.8+"
echo "   • GPU recommended for faster processing"
echo "   • ~2-4GB RAM for base models"
echo "   • ~4-8GB RAM for large models"