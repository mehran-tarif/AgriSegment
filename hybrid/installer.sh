#!/bin/bash
echo "🌱 Installing SegFormer + SAM Plant Segmentation"
echo "================================================"

# Install required packages
echo "📦 Installing Python packages..."
pip install torch torchvision torchaudio

# Segformer dependencies
echo "📦 Installing Transformers and HuggingFace..."
pip install transformers
pip install datasets

# SAM dependencies  
echo "📦 Installing SAM..."
pip install segment-anything

# FastAPI web server
echo "📦 Installing FastAPI..."
pip install fastapi uvicorn jinja2 python-multipart

# Image processing
echo "📦 Installing OpenCV and PIL..."
pip install opencv-python pillow

# Scientific computing
echo "📦 Installing NumPy, SciPy..."
pip install numpy scipy scikit-learn

# Create directories
echo "📁 Creating directories..."
mkdir -p templates uploads output

# Download SAM models
echo "⬇️ Downloading SAM models..."
echo "This may take a few minutes..."

if [ ! -f "sam_vit_b_01ec64.pth" ]; then
    echo "Downloading SAM Base model (91MB)..."
    wget -q --show-progress https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
fi

if [ ! -f "sam_vit_l_0b3195.pth" ]; then
    echo "Downloading SAM Large model (308MB)..."
    wget -q --show-progress https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
fi

if [ ! -f "sam_vit_h_4b8939.pth" ]; then
    echo "Downloading SAM Huge model (636MB)..."
    wget -q --show-progress https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
fi

# Move index.html to templates directory
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
echo "🌐 Then open: http://localhost:8000"
echo ""
echo "💡 Features:"
echo "   • Upload image"
echo "   • Auto-generate points with Segformer"
echo "   • Add manual include/exclude points"
echo "   • Generate final mask with SAM"
echo "   • Download results"
echo ""
echo "🔧 Troubleshooting:"
echo "   • GPU recommended for better performance"
echo "   • Large images may need more RAM"
echo "   • Check health: http://localhost:8000/health"
