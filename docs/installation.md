# Installation Guide

Complete installation instructions for AgriSegment Suite.

---

## 📋 Prerequisites

### System Requirements

**Minimum:**
- OS: Linux, macOS, or Windows 10+
- Python: 3.8 or higher
- RAM: 8GB
- Storage: 10GB free space
- Internet: Required for model downloads

**Recommended:**
- OS: Ubuntu 20.04+ or similar Linux
- Python: 3.9 or 3.10
- RAM: 16GB or more
- Storage: 20GB SSD
- GPU: NVIDIA with CUDA support (8GB+ VRAM)

### Software Dependencies

#### 1. Python Installation

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
```

**macOS:**
```bash
brew install python@3.10
```

**Windows:**
Download from [python.org](https://www.python.org/downloads/)

Verify installation:
```bash
python --version  # Should show 3.8+
pip --version
```

#### 2. CUDA (Optional, for GPU acceleration)

**Check if CUDA available:**
```bash
nvidia-smi
```

**Install CUDA Toolkit:**
- Download from [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)
- Recommended: CUDA 11.8 or 12.1
- Follow NVIDIA's official installation guide

---

## 🚀 Installation Methods

### Method 1: Individual Tool Installation (Recommended)

Install only the tool(s) you need.

#### Step 1: Clone Repository
```bash
git clone https://github.com/vahidshokrani415/AgriSegment.git
cd AgriSegment
```

#### Step 2: Choose Tool and Install

**For hybrid/ (Port 8000):**
```bash
cd hybrid/
bash installer.sh
```

**For interactive/ (Port 8001):**
```bash
cd interactive/
bash installer.sh
```

**For semantic/ (Port 8002):**
```bash
cd semantic/
bash installer.sh
```

**For panoptic/ (Port 8003):**
```bash
cd panoptic/
bash installer.sh
```

#### Step 3: Run Server
```bash
python server.py
```

---

### Method 2: Install All Tools

Install all four tools at once.

```bash
# Clone repository
git clone https://github.com/vahidshokrani415/AgriSegment.git
cd AgriSegment

# Install each tool
cd hybrid && bash installer.sh && cd ..
cd interactive && bash installer.sh && cd ..
cd semantic && bash installer.sh && cd ..
cd panoptic && bash installer.sh && cd ..
```

---

### Method 3: Docker Installation (Coming Soon)

Docker support planned for future release.

```bash
# Clone repository
git clone https://github.com/vahidshokrani415/AgriSegment.git
cd AgriSegment

# Build and run
docker-compose up
```

---

## 📦 Manual Installation

If automatic installer fails, follow these manual steps:

### Step 1: Create Virtual Environment

```bash
cd [tool-folder]  # e.g., hybrid/

# Create virtual environment
python -m venv venv

# Activate it
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### Step 2: Install PyTorch

**With CUDA (GPU):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CPU Only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Verify PyTorch:
```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Test Installation

```bash
python server.py
```

If server starts without errors, installation successful!

---

## 🔧 Common Installation Issues

### Issue 1: Python Version Too Old

**Error:** "Python 3.8 or higher required"

**Solution:**
```bash
# Ubuntu
sudo apt install python3.10

# Or use pyenv
pyenv install 3.10.0
pyenv global 3.10.0
```

### Issue 2: pip Not Found

**Error:** "pip: command not found"

**Solution:**
```bash
# Ubuntu/Debian
sudo apt install python3-pip

# macOS
python3 -m ensurepip

# Verify
pip --version
```

### Issue 3: CUDA Not Detected

**Error:** "torch.cuda.is_available() = False"

**Check:**
```bash
# Verify NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version
```

**Solution:**
- Install/update NVIDIA drivers
- Install matching CUDA toolkit
- Reinstall PyTorch with correct CUDA version

### Issue 4: Permission Denied

**Error:** "Permission denied" during installation

**Solution:**
```bash
# Don't use sudo with pip
# Instead use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Or use --user flag
pip install --user -r requirements.txt
```

### Issue 5: Out of Disk Space

**Error:** "No space left on device"

**Check space:**
```bash
df -h
```

**Solution:**
- Clear pip cache: `pip cache purge`
- Remove unused files
- Models require ~5-10GB, ensure sufficient space

### Issue 6: Requirements Installation Fails

**Error:** "ERROR: Could not find a version that satisfies..."

**Solution:**
```bash
# Update pip first
pip install --upgrade pip setuptools wheel

# Then retry
pip install -r requirements.txt

# If still fails, install one by one
pip install torch
pip install transformers
pip install fastapi
# etc.
```

---

## 🎯 Post-Installation Setup

### 1. Verify Installation

Test each tool:

```bash
# Test hybrid/
cd hybrid/
python -c "from transformers import AutoModelForSemanticSegmentation; print('SegFormer OK')"
python -c "import torch; print('PyTorch OK')"

# Test server
python server.py &
curl http://localhost:8000
kill %1
```

### 2. Download Models

Models download automatically on first run. To pre-download:

```bash
# In Python
from transformers import AutoModelForSemanticSegmentation

# SegFormer
model = AutoModelForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b4-finetuned-ade-512-512"
)

# Mask2Former
model = AutoModelForSemanticSegmentation.from_pretrained(
    "facebook/mask2former-swin-base-ade-semantic"
)
```

### 3. Configure Ports

If default ports are in use, change them:

**Edit server.py:**
```python
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=XXXX)  # Change port
```

### 4. Set Up Environment Variables

Create `.env` file (optional):

```bash
# .env file
CUDA_VISIBLE_DEVICES=0  # Which GPU to use
MAX_UPLOAD_SIZE=52428800  # 50MB in bytes
MODEL_CACHE_DIR=/path/to/models  # Custom model location
```

---

## 🚀 Performance Optimization

### GPU Acceleration

**Enable CUDA:**
```bash
# Verify GPU available
python -c "import torch; print('GPU Count:', torch.cuda.device_count())"

# If 0, check CUDA installation
```

**Select Specific GPU:**
```bash
# In terminal before running
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0

# Or in Python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

### Memory Optimization

**For 8GB GPU:**
```python
# In server.py
BATCH_SIZE = 1  # Reduce batch size
USE_HALF_PRECISION = True  # Use FP16
```

**For CPU-only:**
```python
# Force CPU mode
device = "cpu"
```

### Model Caching

**Set custom cache directory:**
```bash
export TRANSFORMERS_CACHE=/path/to/large/disk/models
export HF_HOME=/path/to/large/disk
```

**Pre-download all models:**
```python
# Run once to download all
from transformers import AutoModelForSemanticSegmentation

models = [
    "nvidia/segformer-b4-finetuned-ade-512-512",
    "facebook/mask2former-swin-base-ade-semantic",
    "facebook/mask2former-swin-large-ade-semantic",
    # Add others...
]

for model_name in models:
    print(f"Downloading {model_name}...")
    AutoModelForSemanticSegmentation.from_pretrained(model_name)
```

---

## 🐳 Docker Installation (Alternative)

**Coming soon in future release.**

Planned Docker support will include:
```bash
# Single tool
docker run -p 8000:8000 agrisegment/hybrid

# All tools
docker-compose up

# With GPU
docker run --gpus all -p 8000:8000 agrisegment/hybrid
```

---

## 📝 Next Steps

After successful installation:

1. **Test each tool** - Visit http://localhost:PORT
2. **Read tool-specific docs** - Check individual README files
3. **Try example images** - Process test images
4. **Configure settings** - Adjust ports, models, etc.
5. **Join community** - Report issues, ask questions

---

## 📞 Getting Help

**Installation problems?**

1. Check [Troubleshooting Guide](troubleshooting.md)
2. Search [GitHub Issues](https://github.com/vahidshokrani415/AgriSegment/issues)
3. Open new issue with details:
   - Operating system
   - Python version
   - Error messages
   - Installation method used

**Contact:**
- Email: mehran.tarifhokmabadi@univr.it
- GitHub: [Open an issue](https://github.com/vahidshokrani415/AgriSegment/issues/new)

---

## 🔄 Updating

To update to latest version:

```bash
cd AgriSegment
git pull origin main

# Reinstall dependencies in each tool
cd hybrid/
pip install -r requirements.txt --upgrade
cd ../interactive/
pip install -r requirements.txt --upgrade
# etc.
```

---

<div align="center">

**[← Back to Main README](../README.md)** | **[Troubleshooting →](troubleshooting.md)**

</div>