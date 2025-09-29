# Troubleshooting Guide

Common issues and solutions for AgriSegment Suite.

---

## 🔍 Quick Diagnosis

**Before troubleshooting, collect this information:**

```bash
# System info
python --version
pip --version
nvidia-smi  # If using GPU

# Check imports
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"

# Check server
curl http://localhost:8000  # Replace with your port
```

---

## 🚨 Installation Issues

### Python Version Error

**Symptom:**
```
ERROR: Python 3.8 or higher is required
```

**Solution:**
```bash
# Check current version
python --version

# Install Python 3.10 (Ubuntu)
sudo apt install python3.10 python3.10-venv

# Use specific version
python3.10 -m venv venv
source venv/bin/activate
```

---

### pip Installation Fails

**Symptom:**
```
ERROR: Could not install packages due to an EnvironmentError
```

**Solutions:**

**Option 1: Use virtual environment (Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

**Option 2: User installation**
```bash
pip install --user -r requirements.txt
```

**Option 3: Update pip**
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

### CUDA/GPU Not Detected

**Symptom:**
```python
torch.cuda.is_available() = False
```

**Diagnosis:**
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print(torch.version.cuda)"
```

**Solutions:**

**1. Install NVIDIA Drivers**
```bash
# Ubuntu
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
sudo reboot
```

**2. Install CUDA Toolkit**
- Download from [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)
- Install matching version with your driver

**3. Reinstall PyTorch with CUDA**
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**4. If GPU not needed, use CPU mode**
```python
# In server.py
device = "cpu"  # Force CPU mode
```

---

### Models Won't Download

**Symptom:**
```
HTTPError: Connection refused
OSError: Can't load model
```

**Solutions:**

**1. Check internet connection**
```bash
ping huggingface.co
```

**2. Use proxy if needed**
```bash
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
```

**3. Manual download**
```bash
# Download model manually
git lfs install
git clone https://huggingface.co/nvidia/segformer-b4-finetuned-ade-512-512

# Set path in code
model_path = "./segformer-b4-finetuned-ade-512-512"
```

**4. Change cache directory**
```bash
export TRANSFORMERS_CACHE=/path/to/writeable/dir
export HF_HOME=/path/to/writeable/dir
```

---

## 🖥️ Server Issues

### Port Already in Use

**Symptom:**
```
OSError: [Errno 48] Address already in use
```

**Solutions:**

**1. Find and kill process**
```bash
# Linux/macOS
lsof -ti:8000 | xargs kill -9

# Windows
netstat -ano | findstr :8000
taskkill /PID [PID_NUMBER] /F
```

**2. Change port**
```python
# In server.py
uvicorn.run(app, host="0.0.0.0", port=8888)  # Different port
```

---

### Server Won't Start

**Symptom:**
```
ModuleNotFoundError: No module named 'fastapi'
```

**Solution:**
```bash
# Ensure virtual environment activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt

# Verify installation
pip list | grep fastapi
```

---

### Can't Access Web Interface

**Symptom:**
- Browser shows "Can't reach this page"
- Connection refused

**Solutions:**

**1. Check if server running**
```bash
curl http://localhost:8000
# Should return HTML or JSON
```

**2. Check firewall**
```bash
# Ubuntu
sudo ufw allow 8000

# Windows
# Add firewall rule in Windows Security
```

**3. Use correct URL**
```
Local: http://localhost:8000
Network: http://[YOUR_IP]:8000
```

**4. Check host binding**
```python
# In server.py, ensure:
uvicorn.run(app, host="0.0.0.0", port=8000)  # Not "127.0.0.1"
```

---

## 💾 Memory Issues

### Out of Memory (GPU)

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

**1. Reduce batch size**
```python
# In server.py
BATCH_SIZE = 1  # Reduce from default
```

**2. Use smaller model**
```python
# Use base instead of large
# Use vit_b instead of vit_h
```

**3. Reduce image size**
```python
# Resize images before processing
max_size = 1024  # Reduce from 2048 or higher
```

**4. Clear CUDA cache**
```python
import torch
torch.cuda.empty_cache()
```

**5. Use CPU mode**
```python
device = "cpu"
```

---

### Out of Memory (RAM)

**Symptom:**
```
MemoryError: Unable to allocate array
```

**Solutions:**

**1. Process fewer images**
- Upload 5-10 images at a time instead of 50+

**2. Close other applications**
```bash
# Check memory usage
free -h  # Linux
# or
top
```

**3. Increase swap space** (Linux)
```bash
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## 🖼️ Image Processing Issues

### Upload Fails

**Symptom:**
```
413 Request Entity Too Large
```

**Solutions:**

**1. Reduce image size**
```bash
# Resize before upload
convert input.jpg -resize 2048x2048 output.jpg
```

**2. Increase upload limit**
```python
# In server.py
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB
```

---

### Poor Segmentation Quality

**Symptom:**
- Plants not detected
- Background included in mask
- Inaccurate boundaries

**Solutions:**

**1. Check image quality**
- Resolution: At least 512x512
- Lighting: Good, even lighting
- Focus: Sharp, not blurry
- Contrast: Clear plant vs background

**2. Try different tool**
- `semantic/`: Fast automatic → Try if `hybrid/` slow
- `hybrid/`: Add manual refinement → Try if `semantic/` inaccurate
- `interactive/`: Full manual → Try if automatic fails
- `panoptic/`: Different model → Try if others fail

**3. Adjust confidence threshold** (`panoptic/`)
```python
threshold = 0.3  # Lower for more detections
threshold = 0.7  # Higher for more confident
```

**4. Add more points** (`hybrid/` or `interactive/`)
- Add 5-10 include points on plant
- Add 2-5 exclude points on background

---

### Wrong Plant Classes Detected

**Symptom:**
- Trees detected as grass
- Background classified as plant

**Cause:**
- Model limitations
- Unusual plant types
- Poor image quality

**Solutions:**

**1. Use different tool**
- Each tool uses different models/datasets
- Try `panoptic/` if `semantic/` fails

**2. Improve image**
- Better lighting
- Higher resolution
- Clear plant visibility

**3. Use interactive refinement**
- Start with `semantic/` for speed
- Refine with `hybrid/` or `interactive/`

---

## ⚡ Performance Issues

### Very Slow Processing

**Symptom:**
- Takes 30+ seconds per image
- Much slower than benchmarks

**Solutions:**

**1. Enable GPU**
```bash
# Verify GPU used
python -c "import torch; print(torch.cuda.is_available())"

# Should print: True
```

**2. Use smaller model**
- Base instead of Large/Huge
- Faster but still good quality

**3. Reduce image resolution**
```python
# Resize to 1024x1024 or 512x512
```

**4. Close other applications**
- Free up GPU/CPU
- Close browser tabs
- Stop background processes

**5. Check CPU/GPU usage**
```bash
# Monitor resources
nvidia-smi  # GPU
htop        # CPU/RAM
```

---

### Browser Becomes Unresponsive

**Symptom:**
- Page freezes during processing
- Can't interact with interface

**Solution:**
- This is normal for large batches
- Processing happens server-side
- Wait for completion
- Or reduce batch size

---

## 🔧 Configuration Issues

### Settings Not Saving

**Symptom:**
- Changes reset after refresh
- Configuration not persisting

**Cause:**
- Sessions are temporary (by design)
- Server restart clears sessions

**Solution:**
```python
# Modify defaults in server.py
DEFAULT_MODEL = "vit_l"
DEFAULT_THRESHOLD = 0.5
BATCH_SIZE = 4
```

---

### Can't Change Port

**Symptom:**
- Port change doesn't work
- Still uses old port

**Solution:**

**1. Edit correct file**
```python
# In server.py (not config.py or other files)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=NEW_PORT)
```

**2. Restart server**
```bash
# Kill old server first
pkill -f "python server.py"

# Start with new config
python server.py
```

---

## 📦 Dependency Conflicts

### Version Conflicts

**Symptom:**
```
ERROR: Cannot install incompatible versions
```

**Solutions:**

**1. Create fresh environment**
```bash
# Remove old environment
rm -rf venv/

# Create new
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Update dependencies**
```bash
pip install --upgrade -r requirements.txt
```

**3. Install specific versions**
```bash
pip install torch==2.0.0 transformers==4.30.0
```

---

### ImportError After Update

**Symptom:**
```
ImportError: cannot import name 'X' from 'Y'
```

**Solution:**
```bash
# Reinstall all packages
pip uninstall -r requirements.txt -y
pip install -r requirements.txt

# Or reinstall specific package
pip uninstall transformers
pip install transformers
```

---

## 🐛 Application Errors

### "Session Not Found"

**Symptom:**
```
404: Session not found
```

**Cause:**
- Server restarted (sessions cleared)
- Session expired
- Browser cache issue

**Solutions:**

**1. Refresh page and reupload**
```
Ctrl+F5 (hard refresh)
```

**2. Clear browser cache**

**3. Increase session timeout**
```python
# In server.py
SESSION_TIMEOUT = 3600  # 1 hour instead of default
```

---

### Results Don't Download

**Symptom:**
- Click download but nothing happens
- ZIP file empty or corrupt

**Solutions:**

**1. Check browser downloads folder**

**2. Try different browser**
- Chrome, Firefox, Safari, Edge

**3. Right-click and "Save As"**

**4. Check server logs**
```bash
# Run server with verbose logging
python server.py --log-level debug
```

---

## 🔍 Debugging Tips

### Enable Debug Logging

```python
# In server.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Server Logs

```bash
# Run server in foreground to see logs
python server.py

# Or save logs to file
python server.py > server.log 2>&1
```

### Test API Directly

```bash
# Test endpoints with curl
curl -X POST http://localhost:8000/segment \
  -F "file=@test.jpg" \
  -o response.json

# Check response
cat response.json
```

### Verify Model Loading

```python
# Test in Python
from transformers import AutoModelForSemanticSegmentation

model = AutoModelForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b4-finetuned-ade-512-512"
)
print("Model loaded successfully!")
```

---

## 📞 Getting Further Help

### Information to Provide When Reporting Issues

1. **System information:**
```bash
uname -a  # OS
python --version
pip list  # All installed packages
nvidia-smi  # If using GPU
```

2. **Error messages:**
- Full error traceback
- Server logs
- Browser console errors (F12 → Console)

3. **Steps to reproduce:**
- Exact commands run
- Images used (if possible)
- Settings selected

4. **What you tried:**
- Solutions attempted
- Results of each attempt

### Where to Get Help

1. **Check documentation:**
   - [Installation Guide](installation.md)
   - Tool-specific READMEs
   - [Main README](../README.md)

2. **Search existing issues:**
   - [GitHub Issues](https://github.com/vahidshokrani415/AgriSegment/issues)
   - Someone may have had same problem

3. **Open new issue:**
   - [Create Issue](https://github.com/vahidshokrani415/AgriSegment/issues/new)
   - Use issue template
   - Provide all information above

4. **Contact developers:**
   - Email: mehran.tarifhokmabadi@univr.it
   - Include system info and error messages

---

## 💡 Prevention Tips

### Avoid Common Issues

**1. Use virtual environments**
```bash
# Always use venv
python -m venv venv
source venv/bin/activate
```

**2. Keep dependencies updated**
```bash
pip install --upgrade -r requirements.txt
```

**3. Monitor resources**
```bash
# Check before processing large batches
free -h        # RAM
df -h          # Disk space
nvidia-smi     # GPU
```

**4. Test with small batches first**
- Try 1-5 images before processing 100+
- Verify everything works
- Then scale up

**5. Keep backups**
- Save important configurations
- Export results frequently
- Don't rely on session storage

---

<div align="center">

**[← Back to Main README](../README.md)** | **[Installation Guide →](installation.md)**

</div>