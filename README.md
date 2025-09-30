[![DOI](https://zenodo.org/badge/1066343040.svg)](https://doi.org/10.5281/zenodo.17237438)

# 🌱 AgriSegment

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-ICROPM%202026-red.svg)](https://icropm2026.org)
[![Maintenance](https://img.shields.io/badge/Maintained-Yes-brightgreen.svg)](https://github.com/mehran-tarif/AgriSegment)

**Multi-modal Plant Segmentation Suite for Agricultural Research**

A web-based platform that makes AI-powered plant segmentation accessible to agronomists and researchers without programming expertise.

---

## 📦 Suite Components

AgriSegment provides **four specialized tools** for different segmentation workflows:

### 🎨 `hybrid/` - Port 8000
**Hybrid SegFormer + SAM Workflow**

- **Best for:** Research projects requiring highest accuracy
- **Method:** Automatic detection with interactive refinement
- **Key feature:** Feedback loop where corrections improve AI performance
- **Output:** High-quality masks with editable point annotations

### ✏️ `interactive/` - Port 8001
**Interactive SAM Segmentation**

- **Best for:** Quick single-image tasks
- **Method:** Point-based interactive segmentation
- **Key feature:** Real-time mask generation from user clicks
- **Output:** Binary masks, overlays, transparent PNGs

### ⚡ `semantic/` - Port 8002
**Automated SegFormer Processing**

- **Best for:** Large-scale batch processing
- **Method:** Fully automatic plant detection
- **Key feature:** Multi-image upload with ZIP download
- **Output:** 2x3 visualization grid with statistics

### 🔬 `panoptic/` - Port 8003
**Advanced Mask2Former Analysis**

- **Best for:** Detailed multi-modal segmentation analysis
- **Method:** Semantic, instance, and panoptic segmentation modes
- **Key feature:** Comprehensive statistics and confidence thresholds
- **Output:** Labeled segments with coverage percentages

---

## 🎯 Quick Selection Guide

| Your Need | Recommended Tool | Why |
|-----------|------------------|-----|
| Highest accuracy for research | **hybrid/** | Manual refinement + feedback loop |
| Quick single-image segmentation | **interactive/** | Fast, interactive, no setup |
| Process 100+ images | **semantic/** | Automated bulk processing |
| Instance-level detection | **panoptic/** | Multiple segmentation modes |
| Learning/experimenting | **interactive/** → **hybrid/** | Start simple, then advanced |
| Production pipeline | **semantic/** → **hybrid/** | Auto-process, refine when needed |

---

## 🚀 Quick Start

### Method 1: Run Individual Tool

```bash
# Navigate to desired tool
cd hybrid/        # Port 8000
cd interactive/   # Port 8001
cd semantic/      # Port 8002
cd panoptic/      # Port 8003

# Install and run
bash installer.sh
python server.py
```

**Access:** `http://localhost:PORT`
- hybrid/: 8000
- interactive/: 8001
- semantic/: 8002
- panoptic/: 8003

### Method 2: Docker (All Tools)

```bash
# Start all four tools simultaneously
docker-compose up

# Or run in detached mode
docker-compose up -d
```

**Access all tools:**
- hybrid/: http://localhost:8000
- interactive/: http://localhost:8001
- semantic/: http://localhost:8002
- panoptic/: http://localhost:8003

---

## 💡 The Hybrid Workflow Innovation (`hybrid/`)

AgriSegment `hybrid/` implements a continuous improvement cycle:

```
┌─────────────────────────────────────────────┐
│  1. SegFormer detects plant regions         │
│            ↓                                 │
│  2. User refines with SAM (click points)    │
│            ↓                                 │
│  3. Refined masks = better training data    │
│            ↓                                 │
│  4. Retrain SegFormer with improved data    │
│            ↓                                 │
│  5. Better automatic detection              │
│            ↓                                 │
│        (cycle repeats)                       │
└─────────────────────────────────────────────┘
```

This feedback loop enables continuous dataset improvement and model refinement.

---

## 🛠️ System Requirements

### Minimum
- **OS:** Linux, macOS, Windows 10+
- **Python:** 3.8 or higher
- **RAM:** 8GB
- **Storage:** 10GB free space
- **Internet:** Required for model downloads

### Recommended
- **RAM:** 16GB or more
- **GPU:** CUDA-capable (NVIDIA) for faster processing
- **Storage:** 20GB SSD

---

## 📥 Installation

### Prerequisites

```bash
# Verify Python version
python --version  # Should be 3.8+

# Verify pip
pip --version
```

### Install Single Tool

```bash
# Clone repository
git clone https://github.com/mehran-tarif/AgriSegment.git
cd AgriSegment

# Install specific tool
cd [tool-folder]  # hybrid, interactive, panoptic, or semantic
bash installer.sh
```

### Install All Tools

```bash
# Navigate to each folder and install
cd hybrid && bash installer.sh && cd ..
cd interactive && bash installer.sh && cd ..
cd panoptic && bash installer.sh && cd ..
cd semantic && bash installer.sh && cd ..
```

### First Run
Models download automatically on first startup (5-10 minutes depending on internet speed).

---

## 📖 Usage

### 1. Upload Images
- **Drag & drop** files into the upload area
- Or click **"Upload"** button to browse
- **Supported formats:** JPG, PNG, JPEG, BMP
- **Max size:** 50MB per image (recommended: under 10MB)

### 2. Configure Settings
- **hybrid/:** Adjust point generation parameters
- **panoptic/:** Select segmentation mode (semantic/instance/panoptic)
- **interactive/:** Choose model size (base/large/huge)
- **semantic/:** Set number of concurrent processes

### 3. Process Images
- Click **"Generate"** or **"Segment"** button
- Wait for processing (1-30 seconds depending on image size and tool)
- View results in real-time

### 4. Download Results
- **Individual files:** Click download icon on each result
- **Batch download:** Click "Download All as ZIP"
- **Available formats:**
  - Binary masks (PNG)
  - Colored overlays (PNG)
  - Transparent backgrounds (PNG)
  - Statistics (JSON/CSV)

---

## 📊 Output Formats

### All Tools Provide
- ✅ Original image
- ✅ Segmentation mask
- ✅ Overlay visualization
- ✅ Downloadable files

### `hybrid/` & `panoptic/` Include
- ✅ Confidence scores
- ✅ Area measurements (pixels & percentage)
- ✅ Class labels
- ✅ Point coordinates (`hybrid/`)

### File Structure
```
results/
├── session_[id]/
│   ├── original_image.jpg
│   ├── mask.png
│   ├── overlay.png
│   ├── transparent.png
│   └── statistics.json
```

---

## 🔧 Configuration

### Change Port

Edit `server.py` in any tool folder:

```python
if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000  # Change this number
    )
```

### Adjust Memory Usage

For low-memory systems, reduce batch size:

```python
# In server.py
BATCH_SIZE = 1  # Default: 4
```

### Enable GPU

GPU automatically detected if CUDA available. To force CPU mode:

```python
# In server.py
device = "cpu"  # Default: "cuda" if available else "cpu"
```

---

## 🐛 Troubleshooting

### Port Already in Use

```bash
# Find and kill process using port
lsof -ti:8000 | xargs kill -9

# Or change port in server.py
```

### Out of Memory Error

**Solutions:**
- Reduce image size (resize to 1024x1024 or smaller)
- Close other applications
- Set `BATCH_SIZE=1` in configuration
- Use CPU mode instead of GPU

### Models Not Downloading

**Causes:**
- Internet connection interrupted
- Firewall blocking Hugging Face
- Insufficient disk space

**Solutions:**
```bash
# Manually download models
python -c "from transformers import AutoModel; AutoModel.from_pretrained('nvidia/segformer-b4-finetuned-ade-512-512')"

# Check available disk space
df -h
```

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Or use installer
bash installer.sh
```

---

## 📚 Documentation

Detailed guides for each tool:

- [📖 hybrid/ Documentation](hybrid/README.md)
- [📖 interactive/ Documentation](interactive/README.md)
- [📖 semantic/ Documentation](semantic/README.md)
- [📖 panoptic/ Documentation](panoptic/README.md)

Additional resources:
- [Quick Start Guide](docs/quickstart.md)
- [Installation Guide](docs/installation.md)
- [Best Practices](docs/best-practices.md)
- [Troubleshooting Guide](docs/troubleshooting.md)
- [API Reference](docs/api.md)

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Report Bugs
Open an issue with:
- Tool name and version
- Steps to reproduce
- Expected vs actual behavior
- Screenshots if applicable

### Suggest Features
Open an issue with:
- Clear feature description
- Use case explanation
- Potential implementation approach

### Submit Pull Requests
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📄 Citation

*(coming soon)*

```bibtex
# Citation will be available after ICROPM 2026 publication
```

---

## 👥 Authors

**Tarif Mehran**  
Department of Computer Science  
University of Verona, Italy  
📧 mehran.tarifhokmabadi@univr.it

**Davide Quaglia**  
Department of Computer Science  
University of Verona, Italy

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

**Funding:**

This work was supported by:

- **Italian Space Agency (ASI)** - Project: "An Open, Efficient, and Customizable Pipeline for the Automated Processing of Remote Sensed Data for Computational Agro-Ecology"

- **Regione del Veneto** - PR FESR 2021-2027, Action 1.1.1 - Project: "AGRIFUTURE: Il Futuro della Sostenibilità per le Sfide Competitive delle Aziende Agroalimentari Venete" (Project ID: 24279_001587_04499230235)

The funders had no role in study design, data collection and analysis, decision to publish, or preparation of this work.

**Institutional Support:**  
University of Verona, Department of Computer Science

**Built with open-source technologies:**
- [SegFormer](https://github.com/NVlabs/SegFormer) by NVIDIA
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) by Meta AI
- [Mask2Former](https://github.com/facebookresearch/Mask2Former) by Meta AI
- [FastAPI](https://fastapi.tiangolo.com/) web framework
- [PyTorch](https://pytorch.org/) deep learning framework
- [Transformers](https://huggingface.co/transformers/) by Hugging Face

---

## 📞 Support

**Need help?**
- 📖 Check [Documentation](docs/)
- 🐛 Report [Issues](https://github.com/mehran-tarif/AgriSegment/issues)
- 💬 Join [Discussions](https://github.com/mehran-tarif/AgriSegment/discussions)
- 📧 Email: mehran.tarifhokmabadi@univr.it

---

## ⭐ Quick Tips

- **First time?** Start with **interactive/** - it's the simplest
- **Need accuracy?** Use **hybrid/** with manual refinement points
- **Processing many images?** Use **semantic/** for automated workflows
- **Research paper?** Use **panoptic/** for detailed statistics
- **Production deployment?** Combine **semantic/** + **hybrid/** pipeline

---

## 🔗 Links

- Project Homepage *(coming soon)*
- [GitHub Repository](https://github.com/mehran-tarif/AgriSegment)
- [ICROPM 2026 Paper](https://icropm2026.org)
- [University of Verona](https://www.univr.it)

---

<div align="center">

**Made with 🌱 for agricultural research**

[⭐ Star on GitHub](https://github.com/mehran-tarif/AgriSegment) | [🐛 Report Bug](https://github.com/mehran-tarif/AgriSegment/issues) | [✨ Request Feature](https://github.com/mehran-tarif/AgriSegment/issues)

</div>
