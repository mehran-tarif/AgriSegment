# 🎨 hybrid/ - Hybrid SegFormer + SAM Workflow

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Port](https://img.shields.io/badge/Port-8000-green.svg)](http://localhost:8000)
[![Model](https://img.shields.io/badge/Models-SegFormer%20%2B%20SAM-orange.svg)](https://huggingface.co)

**Automatic detection with interactive refinement and feedback loop**

Part of the [AgriSegment Suite](../README.md) - The most advanced tool for high-accuracy plant segmentation.

---

## 🎯 Overview

The `hybrid/` tool combines two powerful AI models in a unique workflow:

1. **SegFormer** automatically detects plant regions and generates initial points
2. **SAM** allows you to refine segmentation by adding/removing points interactively
3. **Feedback Loop**: Your corrections create better training data for improving SegFormer

This creates a continuous improvement cycle for dataset preparation and model refinement.

---

## ✨ Key Features

- ✅ **Automatic point generation** from SegFormer segmentation
- ✅ **Interactive refinement** with include/exclude points
- ✅ **Batch processing** for multiple images
- ✅ **Session management** to track your work
- ✅ **Multiple output formats** (binary masks, overlays, transparent PNGs)
- ✅ **Flexible workflow** - skip automatic detection if you prefer manual
- ✅ **SAM model selection** (Base/Large/Huge - 91M/308M/636M parameters)
- ✅ **Downloadable results** as individual files or ZIP archive

---

## 🚀 Quick Start

### Installation

```bash
cd hybrid/
bash installer.sh
```

This will install all required dependencies including:
- PyTorch with CUDA support (if available)
- Transformers (Hugging Face)
- SAM model
- FastAPI and Uvicorn
- All other dependencies

### Run Server

```bash
python server.py
```

**Access:** http://localhost:8000

Server will start on port 8000. Models download automatically on first run (5-10 minutes).

---

## 📖 How to Use

### Step 1: Upload Images

1. Navigate to http://localhost:8000
2. Click **"Upload Images"** or drag and drop files
3. Supported formats: JPG, PNG, JPEG, BMP
4. Can upload multiple images at once

### Step 2: Generate Points (Optional)

**Option A: Automatic Point Generation**
1. Click **"Generate Points"** button
2. SegFormer analyzes images and finds plant regions
3. Points appear automatically on detected areas
4. Wait for processing (3-10 seconds per image)

**Option B: Skip to Manual**
1. Skip point generation if you prefer full manual control
2. Proceed directly to adding your own points

### Step 3: Edit Points Interactively

**Add Include Points:**
- Click green **"Include Point"** button
- Click on plant regions you want to segment
- Each click adds a green point

**Add Exclude Points:**
- Click red **"Exclude Point"** button  
- Click on background regions to remove from mask
- Each click adds a red point

**Remove Points:**
- Click **"Remove Last Point"** to undo
- Or clear all and start over

### Step 4: Generate Masks

1. Click **"Generate Masks"** button
2. Select SAM model size:
   - **Base** (91M parameters) - Faster
   - **Large** (308M parameters) - Balanced
   - **Huge** (636M parameters) - Most accurate
3. Wait for processing (5-30 seconds depending on model)
4. View results in the interface

### Step 5: Download Results

**Individual Downloads:**
- Click download icon on each result image
- Available formats: Binary mask, Overlay, Transparent PNG

**Batch Download:**
- Click **"Download All as ZIP"**
- All results packaged together
- Organized by session ID

---

## 🔄 The Feedback Loop

This tool's unique innovation is the **continuous improvement cycle**:

```
┌─────────────────────────────────────────────┐
│  1. SegFormer detects plant regions         │
│            ↓                                 │
│  2. Points generated on detected areas      │
│            ↓                                 │
│  3. You refine with SAM (add/remove points) │
│            ↓                                 │
│  4. Refined masks = high-quality labels     │
│            ↓                                 │
│  5. Use these labels to retrain SegFormer   │
│            ↓                                 │
│  6. Improved SegFormer detects better       │
│            ↓                                 │
│        (cycle repeats)                       │
└─────────────────────────────────────────────┘
```

**How to leverage this:**
1. Use `hybrid/` to create high-quality segmentation masks
2. Export refined masks as training data
3. Fine-tune SegFormer on your corrected dataset
4. Deploy improved SegFormer for better automatic detection
5. Repeat the cycle for continuous improvement

---

## 🛠️ Configuration

### Change Port

Edit `server.py`:

```python
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000  # Change this
    )
```

### Select SAM Model

Models are loaded on-demand. You can preload a specific size by editing `server.py`:

```python
# Preload specific model
SAM_MODEL_TYPE = "vit_h"  # vit_b (base), vit_l (large), vit_h (huge)
```

### Adjust Memory Usage

For low-memory systems:

```python
# In server.py
BATCH_SIZE = 1  # Reduce from default 4
MAX_POINTS = 50  # Reduce from default 100
```

---

## 📊 Output Structure

Results are saved in `results/session_[id]/`:

```
results/
└── session_abc123/
    ├── image1/
    │   ├── original.jpg
    │   ├── points.json
    │   ├── mask.png
    │   ├── overlay.png
    │   └── transparent.png
    ├── image2/
    │   └── ...
    └── all_results.zip
```

**File descriptions:**
- `original.jpg` - Your uploaded image
- `points.json` - Coordinates of include/exclude points
- `mask.png` - Binary segmentation mask (black/white)
- `overlay.png` - Mask overlaid on original image
- `transparent.png` - Plant with transparent background
- `all_results.zip` - All files packaged for download

---

## 🎨 Workflow Strategies

### Strategy 1: Maximum Accuracy (Research)
1. Upload single high-quality image
2. Generate automatic points with SegFormer
3. Carefully refine with 5-15 additional SAM points
4. Use **Huge** model for final mask
5. Export for ground truth annotations

### Strategy 2: Balanced Speed + Quality
1. Upload 5-10 images as batch
2. Generate automatic points
3. Quick refinement with 2-5 points per image
4. Use **Large** model
5. Suitable for dataset preparation

### Strategy 3: Rapid Prototyping
1. Upload images
2. Skip automatic points
3. Add only 1-3 key points manually
4. Use **Base** model
5. Quick iteration for testing

---

## 🔧 Troubleshooting

### Points Not Generating

**Problem:** "Generate Points" doesn't produce results

**Solutions:**
- Ensure image contains visible plants/vegetation
- Try higher resolution images (min 512x512)
- Check server logs for errors
- SegFormer may not detect very small or unusual plants

### SAM Mask Quality Poor

**Problem:** Masks don't capture plant accurately

**Solutions:**
- Add more include points on plant edges
- Add exclude points on background near plant
- Try different SAM model size (Huge = best quality)
- Ensure good image quality and lighting

### Out of Memory

**Problem:** Server crashes during processing

**Solutions:**
- Reduce image size (resize to 1024x1024)
- Use smaller SAM model (Base instead of Huge)
- Process fewer images at once
- Set `BATCH_SIZE=1` in configuration
- Close other applications

### Slow Processing

**Problem:** Takes too long to generate masks

**Solutions:**
- Use smaller SAM model (Base = fastest)
- Enable GPU acceleration (CUDA)
- Reduce image resolution
- Process images one at a time

---

## 💡 Tips & Best Practices

### Point Placement
- **Include points**: Place in center of plant regions
- **Exclude points**: Place on clear background areas
- **Edges**: Add points near boundaries for better edge detection
- **Quantity**: Start with 3-5 points, add more if needed

### Image Preparation
- Use good lighting (avoid shadows)
- Higher resolution = better results (but slower)
- Clear background helps automatic detection
- Avoid motion blur or out-of-focus images

### Batch Processing
- Keep batches under 20 images for manageable refinement
- Process similar images together (same plant type/conditions)
- Save frequently (results persist in session)

### Model Selection
- **Base (vit_b)**: Quick tests, many images, good enough quality
- **Large (vit_l)**: Best balance for most use cases
- **Huge (vit_h)**: Research, publications, ground truth creation

---

## 📊 Performance Benchmarks

Tested on NVIDIA RTX 3090 (24GB VRAM):

| Task | Base Model | Large Model | Huge Model |
|------|------------|-------------|------------|
| Point generation (SegFormer) | 2-3s | 2-3s | 2-3s |
| Single mask (1024x1024) | 1-2s | 3-5s | 8-12s |
| Batch 10 images | 15-20s | 35-50s | 90-120s |
| Memory usage | ~4GB | ~8GB | ~12GB |

*CPU-only processing is 5-10x slower*

---

## 🔗 Related Tools

Part of **AgriSegment Suite**:

- [`interactive/`](../interactive/README.md) - Pure SAM without SegFormer (faster for single images)
- [`semantic/`](../semantic/README.md) - Pure SegFormer batch processing (faster automatic)
- [`panoptic/`](../panoptic/README.md) - Mask2Former advanced analysis

**When to use `hybrid/` instead:**
- You need both automatic detection AND refinement capability
- Creating high-quality training datasets
- Research requiring maximum accuracy
- Implementing feedback loop for model improvement

---

## 📞 Support

**Issues?**
- Check [main README troubleshooting](../README.md#troubleshooting)
- Open issue on [GitHub](https://github.com/vahidshokrani415/AgriSegment/issues)
- Email: mehran.tarifhokmabadi@univr.it

**Documentation:**
- [Main README](../README.md)
- [API Reference](../docs/api.md) *(coming soon)*

---

## 📄 Technical Details

**Models Used:**
- **SegFormer**: `nvidia/segformer-b4-finetuned-ade-512-512`
- **SAM**: `facebook/sam-vit-base`, `sam-vit-large`, `sam-vit-huge`

**Dependencies:**
- PyTorch >= 1.13.0
- Transformers >= 4.30.0
- FastAPI >= 0.104.0
- Pillow >= 9.5.0
- NumPy >= 1.24.0

**Server:**
- Framework: FastAPI + Uvicorn
- Port: 8000
- Sessions: In-memory (cleared on restart)
- Max upload: 50MB per image

---

<div align="center">

**Part of AgriSegment Suite 🌱**

[Main README](../README.md) | [Report Bug](https://github.com/vahidshokrani415/AgriSegment/issues) | [Request Feature](https://github.com/vahidshokrani415/AgriSegment/issues)

</div>