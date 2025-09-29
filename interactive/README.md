# ✏️ interactive/ - Interactive SAM Segmentation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Port](https://img.shields.io/badge/Port-8001-green.svg)](http://localhost:8001)
[![Model](https://img.shields.io/badge/Model-SAM-orange.svg)](https://segment-anything.com/)

**Fast, point-based interactive segmentation for quick tasks**

Part of the [AgriSegment Suite](../README.md) - The simplest tool for single-image segmentation.

---

## 🎯 Overview

The `interactive/` tool provides pure SAM (Segment Anything Model) segmentation with an intuitive point-and-click interface. No automatic detection - just you and the AI working together to create perfect masks.

**Perfect for:**
- Quick single-image segmentation
- Learning how SAM works
- Cases where you know exactly what you want to segment
- When you don't need batch processing

---

## ✨ Key Features

- ✅ **Click-based interface** - Add include/exclude points with mouse
- ✅ **Real-time preview** - See points as you add them
- ✅ **Manual coordinates** - Input exact pixel positions if needed
- ✅ **Three output formats** - Binary mask, masked image, transparent PNG
- ✅ **Model size selection** - Choose Base/Large/Huge (91M/308M/636M parameters)
- ✅ **Instant results** - Fast processing (1-15 seconds)
- ✅ **Simple workflow** - Upload → Click → Download
- ✅ **ZIP package** - Download all results at once

---

## 🚀 Quick Start

### Installation

```bash
cd interactive/
bash installer.sh
```

This installs SAM and all dependencies.

### Run Server

```bash
python server.py
```

**Access:** http://localhost:8001

Models download automatically on first run (~5 minutes).

---

## 📖 How to Use

### Basic Workflow

#### Step 1: Upload Image
1. Go to http://localhost:8001
2. Click **"Choose File"** or drag and drop
3. Supported formats: JPG, PNG, JPEG, BMP
4. Image appears in preview area

#### Step 2: Add Points
**Include Points (Green):**
- Click **"Include Point Mode"** button
- Click on areas you want to segment
- Green circles appear where you click

**Exclude Points (Red):**
- Click **"Exclude Point Mode"** button
- Click on areas to remove from mask
- Red circles appear where you click

**Tips:**
- Start with 1-3 include points in center of object
- Add exclude points near edges if needed
- More points = more precise segmentation

#### Step 3: Generate Mask
1. Select **SAM model size**:
   - **Base** (vit_b) - Fast, good quality
   - **Large** (vit_l) - Balanced
   - **Huge** (vit_h) - Best quality, slower
2. Click **"Generate Mask"** button
3. Wait 1-15 seconds (depends on model size)
4. Results appear below

#### Step 4: Download Results
- **Individual files**: Click each image to save
- **ZIP package**: Click "Download All as ZIP"

**Output includes:**
- Binary mask (black and white)
- Masked image (plant only)
- Transparent PNG (for overlays)

---

## 🎨 Output Formats

### Binary Mask
- Pure black and white PNG
- White = segmented region
- Black = background
- Best for: Training data, analysis

### Masked Image
- Original colors preserved
- Background = black
- Best for: Visualization, presentations

### Transparent PNG
- Segmented region with original colors
- Background = transparent
- Best for: Overlays, graphic design

---

## 🛠️ Configuration

### Change Port

Edit `server.py`:

```python
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001  # Change this
    )
```

### Default Model

Set preferred model in `server.py`:

```python
DEFAULT_MODEL = "vit_l"  # vit_b (base), vit_l (large), vit_h (huge)
```

### Max Upload Size

```python
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB default
```

---

## 💡 Tips & Best Practices

### Point Placement Strategy

**For simple objects (single plant):**
1. Add 1 include point in center
2. Check result
3. Add 1-2 more points if boundaries unclear

**For complex objects (multiple plants, overlapping):**
1. Add include points on each separate plant
2. Add exclude points between plants
3. Add points near edges for precise boundaries

**For challenging backgrounds:**
1. Start with include points on obvious plant parts
2. Add exclude points on similar-colored background
3. Refine boundaries with additional points

### Model Selection Guide

| Scenario | Recommended Model | Why |
|----------|------------------|-----|
| Quick test | **Base** | Fast, good enough |
| General use | **Large** | Best balance |
| Publication/research | **Huge** | Maximum accuracy |
| Many iterations | **Base** | Speed for experimentation |
| Final result | **Huge** | Best quality |

### Image Preparation

**Good images:**
- Clear subject
- Good lighting
- High resolution (at least 512x512)
- Good contrast plant vs background

**Avoid:**
- Motion blur
- Very low resolution
- Extreme lighting (over/underexposed)
- Heavily compressed JPEGs

---

## 🔧 Troubleshooting

### Mask Not Accurate

**Problem:** Segmentation includes unwanted areas

**Solutions:**
- Add exclude points on unwanted regions
- Add more include points to define boundaries
- Try larger model (Huge has best accuracy)
- Check if image quality is good enough

### "Too Many Points" Error

**Problem:** Added too many points (>100)

**Solutions:**
- Remove some points and retry
- Start fresh with fewer, more strategic points
- SAM works well with just 3-10 points

### Slow Generation

**Problem:** Takes too long to generate mask

**Solutions:**
- Use smaller model (Base instead of Huge)
- Reduce image size (resize before upload)
- Enable GPU if available
- Close other applications

### Out of Memory

**Problem:** Server crashes or "CUDA out of memory"

**Solutions:**
- Use Base model (requires less memory)
- Reduce image resolution
- Process one image at a time
- Restart server to clear memory

---

## 📊 Performance Benchmarks

Tested on NVIDIA RTX 3090 (24GB VRAM), 1024x1024 images:

| Model | Generation Time | Memory Usage | Quality |
|-------|----------------|--------------|---------|
| Base (vit_b) | 1-2s | ~4GB | Good |
| Large (vit_l) | 3-5s | ~8GB | Better |
| Huge (vit_h) | 8-12s | ~12GB | Best |

*CPU-only is 5-10x slower but works fine for occasional use*

---

## 🎯 Use Cases

### Academic Research
- Annotate individual plants for publications
- Create ground truth for validation
- Phenotyping measurements
- Quick exploratory analysis

### Agriculture
- Measure single plant area
- Identify disease spots
- Count leaves/fruits
- Quick field assessments

### Dataset Creation
- Label training data one image at a time
- Quality control for automatic annotations
- Correct mistakes from other tools
- Create validation sets

### Education
- Teach students about segmentation
- Demonstrate AI capabilities
- Hands-on ML learning
- Simple enough for non-programmers

---

## 🔗 Related Tools

Part of **AgriSegment Suite**:

- [`hybrid/`](../hybrid/README.md) - Combines automatic SegFormer + SAM refinement
- [`semantic/`](../semantic/README.md) - Automatic batch processing with SegFormer
- [`panoptic/`](../panoptic/README.md) - Advanced Mask2Former analysis

**When to use `interactive/` instead:**
- Processing single images only
- You want manual control over every point
- Learning/experimenting with SAM
- Fastest setup and simplest interface
- Don't need automatic detection

**When to use other tools:**
- [`hybrid/`] if you want automatic point generation first
- [`semantic/`] if processing many images without refinement
- [`panoptic/`] if you need instance-level detection

---

## 📞 Support

**Issues?**
- Check [main README troubleshooting](../README.md#troubleshooting)
- Open issue on [GitHub](https://github.com/vahidshokrani415/AgriSegment/issues)
- Email: mehran.tarifhokmabadi@univr.it

**Documentation:**
- [Main README](../README.md)
- [SAM Official Docs](https://segment-anything.com/)

---

## 📄 Technical Details

**Model:**
- **SAM** (Segment Anything Model) by Meta AI
- Variants: vit_b (91M), vit_l (308M), vit_h (636M parameters)
- Trained on SA-1B dataset (11M images, 1.1B masks)

**Server:**
- Framework: FastAPI + Uvicorn
- Port: 8001
- Max upload: 50MB per image
- Supports: CORS for external frontends

**Output:**
- Format: PNG (24-bit RGB for masked, 32-bit RGBA for transparent)
- Compression: Default PNG compression
- Structure: results/[session_id]/

**Dependencies:**
- PyTorch >= 1.13.0
- SAM >= 1.0
- FastAPI >= 0.104.0
- Pillow >= 9.5.0
- NumPy >= 1.24.0

---

## 🚀 Advanced Usage

### Manual Coordinate Input

Instead of clicking, input exact coordinates in format:

```
Include: [(x1,y1), (x2,y2), ...]
Exclude: [(x1,y1), (x2,y2), ...]
```

Example:
```
Include: [(512, 384), (600, 400)]
Exclude: [(200, 200)]
```

Useful for:
- Programmatic usage
- Reproducible results
- Scripted workflows

### API Endpoint

The server exposes REST API:

```bash
# Generate mask
curl -X POST http://localhost:8001/segment \
  -F "file=@plant.jpg" \
  -F "include_points=[[100,100],[200,200]]" \
  -F "exclude_points=[[50,50]]" \
  -F "model_type=vit_l"
```

Response includes base64-encoded masks.

---

<div align="center">

**Part of AgriSegment Suite 🌱**

[Main README](../README.md) | [Report Bug](https://github.com/vahidshokrani415/AgriSegment/issues) | [Request Feature](https://github.com/vahidshokrani415/AgriSegment/issues)

</div>