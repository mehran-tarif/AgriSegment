# ⚡ semantic/ - Automated SegFormer Processing

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Port](https://img.shields.io/badge/Port-8002-green.svg)](http://localhost:8002)
[![Model](https://img.shields.io/badge/Model-SegFormer-orange.svg)](https://huggingface.co/nvidia/segformer-b4-finetuned-ade-512-512)

**Fast automated batch processing for plant detection**

Part of the [AgriSegment Suite](../README.md) - The fastest tool for processing many images automatically.

---

## 🎯 Overview

The `semantic/` tool uses SegFormer for fully automatic plant segmentation. Upload multiple images, and get results in seconds - no manual intervention required.

**Perfect for:**
- Processing large image datasets (100+ images)
- Quick automated plant detection
- Batch analysis workflows
- Cases where you don't need interactive refinement

---

## ✨ Key Features

- ✅ **Fully automatic** - No manual points needed
- ✅ **Batch processing** - Upload multiple images at once
- ✅ **Fast execution** - 2-3 seconds per image
- ✅ **Plant-specific detection** - Trained on 6 plant classes
- ✅ **Comprehensive visualization** - 2x3 grid layout
- ✅ **Detailed statistics** - Coverage percentage, class counts
- ✅ **ZIP download** - All results in one package
- ✅ **Color-coded classes** - Easy visual interpretation

---

## 🚀 Quick Start

### Installation

```bash
cd semantic/
bash installer.sh
```

This installs SegFormer and dependencies.

### Run Server

```bash
python server.py
```

**Access:** http://localhost:8002

Model downloads automatically on first run (~5 minutes).

---

## 📖 How to Use

### Step 1: Upload Images
1. Go to http://localhost:8002
2. Click **"Upload Images"** or drag and drop
3. Can upload multiple files (batch processing)
4. Supported: JPG, PNG, JPEG, BMP

### Step 2: Process Images
1. Click **"Process Images"** button
2. Wait for automatic detection (2-5s per image)
3. No configuration needed - fully automatic

### Step 3: View Results
Results displayed in **2x3 grid** for each image:

1. **Original Image** - Your uploaded photo
2. **Plant Detection Mask** - Binary mask of detected plants
3. **Overlay** - Mask overlaid on original
4. **All Classes (Colored)** - All detected classes with labels
5. **Plants Only** - White background, plants in color
6. **Statistics Panel** - Coverage %, class breakdown

### Step 4: Download Results
- **View individual**: Click images to enlarge
- **Download all**: Click "Download as ZIP"
- Organized by image name in archive

---

## 🌿 Detected Plant Classes

SegFormer detects **6 plant-related classes** from ADE20K dataset:

| Class | Description | Color |
|-------|-------------|-------|
| **tree** | Trees, trunks, branches | Green |
| **grass** | Grass, lawn, ground cover | Light green |
| **plant** | General plants, shrubs | Medium green |
| **field** | Agricultural fields, crops | Yellow-green |
| **flower** | Flowers, flowering plants | Pink/Red |
| **palm** | Palm trees | Dark green |

**Total classes in model:** 150 (ADE20K), but only plant-related shown in results.

---

## 📊 Output Formats

### 2x3 Visualization Grid

Each processed image generates 6 visualizations:

#### Row 1: Basic Segmentation
1. **Original** - Unchanged input image
2. **Plant Mask** - Binary white/black mask
3. **Overlay** - Semi-transparent mask on original

#### Row 2: Detailed Analysis  
4. **All Classes** - All 150 classes color-coded with labels
5. **Plants Only** - Only plant classes, white background
6. **Statistics** - Text panel with metrics

### Statistics Included

```
Coverage: 42.3% of image
Classes detected:
- tree: 1,245 pixels (15.2%)
- grass: 2,134 pixels (26.1%)
- plant: 87 pixels (1.0%)
Total plant pixels: 3,466
```

---

## 🛠️ Configuration

### Change Port

Edit `server.py`:

```python
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002  # Change this
    )
```

### Adjust Batch Size

For memory management:

```python
BATCH_SIZE = 4  # Process 4 images at once (default)
# Reduce to 1-2 for low-memory systems
# Increase to 8-16 for high-memory systems
```

### Modify Plant Classes

Edit the plant class list in `server.py`:

```python
PLANT_CLASSES = [
    'tree', 'grass', 'plant', 'field', 'flower', 'palm',
    # Add more from ADE20K classes if needed
]
```

---

## 💡 Tips & Best Practices

### Image Preparation

**Best results with:**
- Clear plant visibility
- Good lighting (natural daylight preferred)
- Resolution 512x512 or higher
- Minimal occlusions

**Acceptable but may reduce accuracy:**
- Shadows (model handles reasonably well)
- Multiple plant types in one image
- Complex backgrounds
- Varying scales

### Batch Processing Strategy

**Small batch (1-10 images):**
- Process immediately
- Review each result
- Good for quick checks

**Medium batch (10-50 images):**
- Process in one go
- Download ZIP for offline review
- Check random samples for quality

**Large batch (50-500+ images):**
- Split into smaller batches if memory limited
- Process overnight for very large datasets
- Use statistics to identify outliers

### Quality Control

After batch processing:
1. Check a few random samples
2. Look for consistent detection across similar images
3. Note any systematic failures (e.g., specific plant type)
4. Use [`hybrid/`](../hybrid/README.md) to refine problematic images

---

## 🔧 Troubleshooting

### Poor Detection Quality

**Problem:** Plants not detected or background included

**Causes & Solutions:**
- **Low resolution**: Resize images to at least 512x512
- **Poor lighting**: Use images from well-lit conditions
- **Unusual plant types**: Model trained on common plants, may miss exotic species
- **Heavy occlusion**: Pre-crop images to focus on plants

**If automatic detection insufficient:**
- Use [`hybrid/`](../hybrid/README.md) for interactive refinement
- Use [`interactive/`](../interactive/README.md) for full manual control

### Slow Processing

**Problem:** Takes too long for batch

**Solutions:**
- Enable GPU (CUDA) if available
- Reduce image resolution before upload
- Increase `BATCH_SIZE` if you have RAM
- Process fewer images per batch

### Out of Memory

**Problem:** Crashes with large batches

**Solutions:**
- Set `BATCH_SIZE=1` in configuration
- Resize images to 1024x1024 max
- Process fewer images at once (10-20 instead of 100)
- Close other applications
- Use CPU mode if GPU memory limited

### Classes Not Detected

**Problem:** Know there are plants but nothing detected

**Solutions:**
- Verify image quality and resolution
- Try images with more obvious/visible plants
- Check if plant types are in supported classes
- Some plants may be classified as 'background' if unusual

---

## 📊 Performance Benchmarks

Tested on NVIDIA RTX 3090 (24GB VRAM):

| Scenario | Time | Memory |
|----------|------|--------|
| Single image (1024x1024) | 2-3s | ~2GB |
| Batch 10 images | 15-20s | ~4GB |
| Batch 50 images | 60-80s | ~8GB |
| Batch 100 images | 120-160s | ~12GB |

**CPU-only performance:**
- 3-5x slower than GPU
- Can still process large batches (just takes longer)
- Recommended for smaller datasets (<20 images)

---

## 🎯 Use Cases

### Research & Academia
- Phenotyping large plant datasets
- Time-series crop monitoring
- Vegetation coverage studies
- Pre-processing for machine learning

### Agriculture
- Field survey analysis
- Crop health monitoring over time
- Automated plant counting
- Coverage percentage tracking

### Dataset Preparation
- Initial annotation of large datasets
- Quick quality check of image collections
- Filtering images with/without plants
- Batch pre-labeling before manual refinement

### Production Pipelines
- First stage of automated workflow
- Feed results to downstream analysis
- Integration with other tools
- Continuous monitoring systems

---

## 🔗 Related Tools

Part of **AgriSegment Suite**:

- [`hybrid/`](../hybrid/README.md) - Add interactive refinement after automatic detection
- [`interactive/`](../interactive/README.md) - Full manual control for single images
- [`panoptic/`](../panoptic/README.md) - Instance-level detection (separates individual plants)

**When to use `semantic/` instead:**
- Processing many images (10+)
- Speed is priority over perfect accuracy
- Don't need interactive refinement
- Acceptable if some errors in automatic detection

**When to use other tools:**
- [`hybrid/`] when you need to refine automatic results
- [`interactive/`] for careful single-image annotation
- [`panoptic/`] when you need to separate individual plant instances

---

## 🔄 Workflow Integration

### Typical Pipeline

```
semantic/ (batch auto-processing)
    ↓
Review results
    ↓
Identify images needing refinement
    ↓
hybrid/ (refine selected images)
    ↓
Final high-quality dataset
```

### Combined with Other Tools

**Quality control workflow:**
1. Process all images with `semantic/`
2. Use statistics to identify outliers
3. Refine outliers with `hybrid/` or `interactive/`

**Iterative improvement:**
1. Initial batch with `semantic/`
2. Manually correct subset with `hybrid/`
3. Retrain SegFormer on corrected data
4. Repeat for continuous improvement

---

## 📞 Support

**Issues?**
- Check [main README troubleshooting](../README.md#troubleshooting)
- Open issue on [GitHub](https://github.com/vahidshokrani415/AgriSegment/issues)
- Email: mehran.tarifhokmabadi@univr.it

**Documentation:**
- [Main README](../README.md)
- [SegFormer Paper](https://arxiv.org/abs/2105.15203)

---

## 📄 Technical Details

**Model:**
- **SegFormer-B4** fine-tuned on ADE20K
- Model ID: `nvidia/segformer-b4-finetuned-ade-512-512`
- Parameters: ~64M
- Input: RGB images (any size, auto-resized)
- Output: 150-class semantic segmentation

**Architecture:**
- Encoder: Hierarchical Transformer
- Decoder: Lightweight All-MLP
- Training: ADE20K dataset (20K+ images, 150 classes)

**Server:**
- Framework: FastAPI + Uvicorn
- Port: 8002
- Max upload: 50MB per image
- Concurrent processing: Configurable batch size

**Dependencies:**
- PyTorch >= 1.13.0
- Transformers >= 4.30.0
- FastAPI >= 0.104.0
- Matplotlib >= 3.7.0
- Pillow >= 9.5.0

**Output Structure:**
```
results/
└── session_[id]/
    ├── image1_grid.png       # 2x3 visualization
    ├── image1_stats.txt      # Statistics text
    ├── image2_grid.png
    └── all_results.zip
```

---

## 🚀 Advanced Usage

### API Endpoint

REST API for programmatic access:

```bash
# Process single image
curl -X POST http://localhost:8002/segment \
  -F "file=@plant.jpg" \
  -o result.png

# Batch processing
curl -X POST http://localhost:8002/segment_batch \
  -F "files=@img1.jpg" \
  -F "files=@img2.jpg" \
  -F "files=@img3.jpg"
```

### Custom Class Selection

Modify which classes to display:

```python
# In server.py
DISPLAY_CLASSES = ['tree', 'grass']  # Show only these

# Or show all 150 classes
DISPLAY_ALL = True
```

### Export Formats

Configure output format in `server.py`:

```python
EXPORT_FORMATS = {
    'grid': True,        # 2x3 visualization
    'mask_only': True,   # Binary mask
    'overlay': True,     # Colored overlay
    'stats_json': True,  # JSON statistics
}
```

---

<div align="center">

**Part of AgriSegment Suite 🌱**

[Main README](../README.md) | [Report Bug](https://github.com/vahidshokrani415/AgriSegment/issues) | [Request Feature](https://github.com/vahidshokrani415/AgriSegment/issues)

</div>