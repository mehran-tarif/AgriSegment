# 🔬 panoptic/ - Advanced Mask2Former Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Port](https://img.shields.io/badge/Port-8003-green.svg)](http://localhost:8003)
[![Model](https://img.shields.io/badge/Model-Mask2Former-orange.svg)](https://github.com/facebookresearch/Mask2Former)

**Multi-modal segmentation with semantic, instance, and panoptic modes**

Part of the [AgriSegment Suite](../README.md) - The most advanced tool for detailed plant analysis.

---

## 🎯 Overview

The `panoptic/` tool uses Mask2Former, providing three different segmentation approaches:

1. **Semantic** - Classifies every pixel (like `semantic/`)
2. **Instance** - Separates individual objects (counts separate plants)
3. **Panoptic** - Combines both semantic + instance

**Perfect for:**
- Separating individual plants in crowded scenes
- Counting discrete plant instances
- Advanced research requiring instance-level data
- Comparative analysis across segmentation modes

---

## ✨ Key Features

- ✅ **Three segmentation modes** - Semantic, Instance, Panoptic
- ✅ **Two model sizes** - Base (faster) and Large (more accurate)
- ✅ **Confidence threshold control** - Adjust detection sensitivity
- ✅ **Plant-specific detection** - Filters for agriculture-relevant classes
- ✅ **Instance counting** - Number of individual plants detected
- ✅ **Comprehensive statistics** - Coverage, confidence scores, class breakdown
- ✅ **2x3 visualization grid** - Multiple views of results
- ✅ **Batch processing** - Handle multiple images
- ✅ **Model preloading option** - Faster startup with PRELOAD_ALL_MODELS flag

---

## 🚀 Quick Start

### Installation

```bash
cd panoptic/
bash installer.sh
```

This installs Mask2Former and dependencies.

### Run Server

```bash
python server.py
```

**Access:** http://localhost:8003

Models download on first use (5-10 minutes per model).

### Optional: Preload All Models

Edit `server.py` before starting:

```python
PRELOAD_ALL_MODELS = True  # Load all 6 models at startup
```

This increases startup time but makes switching between modes instant.

---

## 📖 How to Use

### Step 1: Upload Images
1. Go to http://localhost:8003
2. Click **"Upload Images"** or drag and drop
3. Supports: JPG, PNG, JPEG, BMP
4. Can upload multiple images

### Step 2: Configure Settings

**Select Segmentation Mode:**
- **Semantic** - Pixel-wise classification
- **Instance** - Separate individual objects
- **Panoptic** - Both semantic + instance

**Select Model Size:**
- **Base** - Faster, good quality
- **Large** - Slower, best quality

**Adjust Confidence Threshold:**
- Range: 0.0 to 1.0
- Default: 0.5
- Higher = fewer but more confident detections
- Lower = more detections but may include noise

### Step 3: Process Images
1. Click **"Process"** button
2. Wait for analysis (5-30s per image depending on mode/model)
3. Results appear in 2x3 grid

### Step 4: Review Results

**Grid Layout:**
1. **Original Image**
2. **Segmentation Mask** (colored by class/instance)
3. **Overlay** (mask on original)
4. **All Classes** (all detections labeled)
5. **Plants Only** (filtered to plant classes)
6. **Statistics Panel** (metrics and counts)

### Step 5: Download Results
- Individual images: Click to save
- All results: "Download as ZIP"

---

## 🔍 Segmentation Modes Explained

### Semantic Segmentation
**What it does:** Classifies every pixel into a class

**Example:** "This pixel is tree, that pixel is grass"

**Best for:**
- Total coverage percentage
- General plant vs non-plant classification
- Simple binary masks

**Limitations:**
- Cannot count individual plants
- Cannot separate overlapping instances

---

### Instance Segmentation
**What it does:** Detects and separates individual object instances

**Example:** "Plant #1, Plant #2, Plant #3" (even if touching)

**Best for:**
- Counting individual plants
- Measuring each plant separately
- Tracking specific plants over time

**Limitations:**
- Only works for "thing" classes (countable objects)
- May miss very small or heavily occluded instances

---

### Panoptic Segmentation
**What it does:** Combines semantic + instance segmentation

**Example:** "Plant #1 (instance), Plant #2 (instance), background grass (semantic)"

**Best for:**
- Most comprehensive analysis
- Research requiring both types of info
- Complex scenes with multiple object types

**Provides:**
- Instance counts for "thing" classes
- Semantic coverage for "stuff" classes
- Complete scene understanding

---

## 🌿 Detected Classes

Mask2Former detects from two datasets:

### ADE20K Classes (Semantic/Panoptic models)
Plant-related classes filtered:
- tree, grass, plant, field, flower, palm, shrub, etc.

### COCO Classes (Instance model)
Plant-related classes filtered:
- potted plant, vase with plants, etc.

**Note:** Different models trained on different datasets, so available classes vary by mode.

---

## 📊 Statistics Provided

### For All Modes
- Total pixels processed
- Plant coverage percentage
- Number of classes detected
- Confidence scores

### Semantic Mode
- Pixels per class
- Class coverage percentages

### Instance Mode
- Number of instances (individual plants)
- Area per instance
- Confidence per instance
- Instance IDs

### Panoptic Mode
- Everything from semantic + instance
- Separated by "things" (countable) and "stuff" (amorphous)

---

## 🛠️ Configuration

### Change Port

Edit `server.py`:

```python
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003  # Change this
    )
```

### Preload Models

Edit `server.py`:

```python
PRELOAD_ALL_MODELS = True  # Load all models at startup
# False = load on demand (saves startup time)
```

### Default Settings

```python
DEFAULT_MODE = "panoptic"  # semantic, instance, or panoptic
DEFAULT_MODEL = "large"    # base or large
DEFAULT_THRESHOLD = 0.5    # 0.0 to 1.0
```

### Model Selection

Available model combinations:

```python
MODELS = {
    'semantic_base': 'facebook/mask2former-swin-base-ade-semantic',
    'semantic_large': 'facebook/mask2former-swin-large-ade-semantic',
    'instance_base': 'facebook/mask2former-swin-base-coco-instance',
    'instance_large': 'facebook/mask2former-swin-large-coco-instance',
    'panoptic_base': 'facebook/mask2former-swin-base-ade-panoptic',
    'panoptic_large': 'facebook/mask2former-swin-large-ade-panoptic',
}
```

---

## 💡 Tips & Best Practices

### Mode Selection Guide

| Your Goal | Use This Mode |
|-----------|---------------|
| Count individual plants | **Instance** |
| Measure total coverage | **Semantic** |
| Most complete information | **Panoptic** |
| Separate overlapping plants | **Instance** or **Panoptic** |
| Just need plant vs background | **Semantic** |

### Model Size Selection

| Scenario | Recommended |
|----------|-------------|
| Quick tests | **Base** |
| Research/publications | **Large** |
| Real-time processing | **Base** |
| Maximum accuracy | **Large** |
| Limited GPU memory | **Base** |

### Confidence Threshold Tuning

**High threshold (0.7-0.9):**
- Fewer detections
- More confident predictions
- May miss some plants
- Good for precision-critical tasks

**Medium threshold (0.4-0.6):**
- Balanced
- Good default for most uses
- Reasonable false positive/negative trade-off

**Low threshold (0.1-0.3):**
- More detections
- May include false positives
- Better recall
- Good for ensuring nothing is missed

### Image Preparation

**Best results:**
- High resolution (1024x1024 or larger)
- Clear plant visibility
- Good lighting
- Minimal motion blur

**Acceptable:**
- Multiple plants in scene
- Some occlusion
- Varying scales
- Complex backgrounds

---

## 🔧 Troubleshooting

### No Instances Detected (Instance Mode)

**Problem:** Instance mode shows 0 plants detected

**Solutions:**
- Lower confidence threshold (try 0.3)
- Check if plants are in COCO dataset classes
- Try panoptic mode instead (different training data)
- Ensure plants are clearly visible and not too small

### Oversegmentation

**Problem:** One plant split into multiple instances

**Solutions:**
- Increase confidence threshold
- Use semantic mode instead
- Use larger model (more context understanding)
- Improve image quality (higher resolution)

### Undersegmentation

**Problem:** Multiple plants merged into one instance

**Solutions:**
- Lower confidence threshold
- Use panoptic mode (better separation)
- Ensure good contrast between plants
- Try base model (sometimes faster = better for separation)

### Slow Processing

**Problem:** Takes too long per image

**Solutions:**
- Use base model instead of large
- Reduce image resolution
- Enable GPU (CUDA)
- Process fewer images at once

### Out of Memory

**Problem:** GPU/RAM exhausted

**Solutions:**
- Use base model (smaller memory footprint)
- Reduce image size (resize to 1024x1024 max)
- Process one image at a time
- Set `PRELOAD_ALL_MODELS=False`
- Close other applications

---

## 📊 Performance Benchmarks

Tested on NVIDIA RTX 3090 (24GB VRAM), 1024x1024 images:

| Mode | Base Model | Large Model |
|------|------------|-------------|
| **Semantic** | 3-5s | 8-12s |
| **Instance** | 4-6s | 10-15s |
| **Panoptic** | 5-8s | 12-18s |

**Memory Usage:**
- Base models: ~6-8GB GPU memory
- Large models: ~12-16GB GPU memory

**CPU-only:**
- 5-10x slower than GPU
- Recommended only for small batches

---

## 🎯 Use Cases

### Research Applications

**Plant Phenotyping:**
- Count individual plants in field plots
- Measure per-plant characteristics
- Track plant growth over time
- Instance-level statistics

**Crop Monitoring:**
- Weed vs crop separation (if trained)
- Disease spot detection (instance mode)
- Coverage estimation (semantic mode)
- Yield prediction (count instances)

**Ecological Studies:**
- Species distribution (if classes available)
- Vegetation coverage analysis
- Biodiversity assessment
- Habitat mapping

### Agricultural Applications

**Field Assessment:**
- Plant counting for density analysis
- Individual plant health monitoring
- Harvest readiness evaluation
- Planting pattern verification

**Precision Agriculture:**
- Per-plant treatment planning
- Selective harvesting guidance
- Individual plant tracking
- Performance comparison

---

## 🔗 Related Tools

Part of **AgriSegment Suite**:

- [`hybrid/`](../hybrid/README.md) - Interactive refinement with SegFormer + SAM
- [`interactive/`](../interactive/README.md) - Manual SAM segmentation
- [`semantic/`](../semantic/README.md) - Fast SegFormer batch processing

**When to use `panoptic/` instead:**
- Need to count individual plants (instance mode)
- Want multiple segmentation perspectives
- Research requiring comprehensive analysis
- Need instance-level statistics

**When to use other tools:**
- [`semantic/`] for faster pure automatic segmentation
- [`hybrid/`] when you need interactive refinement
- [`interactive/`] for full manual control

---

## 🔄 Comparative Analysis

### vs semantic/ (SegFormer)

**panoptic/ advantages:**
- Instance counting
- Multiple modes
- Often more accurate (newer architecture)

**semantic/ advantages:**
- Faster processing
- Simpler output
- Lower memory usage

### vs hybrid/ (SegFormer + SAM)

**panoptic/ advantages:**
- Instance segmentation capability
- No manual points needed
- Multiple analysis modes

**hybrid/ advantages:**
- Interactive refinement
- Feedback loop for improvement
- More control over results

---

## 📞 Support

**Issues?**
- Check [main README troubleshooting](../README.md#troubleshooting)
- Open issue on [GitHub](https://github.com/vahidshokrani415/AgriSegment/issues)
- Email: mehran.tarifhokmabadi@univr.it

**Documentation:**
- [Main README](../README.md)
- [Mask2Former Paper](https://arxiv.org/abs/2112.01527)
- [Official Repo](https://github.com/facebookresearch/Mask2Former)

---

## 📄 Technical Details

**Model:**
- **Mask2Former** by Meta AI
- Architecture: Masked-attention Transformer decoder
- Training: ADE20K (semantic/panoptic), COCO (instance)

**Model Sizes:**
- Base: Swin-Base backbone (~90M parameters)
- Large: Swin-Large backbone (~220M parameters)

**Modes:**
- Semantic: 150 classes (ADE20K)
- Instance: 80 classes (COCO)
- Panoptic: 150 semantic + instances

**Server:**
- Framework: FastAPI + Uvicorn
- Port: 8003
- Max upload: 50MB per image
- Model caching: Optional preloading

**Dependencies:**
- PyTorch >= 1.13.0
- Transformers >= 4.30.0
- Detectron2 >= 0.6
- FastAPI >= 0.104.0
- Pillow >= 9.5.0

**Output Structure:**
```
results/
└── session_[id]/
    ├── image1_grid.png          # 2x3 visualization
    ├── image1_stats.json        # Detailed statistics
    ├── image1_instances.json    # Instance data (if applicable)
    └── all_results.zip
```

---

## 🚀 Advanced Usage

### API Endpoints

```bash
# Semantic segmentation
curl -X POST http://localhost:8003/segment \
  -F "file=@plant.jpg" \
  -F "mode=semantic" \
  -F "model_size=large" \
  -F "threshold=0.5"

# Instance segmentation
curl -X POST http://localhost:8003/segment \
  -F "file=@plant.jpg" \
  -F "mode=instance" \
  -F "threshold=0.6"

# Panoptic segmentation
curl -X POST http://localhost:8003/segment \
  -F "file=@plant.jpg" \
  -F "mode=panoptic"
```

### Custom Class Filtering

Modify which classes to show:

```python
# In server.py
PLANT_CLASSES_ADE = [
    'tree', 'grass', 'plant', 'field', 'flower', 'palm'
    # Add/remove as needed
]

PLANT_CLASSES_COCO = [
    'potted plant', 'vase'
    # Limited plant classes in COCO
]
```

### Statistics Export

Configure output detail:

```python
EXPORT_DETAILED_STATS = True  # Include per-instance data
EXPORT_JSON = True            # JSON statistics
EXPORT_CSV = True             # CSV format
```

---

## 🔬 Research Applications

### Example Analyses

**Plant Counting Study:**
1. Use instance mode
2. Set threshold to 0.6
3. Count instances per image
4. Export statistics to CSV
5. Statistical analysis in R/Python

**Coverage Analysis:**
1. Use semantic mode
2. Process time-series images
3. Calculate coverage % over time
4. Correlate with growth conditions

**Multi-modal Comparison:**
1. Process same image in all 3 modes
2. Compare semantic coverage vs instance count
3. Validate instance segmentation quality
4. Choose best mode for dataset

---

<div align="center">

**Part of AgriSegment Suite 🌱**

[Main README](../README.md) | [Report Bug](https://github.com/vahidshokrani415/AgriSegment/issues) | [Request Feature](https://github.com/vahidshokrani415/AgriSegment/issues)

</div>