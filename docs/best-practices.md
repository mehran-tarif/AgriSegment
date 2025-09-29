# Best Practices Guide

Guidelines for getting optimal results from AgriSegment Suite.

---

## 📸 Image Preparation

### Resolution Guidelines

**Minimum acceptable:** 512x512 pixels

**Recommended:** 1024x1024 to 2048x2048 pixels

**Maximum practical:** 4096x4096 pixels

**Why:**
- Too small: Poor segmentation accuracy, missed details
- Too large: Slow processing, high memory usage
- Sweet spot: 1024-2048px balances quality and speed

**Resize images before upload:**
```bash
# Using ImageMagick
convert input.jpg -resize 1024x1024 output.jpg

# Using Python/Pillow
from PIL import Image
img = Image.open('input.jpg')
img.thumbnail((1024, 1024))
img.save('output.jpg')
```

---

### Lighting and Exposure

**Best practices:**
- ✅ Natural daylight (overcast is ideal)
- ✅ Even, diffused lighting
- ✅ Avoid direct harsh sunlight
- ✅ No strong shadows
- ✅ Proper exposure (not too dark/bright)

**Acceptable:**
- ⚠️ Light shadows
- ⚠️ Slightly overcast
- ⚠️ Indoor with good artificial light

**Avoid:**
- ❌ Heavy shadows obscuring plants
- ❌ Backlit images (silhouettes)
- ❌ Extreme over/underexposure
- ❌ Mixed lighting (half sun, half shadow)

---

### Image Quality

**Critical factors:**
- ✅ Sharp focus on plants
- ✅ High contrast plant vs background
- ✅ Clear plant edges
- ✅ Minimal motion blur
- ✅ Good color saturation

**Tips:**
- Use camera stabilization
- Higher ISO if needed for sharpness
- Shoot in RAW and export as high-quality JPEG
- Avoid heavy compression

---

### Background Considerations

**Best backgrounds:**
- Soil/dirt (dark, uniform)
- Mulch
- Gravel
- Clear sky (for tree canopy)

**Challenging but workable:**
- Mixed vegetation
- Partial grass cover
- Complex backgrounds

**Difficult:**
- Very similar color to plants
- Patterned backgrounds
- Reflective surfaces

**Tips:**
- If possible, use contrasting backgrounds
- Remove debris near plants
- Use controlled imaging setups when available

---

## 🎯 Tool Selection Strategy

### Decision Tree

```
Need to segment plants?
├─ Single image, quick task → interactive/
├─ Many images (10+), automatic OK → semantic/
├─ Need highest accuracy, can refine → hybrid/
└─ Need instance counting/separation → panoptic/
```

### Detailed Selection Guide

#### Use `interactive/` when:
- Processing 1-5 images only
- Want full manual control
- Learning how segmentation works
- Need quick results without setup
- Automatic detection not working well

#### Use `semantic/` when:
- Processing 10-500+ images
- Speed more important than perfection
- Results don't need refinement
- Creating initial annotations
- Coverage percentage is main metric

#### Use `hybrid/` when:
- Need research-grade accuracy
- Can spend time refining results
- Building training datasets
- Implementing feedback loop
- Automatic results need corrections

#### Use `panoptic/` when:
- Need to count individual plants
- Require instance-level statistics
- Separating overlapping plants
- Comparing semantic vs instance segmentation
- Advanced analysis needed

---

## 🔄 Workflow Strategies

### Strategy 1: Fast Batch Processing

**Goal:** Process many images quickly

**Steps:**
1. Prepare images (resize, good quality)
2. Use `semantic/` for batch processing
3. Review sample of results (10-20%)
4. Accept results if quality sufficient
5. If issues, refine selected images with `hybrid/`

**Best for:**
- Initial dataset screening
- Coverage analysis
- Time-constrained projects
- When approximate accuracy acceptable

---

### Strategy 2: High-Quality Annotations

**Goal:** Create perfect segmentation masks

**Steps:**
1. Start with `semantic/` or `hybrid/` automatic detection
2. Review all results carefully
3. Refine each image with `hybrid/` or `interactive/`
4. Verify edge accuracy
5. Export final masks

**Best for:**
- Ground truth creation
- Publication-quality figures
- Training data for ML models
- Phenotyping measurements

---

### Strategy 3: Iterative Improvement

**Goal:** Continuously improve automatic segmentation

**Steps:**
1. Process batch with `semantic/`
2. Manually correct 10-20% with `hybrid/`
3. Export corrected masks as training data
4. Fine-tune SegFormer on corrected data
5. Reprocess with improved model
6. Repeat cycle

**Best for:**
- Long-term projects
- Large datasets (1000+ images)
- Domain-specific applications
- Building custom models

---

### Strategy 4: Quality Control Pipeline

**Goal:** Ensure consistent quality across large datasets

**Steps:**
1. `semantic/` processes all images
2. Calculate statistics (coverage, confidence)
3. Flag outliers (too high/low coverage)
4. Review flagged images manually
5. Refine problematic images with `hybrid/`
6. Final dataset combines auto + refined

**Best for:**
- Production environments
- Research requiring consistency
- Automated monitoring systems
- Large-scale phenotyping

---

## 🎨 Interactive Refinement Tips

### Point Placement Strategy

#### For `interactive/` and `hybrid/`

**Include points (green):**
- Place in center of plants
- Add near edges for precise boundaries
- Use 3-5 points for simple plants
- Use 10-15 points for complex morphology

**Exclude points (red):**
- Place on clear background
- Add near plant edges if background similar color
- Use sparingly (2-5 points usually enough)

**Optimization:**
- Start with minimal points (1-3)
- Add more only if boundaries inaccurate
- More points ≠ always better
- Strategic placement > quantity

---

### Model Size Selection

#### SAM Models (`interactive/` and `hybrid/`)

| Model | Use When | Speed | Quality |
|-------|----------|-------|---------|
| **Base (vit_b)** | Testing, iteration, many images | Fast (1-2s) | Good |
| **Large (vit_l)** | General use, balanced needs | Medium (3-5s) | Better |
| **Huge (vit_h)** | Final results, publications | Slow (8-12s) | Best |

**Recommendation:**
- Start with Base for testing
- Use Large for most work
- Switch to Huge for final export

---

### Refinement Workflow

**Efficient refinement process:**

1. **Quick first pass** (Base model, 2-3 points)
   - Get rough segmentation fast
   - Identify problem areas

2. **Targeted refinement** (Large model, 5-10 points)
   - Add points to problem areas
   - Fix major boundary errors

3. **Final polish** (Huge model if needed, 10-15 points)
   - Perfect edges
   - Export final result

**Time estimates:**
- Quick: 30 seconds per image
- Standard: 2-3 minutes per image
- Detailed: 5-10 minutes per image

---

## 📊 Batch Processing Best Practices

### Batch Size Guidelines

**Small batches (1-10 images):**
- Test new settings
- Quality control checks
- Diverse image types

**Medium batches (10-50 images):**
- Standard workflows
- Similar image conditions
- Balanced speed/monitoring

**Large batches (50-500+ images):**
- Production processing
- Consistent imaging conditions
- Automated pipelines

**Very large (500+):**
- Split into smaller batches
- Process overnight
- Monitor first batch carefully

---

### Organizing Your Data

**Before processing:**
```
project/
├── raw_images/           # Original unprocessed images
│   ├── field_A/
│   ├── field_B/
│   └── greenhouse/
├── prepared_images/      # Resized, renamed for processing
└── results/              # Segmentation outputs
```

**Naming convention:**
```
YYYYMMDD_location_plot_rep_plant.jpg
20250929_fieldA_p01_r01_001.jpg
```

**Why:**
- Easy to track source
- Organize by conditions
- Batch by location/date
- Reproducible processing

---

### Quality Assurance

**Check these regularly:**

1. **Sample random images**
   - Review 5-10% of results
   - Check different conditions
   - Verify consistency

2. **Monitor statistics**
   - Coverage percentages
   - Detection confidence
   - Class distributions

3. **Watch for patterns**
   - Systematic errors?
   - Specific conditions failing?
   - Time-of-day effects?

4. **Document issues**
   - Note problematic image types
   - Record settings used
   - Track accuracy metrics

---

## 💾 Data Management

### Saving Results

**Essential exports:**
- ✅ Binary masks (for analysis)
- ✅ Overlays (for visualization)
- ✅ Statistics (JSON/CSV)
- ✅ Processing settings used

**Organize by purpose:**
```
results/
├── for_analysis/         # Binary masks, stats
├── for_publication/      # High-quality overlays
├── for_training/         # Masks + original images
└── quality_control/      # Flagged images
```

---

### Backup Strategy

**Critical data:**
1. Original images (never delete)
2. Refined segmentations (time-consuming to redo)
3. Processing logs (reproducibility)
4. Model checkpoints (if custom trained)

**Backup schedule:**
- Daily: Active project results
- Weekly: Complete project folder
- Monthly: Archive to external storage

---

## 🚀 Performance Optimization

### Hardware Considerations

**GPU vs CPU:**
- GPU: 5-10x faster, worth investment for large projects
- CPU: Acceptable for small batches (<20 images)

**Memory management:**
- 16GB RAM minimum for comfortable use
- 32GB+ for large batches
- GPU 8GB+ VRAM for all models

**Storage:**
- SSD for model cache (faster loading)
- HDD acceptable for image storage
- 50-100GB free space for models and results

---

### Processing Speed Tips

**Fast processing:**
1. Use smaller models (Base not Huge)
2. Reduce image resolution
3. Enable GPU acceleration
4. Close other applications
5. Process similar images together

**Balanced:**
1. Use Large/medium models
2. Standard resolutions (1024-2048)
3. Moderate batch sizes (10-20)
4. Monitor resource usage

**Maximum quality (slower):**
1. Use Huge/Large models
2. High resolution (2048+)
3. Small batches for review
4. Individual refinement

---

## 📈 Measuring Success

### Quality Metrics

**Visual assessment:**
- Boundaries accurate?
- Plant vs background correct?
- Consistent across similar images?

**Quantitative (if ground truth available):**
- IoU (Intersection over Union)
- Dice coefficient
- Pixel accuracy
- False positive/negative rate

**Practical metrics:**
- Coverage percentage reasonable?
- Instance counts match manual count?
- Statistics consistent across replicates?

---

### When to Stop Refining

**Good enough when:**
- ✅ Boundaries within 2-3 pixels
- ✅ Major plant parts captured
- ✅ No obvious background included
- ✅ Consistent quality across dataset

**Continue refining if:**
- ❌ Large boundary errors (>5 pixels)
- ❌ Missing significant plant parts
- ❌ Excessive background included
- ❌ Quality varies wildly

**Perfection not always needed:**
- Small errors often negligible for analysis
- Time better spent on more images
- Consider accuracy vs effort trade-off

---

## 🔬 Research Applications

### For Publications

**Requirements:**
- High-resolution source images (2048+)
- Huge SAM model for refinement
- Manual verification of all results
- Document all processing steps
- Save processing parameters

**Reporting:**
```
"Plant segmentation was performed using AgriSegment (Mehran & 
Quaglia, 2026) hybrid/ tool. Images were processed with SegFormer 
automatic detection followed by interactive SAM refinement using 
the vit_h (huge) model. All segmentations were manually verified 
and corrected where necessary."
```

---

### For Phenotyping

**Best practices:**
- Process time-series consistently
- Use same tool/settings throughout
- Monitor calibration with standards
- Track imaging conditions
- Replicate measurements

**Key measurements:**
- Coverage area (pixels or cm²)
- Growth rate (area over time)
- Color indices (if applicable)
- Morphological features

---

### For Training Data

**Dataset quality checklist:**
- [ ] Diverse plant types/stages
- [ ] Various imaging conditions
- [ ] Verified ground truth
- [ ] Consistent annotation quality
- [ ] Balanced classes
- [ ] Documented metadata

**Using hybrid/ feedback loop:**
1. Create initial labels automatically
2. Refine 10-20% manually
3. Train model on refined data
4. Generate new labels with improved model
5. Repeat for iterative improvement

---

## 📞 Getting Help

**Before asking:**
1. Read tool-specific README
2. Check [Troubleshooting Guide](troubleshooting.md)
3. Search existing GitHub issues

**When asking:**
- Describe what you tried
- Share example images if possible
- Include error messages
- Specify tool and settings used

**Contact:**
- GitHub Issues: Technical problems
- Email: Research collaboration
- Documentation: Suggestions for improvement

---

<div align="center">

**[← Back to Main README](../README.md)** | **[Troubleshooting →](troubleshooting.md)**

</div>