# Quick Start Guide

Get started with AgriSegment in 5 minutes!

---

## ⚡ 5-Minute Setup

### Step 1: Install (2 minutes)

```bash
# Clone repository
git clone https://github.com/vahidshokrani415/AgriSegment.git
cd AgriSegment

# Choose and install ONE tool to start
cd interactive/  # Recommended for beginners
bash installer.sh
```

Wait for installation to complete (~2 minutes).

---

### Step 2: Start Server (30 seconds)

```bash
python server.py
```

You'll see:
```
INFO:     Uvicorn running on http://0.0.0.0:8001
INFO:     Application startup complete
```

---

### Step 3: Open Web Interface (10 seconds)

Open your browser and go to:
```
http://localhost:8001
```

You should see the AgriSegment interface!

---

### Step 4: Try Your First Segmentation (2 minutes)

1. **Click "Choose File"** or drag and drop an image
2. **Click on the plant** (adds a green include point)
3. **Click "Generate Mask"**
4. **Wait ~3 seconds**
5. **View your result!**

**Download options:**
- Binary mask (black & white)
- Masked image (plant with black background)
- Transparent PNG (plant only)

---

## 🎯 Choose Your Tool

Now that you've tried `interactive/`, explore others:

### For Quick Single Images
✅ **Already installed!** (`interactive/`)
- Fast and simple
- Point and click
- Perfect for learning

### For Many Images
```bash
cd ../semantic/
bash installer.sh
python server.py
```
- Visit http://localhost:8002
- Upload multiple images
- Automatic detection
- Download all as ZIP

### For Best Accuracy
```bash
cd ../hybrid/
bash installer.sh
python server.py
```
- Visit http://localhost:8000
- Automatic + interactive
- Refine results
- Feedback loop

### For Advanced Analysis
```bash
cd ../panoptic/
bash installer.sh
python server.py
```
- Visit http://localhost:8003
- Count individual plants
- Instance segmentation
- Multiple modes

---

## 📸 First Time Tips

### Good Images for Testing
- ✅ Clear plant visibility
- ✅ Good lighting
- ✅ Simple background
- ✅ At least 512x512 pixels

### Where to Click (interactive/)
- **Include point**: Center of plant
- **Exclude point**: Clear background
- **Start simple**: 2-3 points usually enough

### If Results Not Good
1. Add more include points
2. Add exclude points near edges
3. Try different model size (Base/Large/Huge)
4. Check image quality

---

## 🚀 Common First Tasks

### Task 1: Segment One Image

**Tool:** `interactive/`

**Steps:**
1. Upload image
2. Click plant center (green point)
3. Generate mask
4. Download result

**Time:** 1 minute

---

### Task 2: Process 10 Images Quickly

**Tool:** `semantic/`

**Steps:**
1. Select all 10 images
2. Upload (drag and drop)
3. Click "Process"
4. Download ZIP

**Time:** 2-3 minutes

---

### Task 3: Get Perfect Segmentation

**Tool:** `hybrid/`

**Steps:**
1. Upload image
2. Click "Generate Points" (automatic)
3. Review auto-generated points
4. Add/remove points manually
5. Generate final mask
6. Download

**Time:** 3-5 minutes

---

### Task 4: Count Individual Plants

**Tool:** `panoptic/`

**Steps:**
1. Upload image
2. Select "Instance" mode
3. Select "Large" model
4. Process
5. View count in statistics

**Time:** 10-30 seconds

---

## ❓ Quick Troubleshooting

### Server Won't Start
```bash
# Check Python version (need 3.8+)
python --version

# Try reinstalling
bash installer.sh
```

### Can't Access Website
- Check server is running (see terminal)
- Try http://127.0.0.1:8001 instead
- Check firewall settings

### Upload Fails
- File too large (max 50MB)
- Wrong format (use JPG or PNG)
- Resize image to under 4096x4096

### Poor Results
- Use better quality image
- Add more points
- Try different tool
- Read [Best Practices](best-practices.md)

---

## 📚 What's Next?

### Learn More
- [Installation Guide](installation.md) - Detailed setup
- [Best Practices](best-practices.md) - Get better results
- [Troubleshooting](troubleshooting.md) - Fix common issues
- [API Reference](api.md) - Programmatic access

### Try Advanced Features
- Compare different tools on same image
- Process large batches (100+ images)
- Refine automatic results interactively
- Export for training datasets

### Join Community
- Star on [GitHub](https://github.com/vahidshokrani415/AgriSegment)
- Report issues or bugs
- Suggest improvements
- Share your results!

---

## 💡 Pro Tips

1. **Start with `interactive/`** - Simplest to learn
2. **Use `semantic/` for speed** - When you have many images
3. **Use `hybrid/` for quality** - When accuracy matters
4. **Use `panoptic/` for counting** - When you need instances

**Golden Rule:** Right tool for the job = Better results + Less time

---

## 📞 Need Help?

**Quick help:**
- Check [Troubleshooting](troubleshooting.md)
- Read tool-specific README
- Search [GitHub Issues](https://github.com/vahidshokrani415/AgriSegment/issues)

**Still stuck?**
- Email: mehran.tarifhokmabadi@univr.it
- Open [GitHub Issue](https://github.com/vahidshokrani415/AgriSegment/issues/new)

---

<div align="center">

**[← Back to Main README](../README.md)** | **[Best Practices →](best-practices.md)**

**Happy Segmenting! 🌱**

</div>