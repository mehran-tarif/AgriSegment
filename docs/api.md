# API Reference

REST API documentation for AgriSegment Suite.

---

## 📡 Overview

All tools expose REST APIs for programmatic access. This enables:
- Integration with other software
- Automated batch processing scripts
- Custom web interfaces
- Pipeline automation

**Base URLs:**
- `hybrid/`: http://localhost:8000
- `interactive/`: http://localhost:8001
- `semantic/`: http://localhost:8002
- `panoptic/`: http://localhost:8003

---

## 🔧 Common Patterns

### Authentication
Currently, no authentication required. Future versions may add API keys.

### Content Type
- Upload: `multipart/form-data`
- Response: `application/json`

### Error Handling
All endpoints return standard HTTP status codes:
- `200`: Success
- `400`: Bad request (invalid input)
- `413`: File too large
- `500`: Server error

Error response format:
```json
{
  "error": "Error message",
  "detail": "Additional details"
}
```

---

## 🎨 hybrid/ API

### Upload and Generate Points

**Endpoint:** `POST /upload`

**Description:** Upload images and generate automatic segmentation points

**Request:**
```bash
curl -X POST http://localhost:8000/upload \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

**Response:**
```json
{
  "session_id": "abc123",
  "images": [
    {
      "filename": "image1.jpg",
      "points": [[100, 200], [150, 250]],
      "preview": "data:image/jpeg;base64,..."
    }
  ]
}
```

---

### Generate Segmentation Mask

**Endpoint:** `POST /segment`

**Description:** Generate SAM segmentation mask from points

**Request:**
```bash
curl -X POST http://localhost:8000/segment \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "abc123",
    "image_index": 0,
    "include_points": [[100,200], [150,250]],
    "exclude_points": [[50,50]],
    "model_type": "vit_l"
  }'
```

**Parameters:**
- `session_id` (string, required): Session identifier
- `image_index` (int, required): Index of image in session
- `include_points` (array, required): List of [x,y] coordinates to include
- `exclude_points` (array, optional): List of [x,y] coordinates to exclude
- `model_type` (string, optional): "vit_b", "vit_l", or "vit_h" (default: "vit_l")

**Response:**
```json
{
  "mask": "data:image/png;base64,...",
  "overlay": "data:image/png;base64,...",
  "transparent": "data:image/png;base64,...",
  "statistics": {
    "coverage_percent": 42.5,
    "total_pixels": 1048576,
    "plant_pixels": 445645
  }
}
```

---

### Download Results

**Endpoint:** `GET /download/{session_id}`

**Description:** Download all results as ZIP

**Request:**
```bash
curl http://localhost:8000/download/abc123 -o results.zip
```

---

## ✏️ interactive/ API

### Segment with Points

**Endpoint:** `POST /segment`

**Description:** Upload image and segment with include/exclude points

**Request:**
```bash
curl -X POST http://localhost:8001/segment \
  -F "file=@plant.jpg" \
  -F "include_points=[[512,384],[600,400]]" \
  -F "exclude_points=[[200,200]]" \
  -F "model_type=vit_l"
```

**Parameters:**
- `file` (file, required): Image file to segment
- `include_points` (JSON array, required): Points to include in mask
- `exclude_points` (JSON array, optional): Points to exclude
- `model_type` (string, optional): "vit_b", "vit_l", or "vit_h"

**Response:**
```json
{
  "session_id": "xyz789",
  "mask": "data:image/png;base64,...",
  "masked_image": "data:image/png;base64,...",
  "transparent": "data:image/png;base64,...",
  "download_url": "/download/xyz789"
}
```

---

### Get Available Models

**Endpoint:** `GET /models`

**Description:** List available SAM models

**Request:**
```bash
curl http://localhost:8001/models
```

**Response:**
```json
{
  "models": [
    {
      "name": "vit_b",
      "size": "91M parameters",
      "speed": "fast"
    },
    {
      "name": "vit_l",
      "size": "308M parameters",
      "speed": "medium"
    },
    {
      "name": "vit_h",
      "size": "636M parameters",
      "speed": "slow"
    }
  ]
}
```

---

## ⚡ semantic/ API

### Batch Segmentation

**Endpoint:** `POST /segment_batch`

**Description:** Upload and segment multiple images automatically

**Request:**
```bash
curl -X POST http://localhost:8002/segment_batch \
  -F "files=@img1.jpg" \
  -F "files=@img2.jpg" \
  -F "files=@img3.jpg"
```

**Response:**
```json
{
  "session_id": "def456",
  "results": [
    {
      "filename": "img1.jpg",
      "grid": "data:image/png;base64,...",
      "statistics": {
        "coverage_percent": 38.2,
        "classes_detected": ["tree", "grass"],
        "class_breakdown": {
          "tree": {"pixels": 12345, "percent": 15.2},
          "grass": {"pixels": 18765, "percent": 23.0}
        }
      }
    }
  ],
  "download_url": "/download/def456"
}
```

---

### Single Image Segmentation

**Endpoint:** `POST /segment`

**Description:** Segment single image

**Request:**
```bash
curl -X POST http://localhost:8002/segment \
  -F "file=@plant.jpg"
```

**Response:**
```json
{
  "session_id": "ghi789",
  "mask": "data:image/png;base64,...",
  "overlay": "data:image/png;base64,...",
  "statistics": {...}
}
```

---

## 🔬 panoptic/ API

### Segmentation with Mode Selection

**Endpoint:** `POST /segment`

**Description:** Segment image with specified mode and model

**Request:**
```bash
curl -X POST http://localhost:8003/segment \
  -F "file=@plant.jpg" \
  -F "mode=panoptic" \
  -F "model_size=large" \
  -F "threshold=0.5"
```

**Parameters:**
- `file` (file, required): Image to segment
- `mode` (string, required): "semantic", "instance", or "panoptic"
- `model_size` (string, optional): "base" or "large" (default: "base")
- `threshold` (float, optional): Confidence threshold 0.0-1.0 (default: 0.5)

**Response:**
```json
{
  "session_id": "jkl012",
  "mode": "panoptic",
  "mask": "data:image/png;base64,...",
  "overlay": "data:image/png;base64,...",
  "statistics": {
    "coverage_percent": 45.8,
    "instances_detected": 5,
    "instances": [
      {
        "id": 1,
        "class": "plant",
        "area_pixels": 8765,
        "confidence": 0.89,
        "bbox": [100, 150, 300, 400]
      }
    ],
    "semantic_classes": {
      "tree": 12345,
      "grass": 23456
    }
  }
}
```

---

### Batch Processing with Settings

**Endpoint:** `POST /segment_batch`

**Description:** Process multiple images with same settings

**Request:**
```bash
curl -X POST http://localhost:8003/segment_batch \
  -F "files=@img1.jpg" \
  -F "files=@img2.jpg" \
  -F "mode=instance" \
  -F "model_size=large" \
  -F "threshold=0.6"
```

---

## 🔄 Common Endpoints (All Tools)

### Health Check

**Endpoint:** `GET /health`

**Description:** Check if server is running

**Request:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "tool": "hybrid",
  "version": "1.0.0",
  "models_loaded": ["segformer", "sam_vit_l"]
}
```

---

### Server Info

**Endpoint:** `GET /info`

**Description:** Get server configuration and capabilities

**Request:**
```bash
curl http://localhost:8000/info
```

**Response:**
```json
{
  "tool": "hybrid",
  "version": "1.0.0",
  "max_upload_size": 52428800,
  "supported_formats": ["jpg", "jpeg", "png", "bmp"],
  "available_models": ["vit_b", "vit_l", "vit_h"],
  "gpu_available": true,
  "gpu_name": "NVIDIA RTX 3090"
}
```

---

## 📊 Response Formats

### Image Data

Images returned as base64-encoded data URLs:
```
data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...
```

**To decode in Python:**
```python
import base64
from PIL import Image
from io import BytesIO

# Extract base64 data
image_data = response_json['mask'].split(',')[1]

# Decode
image_bytes = base64.b64decode(image_data)
image = Image.open(BytesIO(image_bytes))
image.save('output.png')
```

---

### Statistics Format

**Semantic segmentation:**
```json
{
  "coverage_percent": 42.5,
  "total_pixels": 1048576,
  "plant_pixels": 445645,
  "class_breakdown": {
    "tree": {"pixels": 12345, "percent": 15.2},
    "grass": {"pixels": 18765, "percent": 23.0}
  }
}
```

**Instance segmentation:**
```json
{
  "num_instances": 5,
  "instances": [
    {
      "id": 1,
      "class": "plant",
      "area_pixels": 8765,
      "confidence": 0.89,
      "bbox": [x, y, width, height]
    }
  ]
}
```

---

## 🐍 Python Client Examples

### hybrid/ Client

```python
import requests

# Upload images
files = [
    ('files', open('img1.jpg', 'rb')),
    ('files', open('img2.jpg', 'rb'))
]
response = requests.post('http://localhost:8000/upload', files=files)
data = response.json()
session_id = data['session_id']

# Generate mask
payload = {
    'session_id': session_id,
    'image_index': 0,
    'include_points': [[100, 200], [150, 250]],
    'exclude_points': [[50, 50]],
    'model_type': 'vit_l'
}
response = requests.post('http://localhost:8000/segment', json=payload)
result = response.json()

# Download ZIP
response = requests.get(f'http://localhost:8000/download/{session_id}')
with open('results.zip', 'wb') as f:
    f.write(response.content)
```

---

### interactive/ Client

```python
import requests

# Segment image
with open('plant.jpg', 'rb') as f:
    files = {'file': f}
    data = {
        'include_points': '[[512,384],[600,400]]',
        'exclude_points': '[[200,200]]',
        'model_type': 'vit_l'
    }
    response = requests.post('http://localhost:8001/segment', 
                            files=files, data=data)

result = response.json()
mask_base64 = result['mask']

# Save mask
import base64
from PIL import Image
from io import BytesIO

image_data = mask_base64.split(',')[1]
image_bytes = base64.b64decode(image_data)
image = Image.open(BytesIO(image_bytes))
image.save('mask.png')
```

---

### semantic/ Client

```python
import requests

# Batch processing
files = [
    ('files', open('img1.jpg', 'rb')),
    ('files', open('img2.jpg', 'rb')),
    ('files', open('img3.jpg', 'rb'))
]
response = requests.post('http://localhost:8002/segment_batch', files=files)
results = response.json()

# Process results
for result in results['results']:
    print(f"{result['filename']}: {result['statistics']['coverage_percent']}%")
```

---

### panoptic/ Client

```python
import requests

# Instance segmentation
with open('plant.jpg', 'rb') as f:
    files = {'file': f}
    data = {
        'mode': 'instance',
        'model_size': 'large',
        'threshold': '0.6'
    }
    response = requests.post('http://localhost:8003/segment',
                            files=files, data=data)

result = response.json()
print(f"Detected {result['statistics']['instances_detected']} plants")

for instance in result['statistics']['instances']:
    print(f"Plant {instance['id']}: {instance['area_pixels']} pixels, "
          f"confidence: {instance['confidence']:.2f}")
```

---

## 🔒 Rate Limiting

Currently no rate limiting. Future versions may implement:
- Max requests per minute per IP
- Max concurrent uploads
- Max file size per request

---

## 📞 Support

**API Issues:**
- Check [Troubleshooting Guide](troubleshooting.md)
- Open [GitHub Issue](https://github.com/vahidshokrani415/AgriSegment/issues)
- Email: mehran.tarifhokmabadi@univr.it

**Feature Requests:**
- API improvements and additions welcome
- Open feature request on GitHub

---

<div align="center">

**[← Back to Main README](../README.md)** | **[Best Practices →](best-practices.md)**

</div>