SAM_MODEL_SIZE = "huge"  # Options: "base", "large", "huge"

from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from contextlib import asynccontextmanager
import uvicorn
import numpy as np
import cv2
import torch
from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import io
import base64
import os
import json
import uuid
from pathlib import Path
from typing import List, Optional
import logging
import zipfile
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SAM Model Configuration
SAM_MODEL_CONFIG = {
    "base": {
        "model_type": "vit_b",
        "checkpoint": "sam_vit_b_01ec64.pth",
        "size": "91M parameters"
    },
    "large": {
        "model_type": "vit_l", 
        "checkpoint": "sam_vit_l_0b3195.pth",
        "size": "308M parameters"
    },
    "huge": {
        "model_type": "vit_h",
        "checkpoint": "sam_vit_h_4b8939.pth", 
        "size": "636M parameters"
    }
}

# Get current model configuration
if SAM_MODEL_SIZE not in SAM_MODEL_CONFIG:
    raise ValueError(f"Invalid SAM_MODEL_SIZE '{SAM_MODEL_SIZE}'. Options: {list(SAM_MODEL_CONFIG.keys())}")

CURRENT_MODEL = SAM_MODEL_CONFIG[SAM_MODEL_SIZE]
MODEL_TYPE = CURRENT_MODEL["model_type"]
CHECKPOINT_PATH = CURRENT_MODEL["checkpoint"]

print(f"🎯 SAM Configuration:")
print(f"   Model Size: {SAM_MODEL_SIZE.upper()}")
print(f"   Model Type: {MODEL_TYPE}")
print(f"   Checkpoint: {CHECKPOINT_PATH}")
print(f"   Parameters: {CURRENT_MODEL['size']}")
print(f"=" * 50)

# Global variables for SAM
sam_predictor = None
current_sessions = {}

def initialize_sam():
    """Initialize SAM model"""
    global sam_predictor
    try:
        if not os.path.exists(CHECKPOINT_PATH):
            logger.error(f"SAM checkpoint not found: {CHECKPOINT_PATH}")
            logger.info(f"Please download the {SAM_MODEL_SIZE} model:")
            
            download_urls = {
                "base": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                "large": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", 
                "huge": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            }
            
            logger.info(f"   wget {download_urls[SAM_MODEL_SIZE]}")
            return False
            
        logger.info(f"Loading SAM {SAM_MODEL_SIZE.upper()} model ({CURRENT_MODEL['size']})...")
        sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
        sam_predictor = SamPredictor(sam)
        logger.info(f"✅ SAM {SAM_MODEL_SIZE.upper()} model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to load SAM model: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    logger.info("Starting up SAM application...")
    if not initialize_sam():
        logger.error("Failed to initialize SAM model!")
    yield
    # Shutdown
    logger.info("Shutting down SAM application...")

app = FastAPI(
    title="SAM Interactive Segmentation", 
    version="1.0.0",
    lifespan=lifespan
)

# Create directories
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)
os.makedirs("static/images", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/output", StaticFiles(directory="output"), name="output")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Global variables for SAM
sam_predictor = None
current_sessions = {}

class SAMSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.current_image = None
        self.image_path = None
        self.original_filename = None
        self.points = []
        self.current_mask = None
        
    def add_point(self, x: float, y: float, label: int):
        """Add a point with label (1 for include, 0 for exclude)"""
        self.points.append([x, y, label])
        
    def clear_points(self):
        """Clear all points"""
        self.points = []
        
    def get_points_for_sam(self):
        """Get points in format expected by SAM"""
        if not self.points:
            return None, None
        points = np.array([[p[0], p[1]] for p in self.points])
        labels = np.array([p[2] for p in self.points])
        return points, labels
    
    def get_base_filename(self):
        """Get base filename without extension"""
        if self.original_filename:
            return Path(self.original_filename).stem
        return "image"

def get_session(session_id: str) -> SAMSession:
    """Get or create session"""
    if session_id not in current_sessions:
        current_sessions[session_id] = SAMSession(session_id)
    return current_sessions[session_id]

def image_to_base64(image_array: np.ndarray) -> str:
    """Convert numpy image to base64 string"""
    # Convert RGB to BGR for cv2
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', image_bgr)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

def visualize_points(image: np.ndarray, points: List[List[float]]) -> np.ndarray:
    """Visualize points on image with numbers"""
    result = image.copy()
    for i, point in enumerate(points):
        x, y, label = int(point[0]), int(point[1]), point[2]
        color = (34, 197, 94) if label == 1 else (239, 68, 68)  # Green for include, red for exclude
        
        # Draw outer circle
        cv2.circle(result, (x, y), 15, color, -1)
        # Draw white border
        cv2.circle(result, (x, y), 15, (255, 255, 255), 3)
        
        # Draw number inside circle
        number_text = str(i + 1)
        text_size = cv2.getTextSize(number_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x - text_size[0] // 2
        text_y = y + text_size[1] // 2
        cv2.putText(result, number_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return result

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_image(file: UploadFile = File(...), session_id: str = Form(...)):
    """Upload and process image"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and save uploaded file
        contents = await file.read()
        filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = f"uploads/{filename}"
        
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Load image with PIL and convert to numpy
        pil_image = Image.open(file_path)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        image_array = np.array(pil_image)
        
        # Get session and store image with original filename
        session = get_session(session_id)
        session.current_image = image_array
        session.image_path = file_path
        session.original_filename = file.filename  # Store original filename
        session.clear_points()  # Clear previous points
        
        # Set image for SAM predictor
        if sam_predictor:
            sam_predictor.set_image(image_array)
        
        # Return image as base64 for display
        img_base64 = image_to_base64(image_array)
        
        return JSONResponse({
            "success": True,
            "image": img_base64,
            "message": f"Image '{file.filename}' uploaded successfully!",
            "filename": filename
        })
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_point")
async def add_point(
    x: float = Form(...),
    y: float = Form(...),
    label: int = Form(...),
    session_id: str = Form(...)
):
    """Add a point to the current session"""
    try:
        session = get_session(session_id)
        
        if session.current_image is None:
            raise HTTPException(status_code=400, detail="No image loaded")
        
        # Add point to session
        session.add_point(x, y, label)
        
        # Visualize points on image
        result_image = visualize_points(session.current_image, session.points)
        img_base64 = image_to_base64(result_image)
        
        point_type = "INCLUDE" if label == 1 else "EXCLUDE"
        
        return JSONResponse({
            "success": True,
            "image": img_base64,
            "message": f"Added {point_type} point #{len(session.points)} at ({x:.0f}, {y:.0f})",
            "total_points": len(session.points)
        })
        
    except Exception as e:
        logger.error(f"Add point error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_mask")
async def generate_mask(session_id: str = Form(...)):
    """Generate segmentation mask"""
    try:
        session = get_session(session_id)
        
        if session.current_image is None:
            raise HTTPException(status_code=400, detail="No image loaded")
        
        if len(session.points) == 0:
            raise HTTPException(status_code=400, detail="No points added")
        
        if sam_predictor is None:
            raise HTTPException(status_code=500, detail="SAM model not loaded")
        
        # Get points for SAM
        input_points, input_labels = session.get_points_for_sam()
        
        # Generate masks
        masks, scores, logits = sam_predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
        
        # Use best mask
        best_mask_idx = np.argmax(scores)
        mask = masks[best_mask_idx]
        score = scores[best_mask_idx]
        
        # Store mask in session
        session.current_mask = mask
        
        # Create visualization
        result = session.current_image.copy()
        
        # Add colored mask overlay
        overlay = np.zeros_like(session.current_image)
        overlay[mask] = [255, 0, 0]  # Red overlay
        result = cv2.addWeighted(result, 0.6, overlay, 0.4, 0)
        
        # Add points
        result = visualize_points(result, session.points)
        
        img_base64 = image_to_base64(result)
        
        return JSONResponse({
            "success": True,
            "image": img_base64,
            "message": f"Generated mask with confidence: {score:.3f}",
            "confidence": float(score)
        })
        
    except Exception as e:
        logger.error(f"Generate mask error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save_mask")
async def save_mask(session_id: str = Form(...)):
    """Save the current mask and masked image"""
    try:
        session = get_session(session_id)
        
        if session.current_mask is None:
            raise HTTPException(status_code=400, detail="No mask generated")
        
        # Get base filename from original upload
        base_name = session.get_base_filename()
        
        # Create unique directory for this save operation
        timestamp = uuid.uuid4().hex[:8]
        save_dir = f"output/{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Save binary mask
        mask_filename = f"mask_{base_name}.png"
        mask_path = f"{save_dir}/{mask_filename}"
        cv2.imwrite(mask_path, (session.current_mask * 255).astype(np.uint8))
        
        # Save masked image (original with background removed)
        masked_image = session.current_image.copy()
        masked_image[~session.current_mask] = [0, 0, 0]
        masked_filename = f"masked_{base_name}.png"
        masked_path = f"{save_dir}/{masked_filename}"
        cv2.imwrite(masked_path, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
        
        # Save segmented object (transparent background)
        segmented = np.dstack([session.current_image, session.current_mask * 255])
        segmented_filename = f"segmented_{base_name}.png"
        segmented_path = f"{save_dir}/{segmented_filename}"
        Image.fromarray(segmented.astype(np.uint8)).save(segmented_path)
        
        # Create thumbnail versions for preview (200x200 max)
        def create_thumbnail(image_array, max_size=200):
            try:
                if image_array is None or image_array.size == 0:
                    return image_array
                    
                h, w = image_array.shape[:2]
                if h == 0 or w == 0:
                    return image_array
                    
                if max(h, w) > max_size:
                    scale = max_size / max(h, w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    
                    # Ensure new dimensions are valid
                    if new_h <= 0 or new_w <= 0:
                        return image_array
                    
                    if len(image_array.shape) == 3:
                        thumbnail = cv2.resize(image_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    else:
                        thumbnail = cv2.resize(image_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    return thumbnail
                return image_array
            except Exception as e:
                logger.error(f"Thumbnail creation error: {e}")
                return image_array
        
        # Create thumbnails for preview
        try:
            mask_thumb = create_thumbnail((session.current_mask * 255).astype(np.uint8))
            masked_thumb = create_thumbnail(masked_image)
            segmented_thumb = create_thumbnail(session.current_image)
            
            # Handle different thumbnail formats safely
            if len(mask_thumb.shape) == 2:
                mask_preview = image_to_base64(cv2.cvtColor(mask_thumb, cv2.COLOR_GRAY2RGB))
            else:
                mask_preview = image_to_base64(mask_thumb)
                
            masked_preview = image_to_base64(masked_thumb)
            
            # Create segmented thumbnail with alpha channel
            if segmented_thumb.shape[:2] == mask_thumb.shape[:2]:
                segmented_alpha = create_thumbnail(session.current_mask * 255)
                segmented_with_alpha = np.dstack([segmented_thumb, segmented_alpha.astype(np.uint8)])
            else:
                segmented_with_alpha = np.dstack([segmented_thumb, np.ones(segmented_thumb.shape[:2], dtype=np.uint8) * 255])
            
            # Convert to base64
            _, buffer = cv2.imencode('.png', segmented_with_alpha)
            segmented_preview = f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"
            
        except Exception as e:
            logger.error(f"Preview creation error: {e}")
            # Fallback to simple previews
            mask_preview = image_to_base64(cv2.cvtColor((session.current_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB))
            masked_preview = image_to_base64(masked_image)
            segmented_preview = image_to_base64(session.current_image)
        
        # Create zip file with all results
        zip_filename = f"sam_results_{base_name}_{timestamp}.zip"
        zip_path = f"output/{zip_filename}"
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(mask_path, mask_filename)
            zipf.write(masked_path, masked_filename)
            zipf.write(segmented_path, segmented_filename)
        
        return JSONResponse({
            "success": True,
            "message": "Mask saved successfully!",
            "files": {
                "mask": f"/output/{timestamp}/{mask_filename}",
                "masked": f"/output/{timestamp}/{masked_filename}",
                "segmented": f"/output/{timestamp}/{segmented_filename}",
                "zip": f"/output/{zip_filename}"
            },
            "previews": {
                "mask": mask_preview,
                "masked": masked_preview,
                "segmented": segmented_preview
            },
            "filenames": {
                "mask": mask_filename,
                "masked": masked_filename,
                "segmented": segmented_filename,
                "zip": zip_filename
            }
        })
        
    except Exception as e:
        logger.error(f"Save mask error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear_points")
async def clear_points(session_id: str = Form(...)):
    """Clear all points"""
    try:
        session = get_session(session_id)
        
        if session.current_image is None:
            raise HTTPException(status_code=400, detail="No image loaded")
        
        session.clear_points()
        
        # Return original image
        img_base64 = image_to_base64(session.current_image)
        
        return JSONResponse({
            "success": True,
            "image": img_base64,
            "message": "Cleared all points"
        })
        
    except Exception as e:
        logger.error(f"Clear points error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "sam_model_size": SAM_MODEL_SIZE,
        "sam_model_type": MODEL_TYPE,
        "checkpoint_file": CHECKPOINT_PATH,
        "model_parameters": CURRENT_MODEL['size'],
        "sam_loaded": sam_predictor is not None,
        "active_sessions": len(current_sessions)
    }

if __name__ == "__main__":
    print("🎯 Starting SAM Interactive Segmentation Server")
    print("=" * 50)
    print(f"🧠 AI Model: SAM {SAM_MODEL_SIZE.upper()} ({CURRENT_MODEL['size']})")
    print(f"📁 Checkpoint: {CHECKPOINT_PATH}")
    print("=" * 50)
    print("🚀 Server will start on: http://0.0.0.0:8001")
    print("🌐 Local access: http://localhost:8001")
    print("📋 API docs: http://localhost:8001/docs")
    print("🔍 Health check: http://localhost:8001/health")
    print("=" * 50)
    print()
    
    # Check if checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        print("❌ Model checkpoint not found!")
        print(f"📥 Download {SAM_MODEL_SIZE} model:")
        
        download_commands = {
            "base": "wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "large": "wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "huge": "wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        }
        
        print(f"   {download_commands[SAM_MODEL_SIZE]}")
        print()
        print("💡 Available model sizes:")
        print("   • base  - 91M parameters  (fastest, good quality)")
        print("   • large - 308M parameters (balanced speed/quality)")  
        print("   • huge  - 636M parameters (best quality, slower)")
        print()
        print("🔧 To change model size, edit SAM_MODEL_SIZE in line 4 of main.py")
        print("=" * 50)
    else:
        print("✅ Model checkpoint found!")
        print()
    
    print("💡 Alternative ways to run:")
    print("   uvicorn main:app --host 0.0.0.0 --port 8001 --reload")
    print("   uvicorn main:app --host 0.0.0.0 --port 8001 --workers 4")
    print()
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
