from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from segment_anything import SamPredictor, sam_model_registry
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import io
import base64
import os
import uuid
from pathlib import Path
from scipy import ndimage
import logging
from typing import List
import zipfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INCLUDE_CLASSES = {
    'tree': 4, 'grass': 9, 'plant': 17, 'field': 29, 'flower': 66, 'palm': 72
}

EXCLUDE_CLASSES = {
    'building': 1, 'sky': 2, 'road': 6, 'person': 12, 'car': 20, 'house': 25
}

sam_predictor = None
segformer_model = None
segformer_processor = None
device = None
current_sessions = {}

def initialize_models():
    global sam_predictor, segformer_model, segformer_processor, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        sam_checkpoint = "sam_vit_b_01ec64.pth"
        if os.path.exists(sam_checkpoint):
            sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
            sam_predictor = SamPredictor(sam)
            logger.info("✅ SAM model loaded")
        else:
            logger.error(f"SAM checkpoint not found: {sam_checkpoint}")
            return False
        
        model_name = "nvidia/segformer-b4-finetuned-ade-512-512"
        segformer_model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        segformer_processor = SegformerImageProcessor.from_pretrained(model_name)
        segformer_model.to(device)
        segformer_model.eval()
        logger.info("✅ Segformer model loaded")
        
        return True
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not initialize_models():
        logger.error("Failed to initialize models!")
    yield

app = FastAPI(title="Multi-Image Plant Segmentation", lifespan=lifespan)

def ensure_directories():
    directories = ["templates", "uploads", "output"]
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            test_file = os.path.join(directory, ".test_write")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            logger.error(f"Failed to create or write to directory {directory}: {e}")
            raise

ensure_directories()
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/output", StaticFiles(directory="output"), name="output")

templates = Jinja2Templates(directory="templates")

class ImageData:
    def __init__(self, image_id: str, filename: str, filepath: str):
        self.image_id = image_id
        self.filename = filename
        self.filepath = filepath
        self.original_image = None
        self.segformer_result = None
        self.auto_points = []
        self.manual_points = []
        self.current_mask = None
        self.status = "uploaded"

class MultiSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.images = {}
        
    def add_image(self, image_data: ImageData):
        self.images[image_data.image_id] = image_data
        
    def get_image(self, image_id: str) -> ImageData:
        return self.images.get(image_id)

def get_session(session_id: str) -> MultiSession:
    if session_id not in current_sessions:
        current_sessions[session_id] = MultiSession(session_id)
    return current_sessions[session_id]

def get_quality_points(mask, max_points=4):
    if mask.sum() == 0:
        return []
    
    region_size = mask.sum()
    
    if region_size < 1000:
        num_points = 1
        min_distance = 30
    elif region_size < 4000:
        num_points = 2
        min_distance = 50
    else:
        num_points = min(max_points, max(2, region_size // 2000))
        min_distance = 80
    
    erosion_size = max(2, min(5, int(np.sqrt(region_size) / 30)))
    kernel = np.ones((erosion_size, erosion_size), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    
    if eroded.sum() < region_size * 0.15:
        eroded = mask.astype(np.uint8)
    
    dist = ndimage.distance_transform_edt(eroded)
    
    points = []
    for i in range(num_points):
        if dist.max() == 0:
            break
            
        y, x = np.unravel_index(dist.argmax(), dist.shape)
        points.append([x, y])
        
        suppress_radius = max(min_distance, int(min_distance * (1.5 + i * 0.3)))
        
        y_min = max(0, y - suppress_radius)
        y_max = min(dist.shape[0], y + suppress_radius)
        x_min = max(0, x - suppress_radius)
        x_max = min(dist.shape[1], x + suppress_radius)
        dist[y_min:y_max, x_min:x_max] = 0
    
    return points

def segment_with_segformer(image_array):
    try:
        inputs = segformer_processor(images=Image.fromarray(image_array), return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)
        
        with torch.no_grad():
            outputs = segformer_model(pixel_values)
            logits = outputs.logits
            
            upsampled = F.interpolate(
                logits, size=image_array.shape[:2], 
                mode="bilinear", align_corners=False
            )
            
            predicted = upsampled.argmax(dim=1).squeeze().cpu().numpy()
            
        return predicted
    except Exception as e:
        logger.error(f"Segformer error: {e}")
        return None

def generate_auto_points(predicted_labels, image_shape):
    points = []
    
    for class_name, class_id in INCLUDE_CLASSES.items():
        mask = (predicted_labels == class_id)
        region_size = mask.sum()
        
        if region_size > 500:
            quality_points = get_quality_points(mask, max_points=3)
            for point in quality_points:
                points.append([point[0], point[1], 1])
    
    for class_name, class_id in EXCLUDE_CLASSES.items():
        mask = (predicted_labels == class_id)
        region_size = mask.sum()
        
        if region_size > 2000:
            quality_points = get_quality_points(mask, max_points=1)
            for point in quality_points:
                points.append([point[0], point[1], 0])
    
    return points

def image_to_base64(image_array):
    try:
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.png', image_bgr)
        return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        raise

def visualize_points(image, points):
    result = image.copy()
    for i, (x, y, label) in enumerate(points):
        color = (0, 255, 0) if label == 1 else (255, 0, 0)
        cv2.circle(result, (int(x), int(y)), 8, color, -1)
        cv2.circle(result, (int(x), int(y)), 8, (255, 255, 255), 2)
        cv2.putText(result, str(i+1), (int(x)-5, int(y)+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return result

def ensure_points_generated(image_data):
    if not image_data.auto_points and not image_data.manual_points:
        if image_data.segformer_result is None:
            predicted = segment_with_segformer(image_data.original_image)
            if predicted is not None:
                image_data.segformer_result = predicted
        
        if image_data.segformer_result is not None:
            auto_points = generate_auto_points(image_data.segformer_result, image_data.original_image.shape)
            image_data.auto_points = auto_points
            
            if image_data.status == "uploaded":
                image_data.status = "points_generated"
    
    return len(image_data.auto_points) + len(image_data.manual_points) > 0

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload_multiple")
async def upload_multiple_images(files: List[UploadFile] = File(...), session_id: str = Form(...)):
    try:
        session = get_session(session_id)
        uploaded_images = []
        
        for file in files:
            if not file.content_type.startswith('image/'):
                continue
                
            image_id = str(uuid.uuid4())
            filename = f"{image_id}_{file.filename}"
            file_path = f"uploads/{filename}"
            
            contents = await file.read()
            with open(file_path, "wb") as f:
                f.write(contents)
            
            pil_image = Image.open(file_path).convert('RGB')
            image_array = np.array(pil_image)
            
            image_data = ImageData(image_id, file.filename, file_path)
            image_data.original_image = image_array
            session.add_image(image_data)
            
            uploaded_images.append({
                "image_id": image_id,
                "filename": file.filename,
                "preview": image_to_base64(image_array),
                "status": "uploaded"
            })
        
        return JSONResponse({
            "success": True,
            "message": f"Uploaded {len(uploaded_images)} images",
            "images": uploaded_images
        })
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_all_points")
async def generate_all_points(session_id: str = Form(...)):
    try:
        session = get_session(session_id)
        results = []
        
        for image_id, image_data in session.images.items():
            try:
                predicted = segment_with_segformer(image_data.original_image)
                if predicted is None:
                    continue
                
                auto_points = generate_auto_points(predicted, image_data.original_image.shape)
                image_data.auto_points = auto_points
                image_data.segformer_result = predicted
                image_data.status = "points_generated"
                
                result_image = visualize_points(image_data.original_image, auto_points)
                
                results.append({
                    "image_id": image_id,
                    "points_count": len(auto_points),
                    "preview": image_to_base64(result_image),
                    "status": "points_generated"
                })
                
            except Exception as e:
                logger.error(f"Error processing image {image_id}: {e}")
                continue
        
        return JSONResponse({
            "success": True,
            "message": f"Generated points for {len(results)} images",
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Points generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_all_masks")
async def generate_all_masks(session_id: str = Form(...)):
    try:
        session = get_session(session_id)
        results = []
        
        for image_id, image_data in session.images.items():
            if image_data.status == "mask_generated" or image_data.status == "saved":
                continue
                
            try:
                if not ensure_points_generated(image_data):
                    logger.warning(f"Could not generate points for image {image_id}")
                    continue
                
                all_points = image_data.auto_points + image_data.manual_points
                
                sam_predictor.set_image(image_data.original_image)
                
                input_points = np.array([[p[0], p[1]] for p in all_points])
                input_labels = np.array([p[2] for p in all_points])
                
                masks, scores, _ = sam_predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output=True
                )
                
                best_mask = masks[np.argmax(scores)]
                image_data.current_mask = best_mask
                image_data.status = "mask_generated"
                
                result = image_data.original_image.copy()
                overlay = np.zeros_like(image_data.original_image)
                overlay[best_mask] = [0, 255, 0]
                result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
                result = visualize_points(result, all_points)
                
                results.append({
                    "image_id": image_id,
                    "confidence": float(scores.max()),
                    "preview": image_to_base64(result),
                    "status": "mask_generated",
                    "points_count": len(all_points)
                })
                
            except Exception as e:
                logger.error(f"Error generating mask for {image_id}: {e}")
                continue
        
        return JSONResponse({
            "success": True,
            "message": f"Generated masks for {len(results)} images",
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Mask generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/edit_image_points")
async def edit_image_points(
    image_id: str = Form(...),
    x: float = Form(...),
    y: float = Form(...),
    label: int = Form(...),
    session_id: str = Form(...)
):
    try:
        session = get_session(session_id)
        image_data = session.get_image(image_id)
        
        if not image_data:
            raise HTTPException(status_code=404, detail="Image not found")
        
        image_data.manual_points.append([x, y, label])
        
        all_points = image_data.auto_points + image_data.manual_points
        result_image = visualize_points(image_data.original_image, all_points)
        
        return JSONResponse({
            "success": True,
            "preview": image_to_base64(result_image),
            "points_count": len(all_points)
        })
        
    except Exception as e:
        logger.error(f"Error adding point: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/remove_image")
async def remove_image(image_id: str = Form(...), session_id: str = Form(...)):
    try:
        session = get_session(session_id)
        image_data = session.get_image(image_id)
        
        if not image_data:
            raise HTTPException(status_code=404, detail="Image not found")
        
        if os.path.exists(image_data.filepath):
            os.remove(image_data.filepath)
        
        del session.images[image_id]
        
        return JSONResponse({
            "success": True,
            "message": f"Image '{image_data.filename}' removed",
            "remaining_count": len(session.images)
        })
        
    except Exception as e:
        logger.error(f"Error removing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset_image_points")
async def reset_image_points(image_id: str = Form(...), session_id: str = Form(...)):
    try:
        session = get_session(session_id)
        image_data = session.get_image(image_id)
        
        if not image_data:
            raise HTTPException(status_code=404, detail="Image not found")
        
        image_data.auto_points = []
        image_data.manual_points = []
        image_data.current_mask = None
        image_data.status = "uploaded"
        
        return JSONResponse({
            "success": True,
            "preview": image_to_base64(image_data.original_image),
            "message": "All points cleared"
        })
        
    except Exception as e:
        logger.error(f"Error resetting points: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/regenerate_single_mask")
async def regenerate_single_mask(image_id: str = Form(...), session_id: str = Form(...)):
    try:
        session = get_session(session_id)
        image_data = session.get_image(image_id)
        
        if not image_data:
            raise HTTPException(status_code=404, detail="Image not found")
        
        if not ensure_points_generated(image_data):
            raise HTTPException(status_code=400, detail="Could not generate points for this image")
            
        all_points = image_data.auto_points + image_data.manual_points
        
        sam_predictor.set_image(image_data.original_image)
        
        input_points = np.array([[p[0], p[1]] for p in all_points])
        input_labels = np.array([p[2] for p in all_points])
        
        masks, scores, _ = sam_predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True
        )
        
        best_mask = masks[np.argmax(scores)]
        image_data.current_mask = best_mask
        image_data.status = "mask_generated"
        
        result = image_data.original_image.copy()
        overlay = np.zeros_like(image_data.original_image)
        overlay[best_mask] = [0, 255, 0]
        result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
        result = visualize_points(result, all_points)
        
        return JSONResponse({
            "success": True,
            "preview": image_to_base64(result),
            "confidence": float(scores.max())
        })
        
    except Exception as e:
        logger.error(f"Error regenerating mask: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save_all_results")
async def save_all_results(session_id: str = Form(...)):
    try:
        session = get_session(session_id)
        
        images_with_masks = [img for img in session.images.values() if img.current_mask is not None]
        
        if not images_with_masks:
            raise HTTPException(status_code=400, detail="No masks to save")
        
        batch_id = str(uuid.uuid4())[:8]
        batch_dir = f"output/batch_{batch_id}"
        
        try:
            os.makedirs(batch_dir, exist_ok=True)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create output directory: {e}")
        
        saved_files = []
        
        for image_id, image_data in session.images.items():
            if image_data.current_mask is None:
                continue
                
            base_name = Path(image_data.filename).stem
            
            try:
                original_path = f"{batch_dir}/original_{base_name}.png"
                success = cv2.imwrite(original_path, cv2.cvtColor(image_data.original_image, cv2.COLOR_RGB2BGR))
                if not success:
                    raise Exception(f"Failed to write original image: {original_path}")
                
                mask_path = f"{batch_dir}/mask_{base_name}.png"
                mask_image = (image_data.current_mask * 255).astype(np.uint8)
                success = cv2.imwrite(mask_path, mask_image)
                if not success:
                    raise Exception(f"Failed to write mask image: {mask_path}")
                
                segmented_path = f"{batch_dir}/segmented_{base_name}.png"
                segmented = image_data.original_image.copy()
                segmented[~image_data.current_mask] = [255, 255, 255]
                success = cv2.imwrite(segmented_path, cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))
                if not success:
                    raise Exception(f"Failed to write segmented image: {segmented_path}")
                
                image_data.status = "saved"
                saved_files.append({
                    "filename": image_data.filename,
                    "original": original_path,
                    "mask": mask_path,
                    "segmented": segmented_path
                })
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to save image files: {e}")
        
        zip_path = f"output/batch_results_{batch_id}.zip"
        
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(batch_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, batch_dir)
                        zipf.write(file_path, arcname)
            
            if not os.path.exists(zip_path):
                raise Exception("Zip file was not created")
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create zip file: {e}")
        
        return JSONResponse({
            "success": True,
            "message": f"Saved {len(saved_files)} results (original + mask + segmented for each)",
            "zip_download": f"/output/batch_results_{batch_id}.zip",
            "saved_files": saved_files
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in save_all_results: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.get("/get_session_status")
async def get_session_status(session_id: str):
    try:
        session = get_session(session_id)
        
        status_summary = {
            "total_images": len(session.images),
            "uploaded": 0,
            "points_generated": 0,
            "mask_generated": 0,
            "saved": 0
        }
        
        for image_data in session.images.values():
            status_summary[image_data.status] += 1
        
        return JSONResponse({
            "success": True,
            "status": status_summary
        })
        
    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🌱 Starting Multi-Image Plant Segmentation Server")
    print("🌐 Access: http://localhost:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
