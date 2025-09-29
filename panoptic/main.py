# Configuration - Set to True to preload all models at startup
PRELOAD_ALL_MODELS = True  # Change to False to load models on demand

from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import numpy as np
import torch
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import base64
import os
import uuid
from pathlib import Path
import logging
from typing import List
import zipfile
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Plant-related class keywords to identify plant classes
PLANT_KEYWORDS = [
    'plant', 'flower', 'tree', 'grass', 'potted_plant', 'palm', 'field',
    'bush', 'shrub', 'leaf', 'branch', 'stem', 'garden', 'vegetation',
    'flora', 'botanical', 'bloom', 'blossom', 'foliage', 'greenery'
]

# Available models
ALL_MODELS = {
    "semantic": {
        "base": {
            "name": "facebook/mask2former-swin-base-ade-semantic",
            "description": "ADE20K Semantic (Base)"
        },
        "large": {
            "name": "facebook/mask2former-swin-large-ade-semantic",
            "description": "ADE20K Semantic (Large)"
        }
    },
    "instance": {
        "base": {
            "name": "facebook/mask2former-swin-base-coco-instance",
            "description": "COCO Instance (Base)"
        },
        "large": {
            "name": "facebook/mask2former-swin-large-coco-instance", 
            "description": "COCO Instance (Large)"
        }
    },
    "panoptic": {
        "base": {
            "name": "facebook/mask2former-swin-base-coco-panoptic",
            "description": "COCO Panoptic (Base)"
        },
        "large": {
            "name": "facebook/mask2former-swin-large-coco-panoptic",
            "description": "COCO Panoptic (Large)"
        }
    }
}

# Global models cache
models_cache = {}
current_sessions = {}

def get_plant_classes(model):
    """Get plant-related classes from model config"""
    plant_classes = {}
    
    if model and hasattr(model.config, 'id2label') and model.config.id2label:
        for class_id, class_name in model.config.id2label.items():
            class_name_lower = class_name.lower()
            for keyword in PLANT_KEYWORDS:
                if keyword in class_name_lower:
                    plant_classes[class_id] = class_name
                    break
    
    return plant_classes

def create_plant_mask(segmentation, plant_classes):
    """Create binary mask for plant classes"""
    plant_mask = np.zeros_like(segmentation, dtype=bool)
    
    for class_id in plant_classes.keys():
        plant_mask |= (segmentation == class_id)
    
    return plant_mask

def load_model(model_name):
    """Load Mask2Former model with caching"""
    if model_name in models_cache:
        logger.info(f"Using cached model: {model_name}")
        return models_cache[model_name]
    
    logger.info(f"Loading model: {model_name}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        processor = Mask2FormerImageProcessor.from_pretrained(model_name)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
        model.to(device)
        model.eval()
        
        models_cache[model_name] = (model, processor, device)
        logger.info(f"✅ Model loaded: {model_name}")
        
        return model, processor, device
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise

def preload_all_models():
    """Preload all available models at startup"""
    if not PRELOAD_ALL_MODELS:
        return
    
    logger.info("🚀 Preloading all models...")
    
    for seg_type, models in ALL_MODELS.items():
        for model_size, model_info in models.items():
            model_name = model_info["name"]
            try:
                logger.info(f"Loading {model_info['description']}...")
                load_model(model_name)
            except Exception as e:
                logger.error(f"Failed to preload {model_name}: {e}")
    
    logger.info(f"✅ Preloaded {len(models_cache)} models")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🎭 Starting Mask2Former Plant Segmentation Web App")
    preload_all_models()
    yield
    # Shutdown
    pass

app = FastAPI(title="Mask2Former Plant Segmentation Web App", lifespan=lifespan)

# Create directories and mount static files
os.makedirs("templates", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("output", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/output", StaticFiles(directory="output"), name="output")

templates = Jinja2Templates(directory="templates")

class ImageData:
    def __init__(self, image_id: str, filename: str, filepath: str):
        self.image_id = image_id
        self.filename = filename
        self.filepath = filepath
        self.original_image = None
        self.result_image_path = None
        self.detected_classes = []
        self.plant_coverage = 0.0
        self.status = "uploaded"  # uploaded, processed, failed

class ProcessingSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.images = {}  # image_id -> ImageData
        
    def add_image(self, image_data: ImageData):
        self.images[image_data.image_id] = image_data
        
    def get_image(self, image_id: str) -> ImageData:
        return self.images.get(image_id)

def get_session(session_id: str) -> ProcessingSession:
    if session_id not in current_sessions:
        current_sessions[session_id] = ProcessingSession(session_id)
    return current_sessions[session_id]

def visualize_segmentation(image, results, seg_type, model, filename):
    """Create visualization based on segmentation type"""
    try:
        # Get plant classes
        plant_classes = get_plant_classes(model)
        
        # Create 2x3 layout
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 1. Original image (top-left)
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontsize=14, weight='bold')
        axes[0].axis('off')
        
        image_array = np.array(image)
        
        if seg_type == "semantic":
            # Semantic segmentation
            segmentation = results.cpu().numpy()
            plant_mask = create_plant_mask(segmentation, plant_classes)
            
            # 2. Plant detection mask (top-center)
            plant_vis = np.zeros_like(segmentation)
            plant_vis[plant_mask] = 1
            axes[1].imshow(plant_vis, cmap='RdYlGn', vmin=0, vmax=1)
            axes[1].set_title('Plant Detection\n(Green=Plants)', fontsize=14, weight='bold')
            axes[1].axis('off')
            
            # 3. Overlay (top-right)
            overlay = image_array.copy().astype(float)
            overlay[plant_mask] = overlay[plant_mask] * 0.6 + np.array([0, 255, 0]) * 0.4
            axes[2].imshow(overlay.astype(np.uint8))
            axes[2].set_title('Overlay\n(Green=Detected Plants)', fontsize=14, weight='bold')
            axes[2].axis('off')
            
            # 4. All classes with labels (bottom-left)
            axes[3].imshow(segmentation, cmap='tab20')
            
            # Add labels to each class segment
            unique_classes = np.unique(segmentation)
            total_pixels = segmentation.size
            
            for class_id in unique_classes:
                # Find pixels belonging to this class
                mask = segmentation == class_id
                if np.any(mask):
                    # Find centroid of the class
                    y_coords, x_coords = np.where(mask)
                    centroid_y = int(np.mean(y_coords))
                    centroid_x = int(np.mean(x_coords))
                    
                    # Calculate percentage
                    class_pixels = mask.sum()
                    percentage = (class_pixels / total_pixels) * 100
                    
                    # Get class name from model config
                    if model and hasattr(model.config, 'id2label') and class_id in model.config.id2label:
                        class_name = model.config.id2label[class_id]
                    else:
                        class_name = f"class_{class_id}"
                    
                    # Create shorter label text
                    if len(class_name) > 10:
                        short_name = class_name[:8] + ".."
                    else:
                        short_name = class_name
                    
                    label_text = f"{short_name} {percentage:.0f}%"
                    
                    # Add text with background
                    axes[3].annotate(label_text, (centroid_x, centroid_y), 
                                color='white', fontsize=8, weight='bold',
                                ha='center', va='center',
                                bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.7))
            
            axes[3].set_title('All Classes\n(With Labels)', fontsize=14, weight='bold')
            axes[3].axis('off')
            
            # 5. Plants only (bottom-center)
            plants_only = np.ones_like(image_array) * 255
            plants_only[plant_mask] = image_array[plant_mask]
            axes[4].imshow(plants_only.astype(np.uint8))
            axes[4].set_title('Plants Only\n(White Background)', fontsize=14, weight='bold')
            axes[4].axis('off')
            
            # Statistics
            detected_plant_classes = [plant_classes[cid] for cid in plant_classes.keys() 
                                    if cid in np.unique(segmentation)]
            plant_coverage = (plant_mask.sum() / plant_mask.size) * 100
            
        else:
            # Instance/Panoptic segmentation
            if "segmentation" in results:
                segmentation = results["segmentation"].cpu().numpy()
                segments_info = results.get("segments_info", [])
                
                # Create plant mask based on segments_info
                plant_mask = np.zeros_like(segmentation, dtype=bool)
                plant_segments = []
                
                for segment in segments_info:
                    segment_id = segment['id']
                    label_id = segment.get('label_id', 0)
                    if label_id in plant_classes:
                        plant_mask |= (segmentation == segment_id)
                        plant_segments.append(segment)
                
                # 2. Plant detection mask (top-center)
                plant_vis = np.zeros_like(segmentation)
                plant_vis[plant_mask] = 1
                axes[1].imshow(plant_vis, cmap='RdYlGn', vmin=0, vmax=1)
                axes[1].set_title('Plant Detection\n(Green=Plants)', fontsize=14, weight='bold')
                axes[1].axis('off')
                
                # 3. Overlay (top-right)
                overlay = image_array.copy().astype(float)
                overlay[plant_mask] = overlay[plant_mask] * 0.6 + np.array([0, 255, 0]) * 0.4
                axes[2].imshow(overlay.astype(np.uint8))
                axes[2].set_title('Overlay\n(Green=Detected Plants)', fontsize=14, weight='bold')
                axes[2].axis('off')
                
                # 4. All segments with labels (bottom-left)
                axes[3].imshow(segmentation, cmap='tab20')
                
                # Add labels to each segment
                total_pixels = segmentation.size
                for segment in segments_info:
                    segment_id = segment['id']
                    label_id = segment.get('label_id', 0)
                    score = segment.get('score', 0.0)
                    
                    # Find pixels belonging to this segment
                    mask = segmentation == segment_id
                    if np.any(mask):
                        # Find centroid of the segment
                        y_coords, x_coords = np.where(mask)
                        centroid_y = int(np.mean(y_coords))
                        centroid_x = int(np.mean(x_coords))
                        
                        # Calculate percentage
                        segment_pixels = mask.sum()
                        percentage = (segment_pixels / total_pixels) * 100
                        
                        # Get class name from model config
                        if model and hasattr(model.config, 'id2label') and label_id in model.config.id2label:
                            class_name = model.config.id2label[label_id]
                        else:
                            class_name = f"class_{label_id}"
                        
                        # Create shorter label text
                        if len(class_name) > 10:
                            short_name = class_name[:8] + ".."
                        else:
                            short_name = class_name
                        
                        label_text = f"{short_name} {score*100:.0f}%"
                        
                        # Add text with background
                        axes[3].annotate(label_text, (centroid_x, centroid_y), 
                                    color='white', fontsize=8, weight='bold',
                                    ha='center', va='center',
                                    bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.7))
                
                axes[3].set_title(f'All Segments\n({len(segments_info)} with labels)', fontsize=14, weight='bold')
                axes[3].axis('off')
                
                # 5. Plants only (bottom-center)
                plants_only = np.ones_like(image_array) * 255
                plants_only[plant_mask] = image_array[plant_mask]
                axes[4].imshow(plants_only.astype(np.uint8))
                axes[4].set_title('Plants Only\n(White Background)', fontsize=14, weight='bold')
                axes[4].axis('off')
                
                # Statistics
                detected_plant_classes = list(set([plant_classes[segment.get('label_id', 0)] 
                                                 for segment in plant_segments 
                                                 if segment.get('label_id', 0) in plant_classes]))
                plant_coverage = (plant_mask.sum() / plant_mask.size) * 100 if plant_mask.size > 0 else 0
            else:
                # No segmentation data
                for i in [1, 2, 3, 4]:
                    axes[i].text(0.5, 0.5, "No segments found", ha='center', va='center',
                                transform=axes[i].transAxes, fontsize=14)
                    axes[i].axis('off')
                detected_plant_classes = []
                plant_coverage = 0
        
        # 6. Statistics (bottom-right)
        axes[5].axis('off')
        
        # Create statistics text
        stats_text = ""
        if detected_plant_classes:
            stats_text += "PLANT CLASSES DETECTED:\n"
            stats_text += "-" * 25 + "\n"
            for class_name in detected_plant_classes:
                stats_text += f"• {class_name}\n"
            stats_text += f"\nPlant Coverage: {plant_coverage:.1f}%\n"
        else:
            stats_text += "NO PLANTS DETECTED\n"
            stats_text += "-" * 18 + "\n"
        
        stats_text += f"\nSEGMENTATION TYPE:\n"
        stats_text += "-" * 18 + "\n"
        stats_text += f"Type: {seg_type.title()}\n"
        
        # Image info
        stats_text += f"\nIMAGE INFO:\n"
        stats_text += "-" * 11 + "\n"
        stats_text += f"Size: {image_array.shape[1]} x {image_array.shape[0]} px\n"
        stats_text += f"Total pixels: {image_array.shape[0] * image_array.shape[1]:,}\n"
        
        # Add the text
        axes[5].text(0.05, 0.95, stats_text, transform=axes[5].transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.6", facecolor="white", alpha=0.95,
                             edgecolor="gray", linewidth=1))
        
        # Main title
        if detected_plant_classes:
            main_title = f"Plant Detection Results - {', '.join(detected_plant_classes[:2])} detected"
        else:
            main_title = "Plant Detection Results - No plants detected"
        
        plt.suptitle(main_title, fontsize=16, weight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.94, hspace=0.1, wspace=0.1)
        
        # Save the result
        base_name = Path(filename).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"output/{seg_type}_detection_{base_name}_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()  # Close to free memory
        
        return save_path, detected_plant_classes, plant_coverage
        
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        return None, [], 0

def image_to_base64(image_path):
    """Convert image file to base64"""
    try:
        with open(image_path, "rb") as img_file:
            return f"data:image/png;base64,{base64.b64encode(img_file.read()).decode('utf-8')}"
    except:
        return None

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/models")
async def get_available_models():
    """Get available models for each segmentation type"""
    return JSONResponse(ALL_MODELS)

@app.post("/upload_images")
async def upload_images(files: List[UploadFile] = File(...), session_id: str = Form(...)):
    try:
        session = get_session(session_id)
        uploaded_images = []
        
        for file in files:
            if not file.content_type.startswith('image/'):
                continue
                
            # Save file
            image_id = str(uuid.uuid4())
            filename = f"{image_id}_{file.filename}"
            file_path = f"uploads/{filename}"
            
            contents = await file.read()
            with open(file_path, "wb") as f:
                f.write(contents)
            
            # Load image
            pil_image = Image.open(file_path).convert('RGB')
            image_array = np.array(pil_image)
            
            # Create image data
            image_data = ImageData(image_id, file.filename, file_path)
            image_data.original_image = image_array
            session.add_image(image_data)
            
            uploaded_images.append({
                "image_id": image_id,
                "filename": file.filename,
                "preview": image_to_base64(file_path),
                "status": "uploaded"
            })
        
        return JSONResponse({
            "success": True,
            "message": f"Uploaded {len(uploaded_images)} images",
            "images": uploaded_images
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_images")
async def process_images(
    session_id: str = Form(...), 
    seg_type: str = Form(...),
    model_size: str = Form(...),
    threshold: float = Form(0.5)
):
    try:
        session = get_session(session_id)
        
        # Get model name
        if seg_type not in ALL_MODELS or model_size not in ALL_MODELS[seg_type]:
            raise HTTPException(status_code=400, detail="Invalid segmentation type or model size")
        
        model_name = ALL_MODELS[seg_type][model_size]["name"]
        
        # Load model
        model, processor, device = load_model(model_name)
        
        results = []
        
        for image_id, image_data in session.images.items():
            try:
                # Convert numpy array to PIL Image
                image = Image.fromarray(image_data.original_image)
                
                # Prepare inputs
                inputs = processor(images=image, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Run inference
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Process results based on segmentation type
                if seg_type == "semantic":
                    processed_results = processor.post_process_semantic_segmentation(
                        outputs, target_sizes=[image.size[::-1]]
                    )[0]
                elif seg_type == "instance":
                    processed_results = processor.post_process_instance_segmentation(
                        outputs, target_sizes=[image.size[::-1]], threshold=threshold
                    )[0]
                else:  # panoptic
                    processed_results = processor.post_process_panoptic_segmentation(
                        outputs, target_sizes=[image.size[::-1]], threshold=threshold
                    )[0]
                
                # Create visualization
                viz_path, detected_classes, plant_coverage = visualize_segmentation(
                    image, processed_results, seg_type, model, image_data.filename
                )
                
                if viz_path:
                    image_data.result_image_path = viz_path
                    image_data.detected_classes = detected_classes
                    image_data.plant_coverage = plant_coverage
                    image_data.status = "processed"
                    
                    results.append({
                        "image_id": image_id,
                        "filename": image_data.filename,
                        "detected_classes": detected_classes,
                        "plant_coverage": plant_coverage,
                        "result_preview": image_to_base64(viz_path),
                        "status": "processed"
                    })
                else:
                    image_data.status = "failed"
                    results.append({
                        "image_id": image_id,
                        "filename": image_data.filename,
                        "status": "failed",
                        "error": "Visualization failed"
                    })
                    
            except Exception as e:
                logger.error(f"Error processing image {image_id}: {e}")
                image_data.status = "failed"
                results.append({
                    "image_id": image_id,
                    "filename": image_data.filename,
                    "status": "failed",
                    "error": str(e)
                })
        
        return JSONResponse({
            "success": True,
            "message": f"Processed {len(results)} images",
            "results": results
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/download_results")
async def download_results(session_id: str = Form(...)):
    try:
        session = get_session(session_id)
        
        # Create output directory for this batch
        batch_id = str(uuid.uuid4())[:8]
        batch_dir = f"output/batch_{batch_id}"
        os.makedirs(batch_dir, exist_ok=True)
        
        saved_files = []
        
        for image_id, image_data in session.images.items():
            if image_data.status == "processed" and image_data.result_image_path:
                base_name = Path(image_data.filename).stem
                
                # Copy original image
                original_dest = f"{batch_dir}/original_{base_name}.jpg"
                pil_image = Image.fromarray(image_data.original_image)
                pil_image.save(original_dest, 'JPEG')
                
                # Copy result visualization
                result_dest = f"{batch_dir}/result_{base_name}.png"
                import shutil
                shutil.copy2(image_data.result_image_path, result_dest)
                
                saved_files.append({
                    "filename": image_data.filename,
                    "classes": image_data.detected_classes,
                    "coverage": image_data.plant_coverage
                })
        
        # Create summary report
        summary_path = f"{batch_dir}/summary_report.txt"
        with open(summary_path, 'w') as f:
            f.write("MASK2FORMER PLANT SEGMENTATION RESULTS\n")
            f.write("=" * 40 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Images Processed: {len(saved_files)}\n\n")
            
            for i, file_info in enumerate(saved_files, 1):
                f.write(f"{i}. {file_info['filename']}\n")
                f.write(f"   Classes: {', '.join(file_info['classes']) if file_info['classes'] else 'None'}\n")
                f.write(f"   Coverage: {file_info['coverage']:.1f}%\n\n")
        
        # Create zip file
        zip_path = f"output/mask2former_results_{batch_id}.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for root, dirs, files in os.walk(batch_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, batch_dir)
                    zipf.write(file_path, arcname)
        
        return JSONResponse({
            "success": True,
            "message": f"Results ready for download",
            "zip_download": f"/output/mask2former_results_{batch_id}.zip",
            "total_files": len(saved_files)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🎭 Starting Mask2Former Plant Segmentation Web App")
    print("🌐 Access: http://localhost:8003")
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=True)
