from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
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

# Plant classes from ADE20K dataset
PLANT_CLASSES = {
    'tree': 4,
    'grass': 9,
    'plant': 17,
    'field': 29,
    'flower': 66,
    'palm': 72,
}

ADE20K_CLASSES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", 
    "windowpane", "grass", "cabinet", "sidewalk", "person", "earth", "door", 
    "table", "mountain", "plant", "curtain", "chair", "car", "water", "painting", 
    "sofa", "shelf", "house", "sea", "mirror", "rug", "field", "armchair", 
    "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub", "railing", 
    "cushion", "base", "box", "column", "signboard", "chest_of_drawers", "counter", 
    "sand", "sink", "skyscraper", "fireplace", "refrigerator", "grandstand", 
    "path", "stairs", "runway", "case", "pool_table", "pillow", "screen_door", 
    "stairway", "river", "bridge", "bookcase", "blind", "coffee_table", "toilet", 
    "flower", "book", "hill", "bench", "countertop", "stove", "palm", "kitchen_island", 
    "computer", "swivel_chair", "boat", "bar", "arcade_machine", "hovel", "bus", 
    "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight", 
    "booth", "television", "airplane", "dirt_track", "apparel", "pole", "land", 
    "bannister", "escalator", "ottoman", "bottle", "buffet", "poster", "stage", 
    "van", "ship", "fountain", "conveyer_belt", "canopy", "washer", "plaything", 
    "swimming_pool", "stool", "barrel", "basket", "waterfall", "tent", "bag", 
    "minibike", "cradle", "oven", "ball", "food", "step", "tank", "trade_name", 
    "microwave", "pot", "animal", "bicycle", "lake", "dishwasher", "screen", 
    "blanket", "sculpture", "hood", "sconce", "vase", "traffic_light", "tray", 
    "ashcan", "fan", "pier", "crt_screen", "plate", "monitor", "bulletin_board", 
    "shower", "radiator", "glass", "clock", "flag"
]

# Global models
segformer_model = None
segformer_processor = None
device = None
current_sessions = {}

def initialize_models():
    """Initialize Segformer model"""
    global segformer_model, segformer_processor, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # Load Segformer
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
    # Startup
    if not initialize_models():
        logger.error("Failed to initialize models!")
    yield
    # Shutdown
    pass

app = FastAPI(title="Plant Segmentation Web App", lifespan=lifespan)

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

def segment_plants(image_array):
    """Segment plants using Segformer - returns predicted_labels, plant_mask, detected_classes"""
    try:
        # Convert numpy array to PIL Image
        image = Image.fromarray(image_array)
        
        # Preprocess image
        inputs = segformer_processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = segformer_model(pixel_values)
            logits = outputs.logits
            
            # Resize to original image size
            upsampled_logits = F.interpolate(
                logits, 
                size=image_array.shape[:2],
                mode="bilinear", 
                align_corners=False
            )
            
            # Get predictions
            predicted_labels = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
        
        # Create plant mask
        plant_mask = np.zeros_like(predicted_labels, dtype=bool)
        detected_plant_classes = []
        
        for class_name, class_id in PLANT_CLASSES.items():
            if class_id < len(ADE20K_CLASSES):
                class_pixels = (predicted_labels == class_id)
                if class_pixels.sum() > 0:  # If this class is detected
                    plant_mask |= class_pixels
                    detected_plant_classes.append(class_name)
        
        return predicted_labels, plant_mask, detected_plant_classes
        
    except Exception as e:
        logger.error(f"Segmentation error: {e}")
        return None, None, []

def create_visualization(image, predicted_labels, plant_mask, detected_classes, filename):
    """Create 2x3 visualization like in full.py"""
    try:
        # Create 2x3 layout
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 1. Original image (top-left)
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontsize=14, weight='bold')
        axes[0].axis('off')
        
        # 2. Plant mask only (top-center)
        axes[1].imshow(plant_mask, cmap='RdYlGn', alpha=0.8)
        axes[1].set_title('Plant Detection\n(Green=Plants)', fontsize=14, weight='bold')
        axes[1].axis('off')
        
        # 3. Overlay (top-right)
        image_array = np.array(image)
        overlay = image_array.copy()
        overlay[plant_mask] = [0, 255, 0]  # Green for plants
        blended = 0.7 * image_array + 0.3 * overlay
        
        axes[2].imshow(blended.astype(np.uint8))
        axes[2].set_title('Overlay\n(Green=Detected Plants)', fontsize=14, weight='bold')
        axes[2].axis('off')
        
        # 4. All classes with different colors (bottom-left)
        colored_segmentation = np.zeros((*predicted_labels.shape, 3), dtype=np.uint8)
        unique_classes = np.unique(predicted_labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_classes)))
        
        for i, class_id in enumerate(unique_classes):
            mask = predicted_labels == class_id
            colored_segmentation[mask] = (colors[i][:3] * 255).astype(np.uint8)
        
        axes[3].imshow(colored_segmentation)
        axes[3].set_title('All Classes\n(Different Colors)', fontsize=14, weight='bold')
        axes[3].axis('off')
        
        # 5. Plant with white background (bottom-center)
        plant_on_white = np.ones_like(image_array) * 255  # White background
        plant_on_white[plant_mask] = image_array[plant_mask]  # Keep plant pixels
        
        axes[4].imshow(plant_on_white.astype(np.uint8))
        axes[4].set_title('Plants Only\n(White Background)', fontsize=14, weight='bold')
        axes[4].axis('off')
        
        # 6. Plant statistics (bottom-right)
        axes[5].axis('off')
        
        # Get all detected classes with their colors and percentages
        unique_classes = np.unique(predicted_labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_classes)))
        
        # Prepare statistics text
        stats_text = ""
        
        # First show plant classes if detected
        if detected_classes:
            stats_text += "PLANT CLASSES DETECTED:\n"
            stats_text += "-" * 25 + "\n"
            for class_name in detected_classes:
                class_id = PLANT_CLASSES[class_name]
                if class_id in unique_classes:
                    class_pixels = (predicted_labels == class_id).sum()
                    percentage = (class_pixels / predicted_labels.size) * 100
                    stats_text += f"* {class_name}: {percentage:.1f}%\n"
            
            plant_percentage = (plant_mask.sum() / plant_mask.size) * 100
            stats_text += f"\nTotal Plant Coverage: {plant_percentage:.1f}%\n"
            stats_text += f"Plant Pixels: {plant_mask.sum():,}\n"
        else:
            stats_text += "NO PLANTS DETECTED\n"
            stats_text += "-" * 18 + "\n"
        
        # Then show all other classes
        stats_text += f"\nALL DETECTED CLASSES:\n"
        stats_text += "-" * 21 + "\n"
        
        # Sort classes by percentage (largest first)
        class_info = []
        for i, class_id in enumerate(unique_classes):
            if class_id < len(ADE20K_CLASSES):
                class_name = ADE20K_CLASSES[class_id]
                class_pixels = (predicted_labels == class_id).sum()
                percentage = (class_pixels / predicted_labels.size) * 100
                color_rgb = colors[i][:3]
                
                # Mark if it's a plant class
                is_plant = class_id in PLANT_CLASSES.values()
                class_info.append({
                    'name': class_name,
                    'id': class_id,
                    'percentage': percentage,
                    'pixels': class_pixels,
                    'color': color_rgb,
                    'is_plant': is_plant
                })
        
        # Sort by percentage
        class_info.sort(key=lambda x: x['percentage'], reverse=True)
        
        # Add class information
        for i, info in enumerate(class_info):
            plant_mark = "[P]" if info['is_plant'] else "   "
            stats_text += f"{plant_mark} {info['name']}: {info['percentage']:.1f}%\n"
            if len(class_info) > 6:  # If too many classes, show only top 6
                if i == 5:  # After 6th item
                    remaining = len(class_info) - 6
                    stats_text += f"    ... and {remaining} more classes\n"
                    break
        
        # Image info
        stats_text += f"\nIMAGE INFO:\n"
        stats_text += "-" * 11 + "\n"
        stats_text += f"Size: {image_array.shape[1]} x {image_array.shape[0]} px\n"
        stats_text += f"Total pixels: {predicted_labels.size:,}\n"
        stats_text += f"Classes found: {len(unique_classes)}\n"
        
        # Add the text with white background
        axes[5].text(0.05, 0.95, stats_text, transform=axes[5].transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.6", facecolor="white", alpha=0.95, 
                             edgecolor="gray", linewidth=1))
        
        # Add colored squares for each class
        y_start = 0.85
        x_start = 0.75
        square_size = 0.03
        
        # Add color legend title
        axes[5].text(x_start, y_start + 0.05, "Colors:", transform=axes[5].transAxes, 
                    fontsize=10, weight='bold', fontfamily='monospace')
        
        # Add colored squares for main classes (top ones only)
        for i, info in enumerate(class_info[:6]):  # Show colors for top 6 classes
            y_pos = y_start - (i * 0.08)
            
            # Draw colored square
            square = patches.Rectangle((x_start, y_pos - square_size/2), square_size, square_size, 
                                  facecolor=info['color'], edgecolor='black', linewidth=0.5,
                                  transform=axes[5].transAxes, clip_on=False)
            axes[5].add_patch(square)
            
            # Add class name next to square
            class_text = f"{info['name'][:8]}..." if len(info['name']) > 8 else info['name']
            axes[5].text(x_start + square_size + 0.01, y_pos, class_text, 
                        transform=axes[5].transAxes, fontsize=8, 
                        verticalalignment='center', fontfamily='monospace')
        
        # Add main title
        if detected_classes:
            main_title = f"Plant Detection Results - {', '.join(detected_classes)} detected"
        else:
            main_title = "Plant Detection Results - No plants detected"
        
        plt.suptitle(main_title, fontsize=16, weight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.94, hspace=0.1, wspace=0.1)
        
        # Save the result
        base_name = Path(filename).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"output/plant_detection_{base_name}_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()  # Close to free memory
        
        return save_path
        
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        return None

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

@app.post("/segment_all")
async def segment_all_images(session_id: str = Form(...)):
    try:
        session = get_session(session_id)
        results = []
        
        for image_id, image_data in session.images.items():
            try:
                # Run segmentation
                predicted_labels, plant_mask, detected_classes = segment_plants(image_data.original_image)
                
                if plant_mask is not None:
                    # Calculate plant coverage
                    plant_coverage = (plant_mask.sum() / plant_mask.size) * 100
                    
                    # Create visualization
                    viz_path = create_visualization(
                        image_data.original_image, 
                        predicted_labels, 
                        plant_mask, 
                        detected_classes, 
                        image_data.filename
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
                else:
                    image_data.status = "failed"
                    results.append({
                        "image_id": image_id,
                        "filename": image_data.filename,
                        "status": "failed",
                        "error": "Segmentation failed"
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
            f.write("PLANT SEGMENTATION RESULTS\n")
            f.write("=" * 30 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Images Processed: {len(saved_files)}\n\n")
            
            for i, file_info in enumerate(saved_files, 1):
                f.write(f"{i}. {file_info['filename']}\n")
                f.write(f"   Classes: {', '.join(file_info['classes']) if file_info['classes'] else 'None'}\n")
                f.write(f"   Coverage: {file_info['coverage']:.1f}%\n\n")
        
        # Create zip file
        zip_path = f"output/plant_results_{batch_id}.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for root, dirs, files in os.walk(batch_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, batch_dir)
                    zipf.write(file_path, arcname)
        
        return JSONResponse({
            "success": True,
            "message": f"Results ready for download",
            "zip_download": f"/output/plant_results_{batch_id}.zip",
            "total_files": len(saved_files)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🌱 Starting Plant Segmentation Web App")
    print("🌐 Access: http://localhost:8002")
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
