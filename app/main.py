import os
import base64
import cv2
import numpy as np
import threading
import logging
import uuid
import platform
import hashlib
from datetime import datetime, timedelta
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image, ImageSequence
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('captcha_solver.log'),
        logging.StreamHandler()
    ]
)

VERSION = '3.0'

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def get_device_fingerprint():
    """Generate a unique device fingerprint based on hardware info"""
    # For web deployment, we'll use a simpler approach
    return str(uuid.uuid4())

class CaptchaEngine:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        configPath = resource_path('data.cfg')
        weightPath = resource_path('data.weights')
        labelPath = resource_path('data.nms')
        
        logging.info(f"Loading model from: config={configPath}, weights={weightPath}, labels={labelPath}")
        
        # Load YOLO model
        self.net = cv2.dnn.readNetFromDarknet(configPath, weightPath)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Load class names
        with open(labelPath, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        
        # Warm up the network
        blank_image = np.zeros((height, width, 3), np.uint8)
        self.net.detect(blank_image, confThreshold=0.4, nmsThreshold=0.4)
        self.lock = threading.Lock()

    def solve_cv2(self, image):
        def get_key_x(item):
            return item[0]
        
        with self.lock:
            blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)
            
            # Get outputs from output layers
            layer_names = self.net.getLayerNames()
            output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            outputs = self.net.forward(output_layers)
            
            # Process detections
            class_ids = []
            confidences = []
            boxes = []
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > 0.5:
                        center_x = int(detection[0] * self.width)
                        center_y = int(detection[1] * self.height)
                        w = int(detection[2] * self.width)
                        h = int(detection[3] * self.height)
                        
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply non-maximum suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            
            result = []
            for i in indices:
                if isinstance(i, list):  # Handle different OpenCV versions
                    i = i[0]
                box = boxes[i]
                x, y, w, h = box
                label = str(self.classes[class_ids[i]])
                result.append((x, label))
            
            # Sort by x-coordinate
            result.sort(key=get_key_x)
            return ''.join([r[1] for r in result])

    def solve(self, sbase64):
        image = self.base64_to_cv2(sbase64)
        return self.solve_cv2(image)

    def base64_to_cv2(self, sbase64):
        img_data = base64.b64decode(sbase64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

def merge_gif_preserve_color(base64_gif_str):
    """Merge GIF frames into a single image while preserving colors"""
    gif_data = base64.b64decode(base64_gif_str)
    gif = Image.open(BytesIO(gif_data))
    
    # Get dimensions
    width, height = gif.size
    
    # Create a new blank image with white background
    merged_image = Image.new('RGB', (width, height), (255, 255, 255))
    
    # Iterate through each frame and overlay it
    for frame in ImageSequence.Iterator(gif):
        # Convert frame to RGB
        rgb_frame = frame.convert('RGB')
        # Overlay the frame onto the merged image
        merged_image = Image.blend(merged_image, rgb_frame, 0.5)
    
    # Convert back to OpenCV format
    buffered = BytesIO()
    merged_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode('utf-8')

# Initialize FastAPI app
app = FastAPI(title="ESCS Captcha Solver API", version=VERSION)

# Configure CORS
origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize captcha solver
width = 256
height = 256
Solver = CaptchaEngine(width, height)
executor = ThreadPoolExecutor(max_workers=4)

# Counter for solved captchas
solve_counter = 0

@app.post("/api/base64")
async def api_base64(request: Request):
    global solve_counter
    try:
        data = await request.json()
        base64_data = data.get('base64', '')
        
        if not base64_data:
            return JSONResponse(status_code=400, content={"error": "No base64 data provided"})
        
        # Check if it's a GIF
        is_gif = False
        if base64_data.startswith('data:image/gif;base64,'):
            is_gif = True
            base64_data = base64_data.replace('data:image/gif;base64,', '')
            # Process GIF
            base64_data = merge_gif_preserve_color(base64_data)
        else:
            # Remove data URL prefix if present
            if ',' in base64_data:
                base64_data = base64_data.split(',', 1)[1]
        
        # Solve captcha
        result = Solver.solve(base64_data)
        solve_counter += 1
        
        logging.info(f"Solved captcha #{solve_counter}: {result}")
        return {"result": result, "counter": solve_counter}
    
    except Exception as e:
        logging.error(f"Error processing captcha: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": VERSION}

@app.get("/")
async def root():
    return {
        "message": "ESCS Captcha Solver API",
        "version": VERSION,
        "endpoints": {
            "/api/base64": "POST - Solve captcha from base64 image",
            "/health": "GET - Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
