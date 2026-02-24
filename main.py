"""
Image Recognition Center — Backend
FastAPI WebSocket server for real-time video processing
"""

import asyncio
import base64
import json
import logging
from contextlib import asynccontextmanager
from typing import Callable

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from processors import hand_tracking, pose_detection, face_mesh

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Processor registry (Lite Version — MediaPipe only)
PROCESSORS: dict[str, Callable] = {
    "hand_tracking": hand_tracking.process,
    "pose_detection": pose_detection.process,
    "face_mesh": face_mesh.process,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup."""
    logger.info("Initializing processors...")
    
    # Warm up each processor
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for name, processor in PROCESSORS.items():
        try:
            processor(dummy_frame)
            logger.info(f"✓ {name} initialized")
        except Exception as e:
            logger.error(f"✗ {name} failed to initialize: {e}")
    
    logger.info("All processors ready")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Image Recognition Center API",
    description="Real-time video processing via WebSocket",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration for Render deployment
# In production, replace with your actual frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://*.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def decode_frame(data: bytes) -> np.ndarray | None:
    """Decode base64 or binary frame data to numpy array."""
    try:
        # Handle base64-encoded data
        if isinstance(data, str):
            # Remove data URL prefix if present
            if "," in data:
                data = data.split(",")[1]
            data = base64.b64decode(data)
        
        # Decode image
        nparr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.error(f"Frame decode error: {e}")
        return None


def encode_frame(frame: np.ndarray, quality: int = 80) -> str:
    """Encode numpy array to base64 JPEG."""
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode("utf-8")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "Image Recognition Center",
        "processors": list(PROCESSORS.keys()),
    }


@app.get("/processors")
async def list_processors():
    """List available processors."""
    return {"processors": list(PROCESSORS.keys())}


@app.websocket("/ws/process")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for video processing.
    
    Protocol:
    - Client sends: {"mode": "processor_name"} to switch processor
    - Client sends: base64 or binary frame data
    - Server sends: {"frame": "base64_data", "fps": float} or {"error": "message"}
    """
    await websocket.accept()
    logger.info("Client connected")
    
    current_mode: str | None = None
    frame_count = 0
    start_time = asyncio.get_event_loop().time()
    
    try:
        while True:
            # Receive message (can be text or binary)
            message = await websocket.receive()
            
            if "text" in message:
                # Text message — could be mode switch or base64 frame
                text = message["text"]
                
                try:
                    data = json.loads(text)
                    
                    # Mode switch command
                    if "mode" in data:
                        new_mode = data["mode"]
                        if new_mode in PROCESSORS:
                            current_mode = new_mode
                            logger.info(f"Mode switched to: {current_mode}")
                            await websocket.send_json({"status": "mode_changed", "mode": current_mode})
                        elif new_mode is None or new_mode == "none":
                            current_mode = None
                            logger.info("Processing disabled")
                            await websocket.send_json({"status": "mode_changed", "mode": None})
                        else:
                            await websocket.send_json({"error": f"Unknown processor: {new_mode}"})
                        continue
                    
                    # Frame data in JSON
                    if "frame" in data:
                        frame_data = data["frame"]
                    else:
                        continue
                        
                except json.JSONDecodeError:
                    # Assume it's raw base64 frame data
                    frame_data = text
                
                # Process frame
                frame = decode_frame(frame_data)
                
            elif "bytes" in message:
                # Binary frame data
                frame = decode_frame(message["bytes"])
            else:
                continue
            
            if frame is None:
                await websocket.send_json({"error": "Invalid frame data"})
                continue
            
            # Apply processor if mode is set
            if current_mode and current_mode in PROCESSORS:
                try:
                    processed = PROCESSORS[current_mode](frame)
                except Exception as e:
                    logger.error(f"Processing error ({current_mode}): {e}")
                    processed = frame
            else:
                processed = frame
            
            # Calculate FPS
            frame_count += 1
            elapsed = asyncio.get_event_loop().time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Reset counter periodically to get recent FPS
            if frame_count >= 30:
                frame_count = 0
                start_time = asyncio.get_event_loop().time()
            
            # Send processed frame
            encoded = encode_frame(processed)
            await websocket.send_json({
                "frame": encoded,
                "fps": round(fps, 1),
                "mode": current_mode,
            })
            
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("Connection closed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
