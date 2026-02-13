# IRC Backend

FastAPI WebSocket server for real-time video processing.

## Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### REST

- `GET /` — Health check, lists available processors
- `GET /processors` — List processor names

### WebSocket

- `WS /ws/process` — Main processing endpoint

#### Protocol

**Client → Server:**
```json
// Switch processing mode
{"mode": "hand_tracking"}

// Send frame (base64 JPEG)
{"frame": "base64_data_here..."}
```

**Server → Client:**
```json
// Mode confirmation
{"status": "mode_changed", "mode": "hand_tracking"}

// Processed frame
{"frame": "base64_data", "fps": 15.2, "mode": "hand_tracking"}

// Error
{"error": "error message"}
```

## Deploy to Render

1. Create new **Web Service**
2. Connect your repo
3. Settings:
   - **Runtime:** Python 3.11
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

Note: First request may be slow due to model loading. Consider using a paid plan for persistent instances.

## Adding Processors

1. Create `processors/your_processor.py`:
```python
import numpy as np

def process(frame: np.ndarray) -> np.ndarray:
    # Your processing logic
    return frame
```

2. Register in `processors/__init__.py`

3. Add to `PROCESSORS` dict in `main.py`
