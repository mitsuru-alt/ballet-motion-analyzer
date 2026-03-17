import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

import uvicorn
from app.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
