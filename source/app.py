#!/usr/bin/env python3
"""
Simple FastAPI app for IntrigueAI
Uses your existing main.py script for predictions
"""

from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
import uvicorn
import logging
import subprocess
import json
import tempfile
import os
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="IntrigueAI", description="YouTube Performance Predictor")

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

class VideoInput(BaseModel):
    title: str
    description: Optional[str] = ""
    transcript_text: Optional[str] = ""  # Added transcript support!
    duration_seconds: int = 600
    uploader: Optional[str] = "Unknown Channel"
    categories: Optional[str] = ""
    tags: Optional[str] = ""

def check_model_exists():
    """Check if the trained model exists"""
    return os.path.exists("models/youtube_predictor/model.h5")

@app.on_event("startup")
async def startup_event():
    """Check model on startup"""
    if check_model_exists():
        logger.info("✅ Model found - ready for predictions!")
    else:
        logger.error("❌ Model not found! Please train your model first:")
        logger.error("   python3 main.py --train ../data/your_dataset.csv")

def predict_with_script(video_data: dict) -> dict:
    """Use your existing main.py script for prediction"""
    
    if not check_model_exists():
        raise Exception("Model not found. Please train your model first.")
    
    try:
        # Create a temporary file with video data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Format data for your existing model
            prediction_input = {
                'title': video_data['title'],
                'description': video_data['description'],
                'transcript_text': video_data.get('transcript_text', ''),  # Include transcript
                'duration_seconds': video_data['duration_seconds'],
                'uploader': video_data['uploader'],
                'categories': video_data['categories'],
                'tags': video_data['tags'],
                'upload_date': datetime.now().strftime('%Y%m%d'),
            }
            
            json.dump(prediction_input, f)
            temp_file = f.name
        
        # Call your existing prediction script
        # We'll create a simple prediction wrapper script
        script_path = "predict_single.py"
        
        # Create the wrapper script if it doesn't exist
        if not os.path.exists(script_path):
            create_prediction_wrapper()
        
        # Run the prediction
        result = subprocess.run([
            'python3', script_path, temp_file
        ], capture_output=True, text=True, timeout=30)
        
        # Clean up temp file
        os.unlink(temp_file)
        
        if result.returncode != 0:
            logger.error(f"Prediction script error: {result.stderr}")
            raise Exception(f"Prediction failed: {result.stderr}")
        
        # Parse the result
        try:
            prediction_result = json.loads(result.stdout)
            return prediction_result
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON output: {result.stdout}")
            raise Exception("Invalid prediction output")
            
    except subprocess.TimeoutExpired:
        raise Exception("Prediction timed out")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise Exception(f"Prediction failed: {str(e)}")

def create_prediction_wrapper():
    """Create a simple wrapper script for predictions"""
    wrapper_script = '''#!/usr/bin/env python3
"""
Prediction wrapper script
Uses your existing main.py YouTubePredictor class
"""

import sys
import json
from main import YouTubePredictor

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 predict_single.py <input_file.json>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        # Load video data
        with open(input_file, 'r') as f:
            video_data = json.load(f)
        
        # Load your trained model
        predictor = YouTubePredictor()
        predictor.load_model("models/youtube_predictor")
        
        # Make prediction
        pred_views, pred_likes = predictor.predict_single(video_data)
        
        # Calculate metrics
        like_rate = (pred_likes / pred_views * 100) if pred_views > 0 else 0
        duration_minutes = video_data.get('duration_seconds', 600) / 60
        
        # Performance category
        avg_views = predictor.stats.get('view_count_mean', 50000)
        if pred_views > avg_views * 1.5:
            performance = "🚀 Above Average"
            confidence = 0.85
        elif pred_views > avg_views * 0.7:
            performance = "📊 Average"
            confidence = 0.75
        else:
            performance = "📉 Below Average"
            confidence = 0.65
        
        # Format result
        result = {
            "predicted_views": int(pred_views),
            "predicted_likes": int(pred_likes),
            "like_rate": round(like_rate, 2),
            "performance_category": performance,
            "confidence_score": round(confidence * 100, 1),
            "duration_minutes": round(duration_minutes, 1),
            "estimated_comments": max(1, int(pred_likes * 0.1)),
            "estimated_shares": max(1, int(pred_likes * 0.05)),
            "prediction_time": "{}"
        }
        
        # Output JSON result
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
'''.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    with open("predict_single.py", 'w') as f:
        f.write(wrapper_script)
    
    logger.info("✅ Created prediction wrapper script")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_exists = check_model_exists()
    return {
        "status": "healthy" if model_exists else "unhealthy",
        "model_loaded": model_exists,
        "timestamp": datetime.now().isoformat(),
        "message": "Model ready" if model_exists else "Please train model first"
    }

@app.post("/predict")
async def predict_video_performance(video: VideoInput):
    """Predict YouTube video performance using your existing script"""
    
    try:
        logger.info(f"Making prediction for: {video.title[:50]}...")
        
        # Convert to dict
        video_data = video.dict()
        
        # Use your existing script for prediction
        result = predict_with_script(video_data)
        
        # Check for errors in result
        if "error" in result:
            raise Exception(result["error"])
        
        logger.info(f"✅ Prediction: {result['predicted_views']:,} views, {result['predicted_likes']:,} likes")
        return result
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-form")
async def predict_from_form(
    request: Request,
    title: str = Form(...),
    description: str = Form(""),
    transcript_text: str = Form(""),  # Added transcript form field
    duration_seconds: int = Form(600),
    uploader: str = Form("Unknown Channel"),
    categories: str = Form(""),
    tags: str = Form("")
):
    """Handle form submission from HTML page"""
    
    video = VideoInput(
        title=title,
        description=description,
        transcript_text=transcript_text,  # Include transcript
        duration_seconds=duration_seconds,
        uploader=uploader,
        categories=categories,
        tags=tags
    )
    
    try:
        result = await predict_video_performance(video)
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "result": result,
            "video": video.dict()
        })
    except HTTPException as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": e.detail,
            "video": video.dict()
        })

@app.get("/examples")
async def get_examples():
    """Get example videos for testing"""
    examples = [
        {
            "title": "How to Build AI Chatbots in Python - Complete Tutorial",
            "description": "Learn to create your own AI chatbot using Python and OpenAI. Step-by-step guide perfect for beginners!",
            "transcript_text": "Welcome everyone to this complete tutorial on building AI chatbots. Today we'll learn how to create your own intelligent chatbot using Python and OpenAI's powerful API. We'll start with the basics, cover installation, API setup, and then build a working chatbot step by step. This tutorial is perfect for beginners, so don't worry if you're new to programming. By the end, you'll have your own functioning AI assistant. Let's get started!",
            "duration_seconds": 1800,
            "uploader": "CodeWithAI",
            "categories": "Education,Technology",
            "tags": "python,AI,chatbot,tutorial,programming"
        },
        {
            "title": "5 Python Tricks Every Developer Should Know",
            "description": "Quick Python tips that will make you more productive.",
            "transcript_text": "What's up developers! Today I'm sharing 5 Python tricks that will make you way more productive. These are hidden features that most people don't know about. First trick: list comprehensions with conditions. Second: using enumerate instead of range. Third: the amazing zip function. Fourth: dictionary comprehensions. And fifth: the power of lambda functions. Each of these will save you tons of time. Let's dive in!",
            "duration_seconds": 480,
            "uploader": "DevTips",
            "categories": "Education,Technology",
            "tags": "python,programming,tips,coding"
        },
        {
            "title": "Making Perfect Pancakes - Secret Recipe!",
            "description": "The ultimate pancake recipe that will change your breakfast game forever.",
            "transcript_text": "Good morning everyone! Today I'm sharing my secret pancake recipe that will absolutely change your breakfast game. These pancakes are incredibly fluffy, perfectly sweet, and so easy to make. The secret ingredient? A touch of vanilla and the perfect ratio of ingredients. We'll start with the dry ingredients, then the wet, and I'll show you the mixing technique that makes all the difference. Trust me, your family will be asking for these every weekend!",
            "duration_seconds": 600,
            "uploader": "FoodMaster",
            "categories": "Food,Lifestyle",
            "tags": "pancakes,recipe,cooking,breakfast"
        }
    ]
    return {"examples": examples}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)