import os
import uuid
import logging
import shutil
import requests
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from PIL import Image  # For dimension detection & resize

from fastapi import (
    FastAPI, UploadFile, File,
    Depends, HTTPException, Request, Form, WebSocket, WebSocketDisconnect
)
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer
from openai import OpenAI

from jose import jwt, JWTError
from passlib.context import CryptContext
from dotenv import load_dotenv
import certifi
from pymongo import MongoClient

# =========================================================
# LOAD ENV + LOGGING
# =========================================================
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ImageAI_Pro")

# =========================================================
# CONFIG
# =========================================================
MONGO_URI = os.getenv("MONGO_URI", "").strip()  # MongoDB Atlas URI (mongodb+srv://...)

JWT_SECRET = os.getenv("JWT_SECRET_KEY")
if not JWT_SECRET:
    raise ValueError("JWT_SECRET_KEY is not set")

CLIPDROP_API_KEY = os.getenv("CLIPDROP_API_KEY")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")

ALGORITHM = "HS256"

# =========================================================
# PASSWORD HASHING
# =========================================================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
MAX_BCRYPT_LEN = 72

def hash_password(password: str) -> str:
    truncated = password[:MAX_BCRYPT_LEN]
    return pwd_context.hash(truncated)

def verify_password(password: str, hashed: str) -> bool:
    truncated = password[:MAX_BCRYPT_LEN]
    return pwd_context.verify(truncated, hashed)

# =========================================================
# DATABASE
# =========================================================
try:
    client = MongoClient(
        MONGO_URI,
        tls=True,                  # force TLS/SSL
        tlsCAFile=certifi.where(), # proper CA bundle
        serverSelectionTimeoutMS=10000  # 10 seconds timeout
    )
    db = client["image_ai"]
    users_col = db["users"]
    image_history_col = db["image_history"]
    # Test connection immediately
    client.admin.command("ping")
    logger.info("MongoDB connected successfully")
except Exception as e:
    logger.error(f"MongoDB connection failed: {e}")
    raise e

# =========================================================
# FASTAPI APP
# =========================================================
app = FastAPI(title="ImageAI Pro API"
            #   docs_url= None,
            #   redoc_url=None,
            #   openapi_url=None
             )

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app.mount("/processed", StaticFiles(directory=OUTPUT_DIR), name="processed")
templates = Jinja2Templates(directory="templates")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# =========================================================
# JWT HELPERS
# =========================================================
def create_access_token(data: dict, hours: int = 12):
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(hours=hours)
    return jwt.encode(payload, JWT_SECRET, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Dependency to authenticate user. 
    Quota checks are now moved to specific routes to allow independent usage.
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = users_col.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Time-based expiration check remains global
    if user.get("expires_at") and datetime.utcnow() > user["expires_at"]:
        raise HTTPException(status_code=403, detail="Subscription expired")

    return user

# =========================================================
# REAL-TIME CONNECTION MANAGER (WEBSOCKETS)
# =========================================================
class ConnectionManager:
    """Manages active WebSocket connections for real-time notifications."""
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, email: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[email] = websocket
        logger.info(f"WebSocket: User {email} connected. Active: {len(self.active_connections)}")

    def disconnect(self, email: str):
        if email in self.active_connections:
            del self.active_connections[email]
            logger.info(f"WebSocket: User {email} disconnected.")

    async def send_personal_message(self, message: dict, email: str):
        websocket = self.active_connections.get(email)
        if websocket:
            await websocket.send_json(message)

manager = ConnectionManager()

# =========================================================
# ROUTES
# =========================================================
@app.get("/", response_class=HTMLResponse)
async def chat_ui(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/app", response_class=HTMLResponse)
async def app_interface(request: Request):
    return templates.TemplateResponse("page.html", {"request": request})

# -------------------- AUTH --------------------
@app.post("/login", tags=["Auth"])
async def login(email: str = Form(...), password: str = Form(...)):
    user = users_col.find_one({"email": email})
    if not user or not verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": email})
    return {"access_token": token, "token_type": "bearer"}

@app.get("/me", tags=["User"])
async def me(user: dict = Depends(get_current_user)):
    return {
        "email": user["email"],
        "image_limit": user["image_limit"],
        "images_used": user["images_used"],
        "expires_at": user["expires_at"],
        "image_generations_limit": user["image_generations_limit"],
        "image_generations_used": user["image_generations_used"],
        "background_removals_limit": user.get("background_removals_limit", 0),
        "background_removals_used": user.get("background_removals_used", 0)
    }

# -------------------- IMAGE UPLOAD (UPSCALING) --------------------
@app.post("/images/upload")
async def upload_image(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user)
):
    # CRITICAL CHANGE: Upscaling quota check is now isolated here
    if user.get("images_used", 0) >= user.get("image_limit", 0):
        raise HTTPException(status_code=403, detail="Upscaling quota reached. Text-to-Image might still be available.")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".png", ".jpg", ".jpeg", ".jfif"]:
        raise HTTPException(status_code=400, detail="Only PNG or JPG images are supported")

    image_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{image_id}{ext}")
    ai_path = os.path.join(UPLOAD_DIR, f"{image_id}_ai.png")
    output_path = os.path.join(OUTPUT_DIR, f"{image_id}_processed.png")

    contents = await file.read()
    with open(input_path, "wb") as f:
        f.write(contents)

    with Image.open(input_path) as img:
        orig_width, orig_height = img.size

    logger.info(f"Original image size → {orig_width}x{orig_height}")

    processed_source = input_path
    ai_used = False

    if CLIPDROP_API_KEY:
        try:
            logger.info("ClipDrop detected → Trying 8K upscale")

            with open(input_path, "rb") as img_file:
                response = requests.post(
                    "https://clipdrop-api.co/image-upscaling/v1/upscale",
                    files={"image_file": img_file},
                    data={"target_width": "7680", "target_height": "4320"},
                    headers={"x-api-key": CLIPDROP_API_KEY},
                    timeout=300
                )

            if response.status_code != 200:
                logger.warning("8K failed → Trying 4K")
                with open(input_path, "rb") as img_file:
                    response = requests.post(
                        "https://clipdrop-api.co/image-upscaling/v1/upscale",
                        files={"image_file": img_file},
                        data={"target_width": "3840", "target_height": "2160"},
                        headers={"x-api-key": CLIPDROP_API_KEY},
                        timeout=180
                    )

            if response.status_code == 200:
                with open(ai_path, "wb") as f:
                    f.write(response.content)
                processed_source = ai_path
                ai_used = True
                logger.info("ClipDrop processing successful")
            else:
                logger.error("ClipDrop failed → Serving original")

        except Exception as e:
            logger.error(f"ClipDrop API error: {e}")

    else:
        logger.warning("CLIPDROP_API_KEY missing → Serving original image")

    with Image.open(processed_source) as img:
        logger.info(f"Processed image size → {img.size[0]}x{img.size[1]}")
        final_img = img.resize((orig_width, orig_height), Image.LANCZOS)
        final_img.save(output_path, format="PNG")

    logger.info(
        f"Final returned size → {orig_width}x{orig_height} | AI used: {ai_used}"
    )

    users_col.update_one(
        {"email": user["email"]},
        {"$inc": {"images_used": 1}}
    )

    # Save to image history
    image_history_col.insert_one({
        "email": user["email"],
        "type": "upscale",
        "original_filename": file.filename,
        "processed_filename": f"{image_id}_processed.png",
        "prompt": None,
        "created_at": datetime.utcnow(),
        "ai_used": ai_used
    })

    return {
        "status": "success",
        "ai_used": ai_used,
        "download_url": f"/processed/{image_id}_processed.png"
    }

@app.get("/download/{filename}")
def download_image(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"error": "File not found"}

# -------------------- ADMIN --------------------
@app.post("/admin/seed", tags=["Admin"])
def seed_users():
    users = [
        {
            "email": "waris123@gmail.com",
            "password_hash": hash_password("123"),
            "password_set_at": datetime.utcnow(),
            "password_expires_at": datetime.utcnow() + timedelta(days=1),
            "image_limit": 233,
            "images_used": 0,
            "image_generations_limit": 35,
            "image_generations_used": 0,
            "active": True,
            "expires_at": datetime.utcnow() + timedelta(days=34)
        }       
    ]

    added = 0
    for u in users:
        if not users_col.find_one({"email": u["email"]}):
            users_col.insert_one(u)
            added += 1
    return {"message": f"{added} users added"}

# -------------------- PASSWORD RESET --------------------
@app.post("/reset-password", tags=["Auth"])
def reset_password(email: str, new_password: str):
    users_col.update_one(
        {"email": email},
        {"$set": {
            "password_hash": hash_password(new_password),
            "password_set_at": datetime.utcnow(),
            "password_expires_at": datetime.utcnow() + timedelta(days=90)
        }}
    )
    return {"message": "Password reset successful"}

# -------------------- Renew user --------------------
@app.post("/admin/renew", tags=["Admin"])
def renew_user(
    email: str,
    extra_days: int = 30,
    reset_counter: bool = True,
    new_image_limit: Optional[int] = None
):
    user = users_col.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    update = {
        "expires_at": datetime.utcnow() + timedelta(days=extra_days),
        "active": True
    }
    if reset_counter:
        update["images_used"] = 0
    if new_image_limit is not None:
        update["image_limit"] = new_image_limit

    users_col.update_one({"email": email}, {"$set": update})
    return {"message": "User renewed successfully"}


# =========================================================
# FEATURE 2: TEXT-TO-IMAGE (OPENAI DALL-E) - FIXED
# =========================================================
@app.post("/images/generate", tags=["AI Features"])
async def generate_from_text(
    prompt: str = Form(...),
    user: dict = Depends(get_current_user)
):
    """Generates an image from prompt using OpenAI and counts against quota."""
    
    # 1. Check Configuration
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI service not configured")

    # 2. Quota Check (Isolated to Text-to-Image)
    if user.get("image_generations_used", 0) >= user.get("image_generations_limit", 10):
        raise HTTPException(status_code=403, detail="Neural generation quota exceeded. Upscaling might still be available.")

    logger.info(f"User {user['email']} generating image: {prompt[:50]}...")

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        
        image_url = response.data[0].url
        
        # 4. Download Image
        img_res = requests.get(image_url)
        if img_res.status_code != 200:
            raise Exception("Failed to download image from OpenAI servers")
        
        img_data = img_res.content
        
        # 5. Ensure Directory Exists
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        image_id = str(uuid.uuid4())
        filename = f"{image_id}_generated.png"
        file_path = os.path.join(OUTPUT_DIR, filename)

        with open(file_path, "wb") as f:
            f.write(img_data)

        # 6. Log usage
        users_col.update_one(
            {"email": user["email"]},
            {"$inc": {"image_generations_used": 1}} 
        )

        # Save to image history
        image_history_col.insert_one({
            "email": user["email"],
            "type": "generation",
            "original_filename": None,
            "processed_filename": filename,
            "prompt": prompt,
            "created_at": datetime.utcnow(),
            "ai_used": None
        })

        return {
            "status": "success",
            "type": "generation",
            "prompt": prompt,
            "download_url": f"/processed/{filename}"
        }

    except Exception as e:
        logger.error(f"OpenAI generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    
# =========================================================
# REAL-TIME WEBSOCKET ENDPOINT
# =========================================================
@app.websocket("/ws/{token}")
async def websocket_endpoint(websocket: WebSocket, token: str):
    """WebSocket for real-time status updates."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if not email:
            await websocket.close(code=4001)
            return
    except:
        await websocket.close(code=4001)
        return

    await manager.connect(email, websocket)
    try:
        while True:
            await websocket.receive_text()
            await manager.send_personal_message({"type": "pong", "data": "alive"}, email)
    except WebSocketDisconnect:
        manager.disconnect(email)
    except Exception as e:
        logger.error(f"WebSocket error for {email}: {e}")
        manager.disconnect(email)
        

# -------------------- Renew user (Alternative endpoint) --------------------
@app.post("/image/renew", tags=["Admin"])
def renew_user_alt(
    email: str,
    extra_days: int = 30,
    reset_counter: bool = True,
    new_image_limit: Optional[int] = None
):
    user = users_col.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    update = {
        "expires_at": datetime.utcnow() + timedelta(days=extra_days),
        "active": True
    }
    if reset_counter:
        update["image_generations_used"] = 0
    if new_image_limit is not None:
        update["image_generations_limit"] = new_image_limit

    users_col.update_one({"email": email}, {"$set": update})
    return {"message": "User renewed successfully"}


# =========================================================
# FEATURE 3: BACKGROUND REMOVAL (CLIPDROP)
# =========================================================
@app.post("/images/remove-background", tags=["AI Features"])
async def remove_background(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user)
):
    """Removes image background using ClipDrop API and tracks quota independently."""

    # 1. Quota check (isolated to background removal)
    if user.get("background_removals_used", 0) >= user.get("background_removals_limit", 0):
        raise HTTPException(
            status_code=403,
            detail="Background removal quota reached. Other features might still be available."
        )

    # 2. Validate file type
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".png", ".jpg", ".jpeg", ".jfif"]:
        raise HTTPException(status_code=400, detail="Only PNG or JPG images are supported")

    # 3. Save uploaded file
    image_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{image_id}{ext}")
    output_filename = f"{image_id}_bg_removed.png"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    contents = await file.read()
    with open(input_path, "wb") as f:
        f.write(contents)

    logger.info(f"User {user['email']} → Background removal started for {file.filename}")

    # 4. Call ClipDrop Remove Background API
    if not CLIPDROP_API_KEY:
        raise HTTPException(status_code=500, detail="ClipDrop service not configured")

    try:
        with open(input_path, "rb") as img_file:
            response = requests.post(
                "https://clipdrop-api.co/remove-background/v1",
                files={"image_file": (file.filename, img_file, f"image/{ext.lstrip('.')}")},
                headers={"x-api-key": CLIPDROP_API_KEY},
                timeout=120
            )

        if response.status_code != 200:
            logger.error(f"ClipDrop remove-background failed: {response.status_code} {response.text}")
            raise HTTPException(status_code=502, detail="Background removal API failed")

        with open(output_path, "wb") as f:
            f.write(response.content)

        logger.info(f"Background removal successful → {output_filename}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ClipDrop remove-background error: {e}")
        raise HTTPException(status_code=500, detail=f"Background removal failed: {str(e)}")

    # 5. Increment quota
    users_col.update_one(
        {"email": user["email"]},
        {"$inc": {"background_removals_used": 1}}
    )

    # 6. Save to image history
    image_history_col.insert_one({
        "email": user["email"],
        "type": "background_removal",
        "original_filename": file.filename,
        "processed_filename": output_filename,
        "prompt": None,
        "created_at": datetime.utcnow(),
        "ai_used": None
    })

    # 7. Optional WebSocket notification
    await manager.send_personal_message(
        {"type": "background_removed", "data": f"/processed/{output_filename}"},
        user["email"]
    )

    return {
        "status": "success",
        "type": "background_removal",
        "download_url": f"/processed/{output_filename}"
    }


# =========================================================
# USER IMAGE HISTORY
# =========================================================
@app.get("/images/history", tags=["AI Features"])
async def get_image_history(user: dict = Depends(get_current_user)):
    """Returns all processed/generated images for the logged-in user, newest first."""

    records = image_history_col.find(
        {"email": user["email"]},
        {"_id": 0}
    ).sort("created_at", -1)

    history = []
    for record in records:
        history.append({
            "type": record.get("type"),
            "download_url": f"/processed/{record['processed_filename']}" if record.get("processed_filename") else None,
            "created_at": record.get("created_at").isoformat() if record.get("created_at") else None,
            "prompt": record.get("prompt"),
            "ai_used": record.get("ai_used")
        })

    logger.info(f"History fetched for {user['email']} → {len(history)} records")
    return history


@app.post("/background/renew", tags=["Admin"])
def renew_background_quota(
    email: str,
    extra_days: int = 30,
    reset_counter: bool = True,
    new_limit: Optional[int] = None
):
    user = users_col.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    update = {
        "expires_at": datetime.utcnow() + timedelta(days=extra_days),
        "active": True
    }

    # Reset usage
    if reset_counter:
        update["background_removals_used"] = 0

    # Update limit if provided
    if new_limit is not None:
        update["background_removals_limit"] = new_limit

    users_col.update_one({"email": email}, {"$set": update})

    return {
        "message": "Background removal quota renewed successfully"
    }
