import os
import uuid
import logging
import shutil
import requests
from datetime import datetime, timedelta
from typing import Optional
from PIL import Image  # For dimension detection & resize

from fastapi import (
    FastAPI, UploadFile, File,
    Depends, HTTPException, Request, Form
)
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer

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
MONGO_URI = os.getenv("MONGO_URI")  # MongoDB Atlas URI (mongodb+srv://...)
JWT_SECRET = os.getenv("JWT_SECRET_KEY", "super-secret-key")
CLIPDROP_API_KEY = os.getenv("CLIPDROP_API_KEY", "")
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
        tls=True,                 # force TLS/SSL
        tlsCAFile=certifi.where(),# proper CA bundle
        serverSelectionTimeoutMS=10000  # 10 seconds timeout
    )
    db = client["image_ai"]
    users_col = db["users"]
    # Test connection immediately
    client.admin.command('ping')
    logger.info("MongoDB connected successfully")
except Exception as e:
    logger.error(f"MongoDB connection failed: {e}")
    raise e

# =========================================================
# FASTAPI APP
# =========================================================
app = FastAPI(title="ImageAI Pro API"
              # docs_url= None,
              # redoc_url=None,
              # openapi_url=None)
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

    if user.get("expires_at") and datetime.utcnow() > user["expires_at"]:
        raise HTTPException(status_code=403, detail="Subscription expired")

    if user.get("images_used", 0) >= user.get("image_limit", 0):
        raise HTTPException(status_code=403, detail="Image quota reached")

    return user

# =========================================================
# ROUTES
# =========================================================
@app.get("/", response_class=HTMLResponse)
async def chat_ui(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

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
    }

# -------------------- IMAGE UPLOAD --------------------
@app.post("/images/upload")
async def upload_image(file: UploadFile = File(...), user: dict = Depends(get_current_user)):

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".png", ".jpg", ".jpeg", ".jfif"]:
        raise HTTPException(status_code=400, detail="Only PNG or JPG images are supported")

    image_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{image_id}{ext}")
    ai_path = os.path.join(UPLOAD_DIR, f"{image_id}_ai.png")
    output_path = os.path.join(OUTPUT_DIR, f"{image_id}_processed.png")

    # Save uploaded file
    contents = await file.read()
    with open(input_path, "wb") as f:
        f.write(contents)

    # Detect original size
    with Image.open(input_path) as img:
        orig_width, orig_height = img.size

    logger.info(f"Original image size → {orig_width}x{orig_height}")

    processed_source = input_path
    ai_used = False

    # =====================================================
    # ClipDrop Upscaling (8K → fallback 4K → resize back)
    # =====================================================
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

    # Resize back to original size
    with Image.open(processed_source) as img:
        logger.info(f"Processed image size → {img.size[0]}x{img.size[1]}")
        final_img = img.resize((orig_width, orig_height), Image.LANCZOS)
        final_img.save(output_path, format="PNG")

    logger.info(
        f"Final returned size → {orig_width}x{orig_height} | AI used: {ai_used}"
    )

    users_col.update_one({"email": user["email"]}, {"$inc": {"images_used": 1}})

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
            "email": "waris12345@gmail.com",
            "password_hash": hash_password("123W"),
            "password_set_at": datetime.utcnow(),
            "password_expires_at": datetime.utcnow() + timedelta(days=1),
            "image_limit": 233,
            "images_used": 0,
            "active": True,
            "expires_at": datetime.utcnow() + timedelta(days=34)
        },
        {
            "email": "user4@gmail.com",
            "password_hash": hash_password("123W"),
            "password_set_at": datetime.utcnow(),
            "password_expires_at": datetime.utcnow() + timedelta(days=1),
            "image_limit": 10,
            "images_used": 0,
            "active": True,
            "expires_at": datetime.utcnow() + timedelta(days=34)
        },
        {
            "email": "user5@gmail.com",
            "password_hash": hash_password("1234W"),
            "password_set_at": datetime.utcnow(),
            "password_expires_at": datetime.utcnow() + timedelta(days=1),
            "image_limit": 10,
            "images_used": 0,
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

# ---------------------- Renew user --------------------
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

    update = {"expires_at": datetime.utcnow() + timedelta(days=extra_days), "active": True}
    if reset_counter:
        update["images_used"] = 0
    if new_image_limit is not None:
        update["image_limit"] = new_image_limit

    users_col.update_one({"email": email}, {"$set": update})
    return {"message": "User renewed successfully"}
