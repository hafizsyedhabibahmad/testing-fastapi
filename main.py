from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from gradio_client import Client, handle_file
from retry import retry
from PIL import Image, ImageEnhance
import os
import uuid
import tempfile
import hashlib
from cachetools import TTLCache

app = FastAPI()

# Configuration from environment variables
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "static/uploads")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "static/output")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Cache setup (TTL: 1 hour)
cache = TTLCache(maxsize=100, ttl=3600)

def allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file(file_path: str) -> bool:
    """Validate if the file exists and is an image."""
    if not os.path.exists(file_path):
        return False
    if not file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        return False
    return True

def get_file_hash(file_content: bytes) -> str:
    """Generate a hash for a file to use as cache key."""
    return hashlib.sha256(file_content).hexdigest()

def compress_image(content: bytes, max_size: int = 1024) -> bytes:
    """Compress image to reduce size while maintaining quality."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        img = Image.open(temp_file_path)
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            img.save(temp_file.name, "PNG", optimize=True, quality=85)
            with open(temp_file.name, "rb") as f:
                compressed_content = f.read()
            os.unlink(temp_file.name)
        os.unlink(temp_file_path)
        return compressed_content
    except Exception:
        return content

def enhance_image(image_path: str) -> None:
    """Enhance image quality using sharpening."""
    try:
        img = Image.open(image_path)
        enhancer = ImageEnhance.Sharpness(img)
        img_enhanced = enhancer.enhance(2.0)  # Increase sharpness
        img_enhanced.save(image_path, "PNG")
    except Exception:
        pass

def save_output_image(result_path: str, output_dir: str, output_name: str) -> str:
    """Save the result image to a specified directory and enhance it."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_name)
        img = Image.open(result_path)
        img = img.convert("RGB")
        img.save(output_path, "PNG")
        enhance_image(output_path)  # Enhance quality
        return output_path
    except Exception:
        return ""

@retry(tries=3, delay=2, backoff=2)
async def face_swap(source_image: str, dest_image: str, source_face_idx: int = 1, dest_face_idx: int = 1) -> str:
    """Perform face swap using Gradio Client with retry logic."""
    try:
        if not all([validate_file(source_image), validate_file(dest_image)]):
            return "Invalid input files"

        client = Client("Dentro/face-swap")
        result = client.predict(
            sourceImage=handle_file(source_image),
            sourceFaceIndex=source_face_idx,
            destinationImage=handle_file(dest_image),
            destinationFaceIndex=dest_face_idx,
            api_name="/predict"
        )

        if result and os.path.exists(result):
            unique_filename = f"face_swap_{uuid.uuid4().hex}.png"
            final_path = save_output_image(result, OUTPUT_FOLDER, unique_filename)
            if final_path:
                return final_path
            return "Failed to save output"
        return "Face swap failed"
    except Exception as e:
        return f"Error: {str(e)}"

@app.get("/health")
async def health_check():
    """Check if the API is running."""
    return {"status": "API is running"}

@app.post("/swap")
async def swap_faces(source_image: UploadFile = File(...), dest_image: UploadFile = File(...)):
    """Swap faces between two uploaded images."""
    # Validate file uploads
    if not source_image.filename or not dest_image.filename:
        return JSONResponse(status_code=400, content={"error": "No file selected"})

    if not (allowed_file(source_image.filename) and allowed_file(dest_image.filename)):
        return JSONResponse(status_code=400, content={"error": "Invalid file format. Only PNG, JPG, JPEG allowed"})

    # Read and compress file contents
    source_content = await source_image.read()
    dest_content = await dest_image.read()
    source_content = compress_image(source_content)
    dest_content = compress_image(dest_content)

    # Generate cache key
    cache_key = f"{get_file_hash(source_content)}:{get_file_hash(dest_content)}"

    # Check cache
    if cache_key in cache:
        result_url = f"/{cache[cache_key]}"
        return {"result_image": result_url}

    # Save files temporarily
    with tempfile.TemporaryDirectory() as temp_dir:
        source_filename = f"source_{uuid.uuid4().hex}.{source_image.filename.rsplit('.', 1)[1]}"
        dest_filename = f"dest_{uuid.uuid4().hex}.{dest_image.filename.rsplit('.', 1)[1]}"
        source_path = os.path.join(temp_dir, source_filename)
        dest_path = os.path.join(temp_dir, dest_filename)

        # Write compressed files
        with open(source_path, "wb") as f:
            f.write(source_content)
        with open(dest_path, "wb") as f:
            f.write(dest_content)

        # Perform face swap
        result = await face_swap(source_path, dest_path)
        if result.startswith("Error") or result == "Invalid input files" or result == "Failed to save output":
            return JSONResponse(status_code=500, content={"error": result})

        # Cache result
        cache[cache_key] = result

        # Return result
        result_url = f"/{result}"
        return {"result_image": result_url}
