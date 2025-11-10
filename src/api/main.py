from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from PIL import Image, ImageOps
import io
from src.inference.colorizer import Colorizer
from src.config import CONFIG
import numpy as np
from fastapi.responses import Response

app = FastAPI()

colorizer = Colorizer(CONFIG["generator_path"], CONFIG["device"])


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = Path(__file__).parent.parent / "api" / "static" / "index.html"
    print(index_path)
    if not index_path.exists():
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = ImageOps.exif_transpose(img)
    img.thumbnail((512, 512))

    img_array = colorizer(img)

    img_uint8 = (img_array * 255).clip(0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_uint8)

    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")
