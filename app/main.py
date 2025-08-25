from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from .preprocess import preprocess_image

app = FastAPI(title="OCR Preprocessor", version="1.0.0")

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/preprocess")
async def preprocess(file: UploadFile = File(...)):
    try:
        data = await file.read()
        processed = preprocess_image(data)
        return StreamingResponse(iter([processed]), media_type="image/png")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
