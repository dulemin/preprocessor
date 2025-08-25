from fastapi import FastAPI, UploadFile, File, Response
from app.pipelines.photo import process_photo
from app.pipelines.scan import process_scan

app = FastAPI()

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/foto")
def endpoint_foto(file: UploadFile = File(...)):
    return Response(process_photo(file.file.read()), media_type="image/png")

@app.post("/scan")
def endpoint_scan(file: UploadFile = File(...)):
    return Response(process_scan(file.file.read()), media_type="image/png")
