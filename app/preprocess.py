from fastapi import FastAPI, UploadFile, File, Response
from app.pipelines.photo import process_photo
from app.pipelines.scan import process_scan
from app.preprocess import preprocess_image as process_default

app = FastAPI()

@app.post("/foto")
def endpoint_foto(file: UploadFile = File(...)):
    return Response(process_photo(file.file.read()), media_type="image/png")

@app.post("/scan")
def endpoint_scan(file: UploadFile = File(...)):
    return Response(process_scan(file.file.read()), media_type="image/png")

@app.post("/preprocess")
def endpoint_preprocess(file: UploadFile = File(...)):
    return Response(process_default(file.file.read()), media_type="image/png")
