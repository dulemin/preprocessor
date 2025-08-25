# OCR Preprocessor

FastAPI-Service, der Kassenbon-Bilder für OCR vorverarbeitet (Grayscale, Deskew, Adaptive Threshold, Schärfen).

## Lokaler Test
```bash
docker build -t ocr-preprocessor:dev .
docker run --rm -p 8000:8000 ocr-preprocessor:dev
curl -X POST http://localhost:8000/preprocess -F "file=@/pfad/zum/bild.jpg" --output out.png
```
