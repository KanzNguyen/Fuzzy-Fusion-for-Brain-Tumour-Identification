from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from inference import inference 

app = FastAPI(
    title="Brain Tumor Classification API",
    description="Phân loại u não từ ảnh MRI — hỗ trợ upload file hoặc URL.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_METHODS  = {"SVM", "Gompertz", "Mitscherlich"}
ALLOWED_MIMETYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

class InferenceResponse(BaseModel):
    method:          str
    prediction:      str
    predicted_class: int
    confidence:      float

@app.post(
    "/predict/upload",
    response_model=InferenceResponse,
    summary="Dự đoán từ file ảnh upload",
)
async def predict_upload(
    file:   UploadFile = File(...,  description="File ảnh MRI (JPEG / PNG / WebP / BMP)"),
    method: str        = Form("Gompertz", description="SVM | Gompertz | Mitscherlich"),
):
    _validate_method(method)

    if file.content_type not in ALLOWED_MIMETYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Định dạng không hỗ trợ: {file.content_type}. "
                   f"Chấp nhận: {sorted(ALLOWED_MIMETYPES)}",
        )

    image_bytes = await file.read()
    try:
        result = inference(image_bytes, method)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result

class URLRequest(BaseModel):
    url:    str
    method: str = "Gompertz"

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "url":    "https://example.com/brain_mri.jpg",
                "method": "Gompertz",
            }]
        }
    }

@app.post(
    "/predict/url",
    response_model=InferenceResponse,
    summary="Dự đoán từ URL ảnh",
)
async def predict_url(body: URLRequest):
    _validate_method(body.method)

    if not (body.url.startswith("http://") or body.url.startswith("https://")):
        raise HTTPException(status_code=400, detail="URL phải bắt đầu bằng http:// hoặc https://")

    try:
        result = inference(body.url, body.method)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result

# 3) Endpoint kết hợp (multipart: truyền url hoặc file, tuỳ 1 trong 2)
@app.post(
    "/predict",
    response_model=InferenceResponse,
    summary="Dự đoán — tự động chọn file hoặc URL",
)
async def predict(
    method: str                  = Form("Gompertz", description="SVM | Gompertz | Mitscherlich"),
    file:   Optional[UploadFile] = File(None,        description="File ảnh (tuỳ chọn)"),
    url:    Optional[str]        = Form(None,         description="URL ảnh (tuỳ chọn)"),
):
    _validate_method(method)

    if file is None and url is None:
        raise HTTPException(status_code=422, detail="Phải cung cấp 'file' hoặc 'url'.")
    if file is not None and url is not None:
        raise HTTPException(status_code=422, detail="Chỉ cung cấp 'file' hoặc 'url', không dùng cả hai.")

    try:
        if file is not None:
            if file.content_type not in ALLOWED_MIMETYPES:
                raise HTTPException(
                    status_code=415,
                    detail=f"Định dạng không hỗ trợ: {file.content_type}.",
                )
            image_source = await file.read()
        else:
            if not (url.startswith("http://") or url.startswith("https://")):
                raise HTTPException(status_code=400, detail="URL không hợp lệ.")
            image_source = url

        result = inference(image_source, method)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health", summary="Kiểm tra trạng thái server")
async def health():
    return {"status": "ok"}


# ── Util ───────────────────────────────────────────────────────────────────────
def _validate_method(method: str):
    if method not in ALLOWED_METHODS:
        raise HTTPException(
            status_code=400,
            detail=f"method '{method}' không hợp lệ. Chọn một trong: {sorted(ALLOWED_METHODS)}",
        )

# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)