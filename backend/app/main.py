import os
import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Sketch2Vision API")
app.state.demo_service = None
app.state.demo_model_error = "The demo generator has not loaded."
app.state.description_service = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check() -> dict[str, str]:
    return {
        "status": "running",
        "sketch_description": "ready" if app.state.description_service else "unavailable",
        "experimental_facades_pix2pix": "ready" if app.state.demo_service else "unavailable",
    }


@app.on_event("startup")
def load_services_once() -> None:
    """Load environment configuration and optional providers once at startup."""
    dotenv_path = Path(__file__).resolve().parents[1] / ".env"
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=dotenv_path, override=False)
    except ImportError:
        if dotenv_path.is_file() and not os.environ.get("GROQ_API_KEY"):
            print("python-dotenv is unavailable; set GROQ_API_KEY in the backend process environment.")

    try:
        from backend.app.providers.groq_sketch_description import GroqSketchDescription
    except ModuleNotFoundError:
        from app.providers.groq_sketch_description import GroqSketchDescription
    try:
        app.state.description_service = GroqSketchDescription.from_environment()
        print(f"Groq sketch description provider configured ({app.state.description_service.model}).")
    except Exception as exc:
        app.state.description_service = None
        print(f"Groq sketch description provider unavailable: {exc}")

    checkpoint = Path(os.environ.get(
        "SKETCH2VISION_DEMO_CHECKPOINT",
        str(Path(__file__).resolve().parents[2] / "demo_points" / "best.pt"),
    ))
    try:
        try:
            from backend.ml.inference import FacadesDemoInference
        except ModuleNotFoundError:
            # Also support the documented launch from inside backend/.
            from ml.inference import FacadesDemoInference

        app.state.demo_service = FacadesDemoInference.from_checkpoint(checkpoint, device="auto")
        app.state.demo_model_error = ""
        print(f"Experimental Facades demo model loaded on {app.state.demo_service.device}.")
    except Exception as exc:
        app.state.demo_service = None
        app.state.demo_model_error = str(exc)
        print(f"Experimental Facades demo model unavailable: {exc}")


def _validate_image_request(image_bytes: bytes, content_type: str) -> None:
    content_type = content_type.split(";", 1)[0].strip().lower()
    if content_type not in {"image/png", "image/jpeg", "application/octet-stream"}:
        raise HTTPException(status_code=415, detail="Send a PNG or JPEG sketch image.")
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Sketch image exceeds the 10 MB limit.")
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The sketch upload is empty.")


def generate_image_response(image_bytes: bytes, content_type: str, service: object | None) -> Response:
    _validate_image_request(image_bytes, content_type)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="The experimental Facade Demo model is unavailable. Check that demo_points/best.pt and inference dependencies are present, then restart the backend.",
        )
    try:
        png = service.generate_png(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        print(f"Demo inference failed: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail="Image generation failed. Check the backend console and try again.") from exc
    return Response(content=png, media_type="image/png", headers={"Cache-Control": "no-store"})


def describe_sketch_response(image_bytes: bytes, content_type: str, service: object | None) -> dict[str, str]:
    _validate_image_request(image_bytes, content_type)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Sketch description is not configured. Add GROQ_API_KEY to backend/.env and restart the backend.",
        )
    try:
        description = service.describe(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        status_code = getattr(exc, "status_code", 502)
        retry_after = getattr(exc, "retry_after", None)
        headers = {"Retry-After": retry_after} if retry_after else None
        if status_code == 429:
            detail = "Groq is rate-limiting requests. Wait, then try again."
        else:
            detail = str(exc)[:400] or "Sketch description failed. Try again."
        raise HTTPException(status_code=status_code, detail=detail, headers=headers) from exc
    return {"description": description}


@app.post("/api/describe")
async def describe_sketch(request: Request) -> dict[str, str]:
    declared_length = request.headers.get("content-length")
    if declared_length:
        try:
            if int(declared_length) > 10 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="Sketch image exceeds the 10 MB limit.")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header.") from exc
    image_bytes = await request.body()
    return await asyncio.to_thread(describe_sketch_response, image_bytes,
                                   request.headers.get("content-type", ""), app.state.description_service)


@app.post("/api/generate/facades")
async def generate_facades_image(request: Request) -> Response:
    declared_length = request.headers.get("content-length")
    if declared_length:
        try:
            if int(declared_length) > 10 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="Sketch image exceeds the 10 MB limit.")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header.") from exc
    image_bytes = await request.body()
    return await asyncio.to_thread(generate_image_response, image_bytes, request.headers.get("content-type", ""),
                                   app.state.demo_service)
