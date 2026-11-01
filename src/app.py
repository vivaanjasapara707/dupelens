"""DupeLens web app: give it a product name, a store link or a photo; it finds every store selling
that exact item and where it is best to buy.

Serves the dashboard (index.html = live search, evidence.html = how the model was tested) plus:
  POST /api/search  name / link / photo -> offers from many stores, grouped into "same item" by the model
  GET  /api/link    the real product page of one offer at its store
  POST /api/ask     a question about the current search -> answer from a Groq-hosted LLM (src/assistant.py)
  POST /api/xray    Photo X-ray: PCA on a listing's product photos (src/photo_xray.py)
  /api/track, /api/tracked   price tracker: history, buy-now-or-wait advice, alerts (src/tracker.py)
  POST /api/check   one listing (title + photo) -> is it already in the 698-listing test catalog?
Run: python -m src.app   then open http://127.0.0.1:8000
"""
import os
import threading

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from src import assistant, config, live, tracker
from src.service import Matcher

DASHBOARD = config.ROOT / "dashboard"
# Online (Hugging Face Spaces sets SPACE_HOST): uploaded photos get a public address Google Lens can reach.
if os.environ.get("SPACE_HOST") and not os.environ.get("PUBLIC_URL"):
    os.environ["PUBLIC_URL"] = "https://" + os.environ["SPACE_HOST"]
app = FastAPI(title="DupeLens")
matcher = Matcher()
# Load the three encoders in the background so the first search does not wait for them.
threading.Thread(target=matcher.encoders, daemon=True).start()


def note_public_address(request):
    """Behind a tunnel or host (anything but localhost), the address the visitor used is public: uploaded photos
    are saved under it so Google Lens can fetch them. Set once; PUBLIC_URL from the environment wins."""
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or ""
    if not os.environ.get("PUBLIC_URL") and host and not host.split(":")[0] in ("127.0.0.1", "localhost", "0.0.0.0"):
        os.environ["PUBLIC_URL"] = f"https://{host}"


@app.get("/api/health")
def health(request: Request):
    note_public_address(request)
    try:
        live.api_key()
        search_ready = True
    except RuntimeError:
        search_ready = False
    return {"live": True, "search": search_ready, "uploads": bool(os.environ.get("PUBLIC_URL")),
            "assistant": bool(config.env("GROQ_API_KEY")), "tracker": True, "catalog": len(matcher.catalog)}


class Track(BaseModel):
    query: str
    title: str
    image: str | None = None
    target: float | None = None
    snapshot: dict


@app.post("/api/track")
def track(t: Track):
    """Start tracking an item; the page sends today's prices of the item it is showing."""
    return {"id": tracker.add(t.query, t.title, t.image, t.target, t.snapshot)}


class XrayRequest(BaseModel):
    images: list[str]


def run_xray(urls):
    import io
    from PIL import Image
    from src.photo_xray import xray
    datas = [live.cached_download(u) for u in urls]
    keep = [i for i, d in enumerate(datas) if d]
    if len(keep) < 3:
        raise ValueError("Need at least 3 photos for the Photo X-ray.")
    ims = [Image.open(io.BytesIO(datas[i])).convert("RGB") for i in keep]
    out = xray(ims, dino=matcher.encoders()[2](ims))
    out["index"] = keep  # positions in the request that had a photo
    return out


@app.post("/api/xray")
async def photo_xray(r: XrayRequest):
    """PCA on one model's listing photos: mean photo, eigen-images, rebuilt photos, residual heatmaps, odd-one-out."""
    try:
        return await run_in_threadpool(run_xray, r.images[:30])
    except ValueError as err:
        raise HTTPException(400, str(err))


@app.get("/api/tracked")
def tracked():
    return tracker.listing()


@app.delete("/api/tracked/{tid}")
def untrack(tid: int):
    tracker.remove(tid)
    return {"ok": True}


@app.post("/api/tracked/update")
async def update_prices():
    """Check every tracked product now (one fresh search each, at most once per day per product)."""
    return {"report": await run_in_threadpool(tracker.update, matcher)}


class Question(BaseModel):
    question: str
    context: dict
    history: list = []


@app.post("/api/ask")
async def ask(q: Question):
    """Shopping question about the search on screen -> short answer grounded in that search's data."""
    if not q.question.strip():
        raise HTTPException(400, "Type a question.")
    try:
        return await run_in_threadpool(assistant.answer, q.question.strip()[:500], q.context, q.history)
    except RuntimeError as err:  # no Groq key
        raise HTTPException(503, str(err))
    except Exception as err:  # network or Groq error
        raise HTTPException(502, f"The AI assistant failed: {err}")


@app.post("/api/search")
async def search(request: Request, q: str | None = Form(None), image_url: str | None = Form(None),
                 image: UploadFile | None = File(None)):
    """Name, store link, photo address or uploaded photo -> grouped offers (see src/live.py)."""
    note_public_address(request)
    data = await image.read() if image is not None and image.filename else None
    try:
        # run in a worker thread so a slow search does not freeze the page for other visitors
        return await run_in_threadpool(live.search, matcher, q, image_bytes=data, image_url=image_url or None)
    except ValueError as err:  # input we could not understand: say what to do instead
        raise HTTPException(400, str(err))
    except RuntimeError as err:  # no Serper key
        raise HTTPException(503, str(err))
    except Exception as err:  # network or Serper error
        raise HTTPException(502, f"Search failed: {err}")


@app.get("/api/link")
def link(title: str, store: str):
    try:
        return {"url": live.store_link(title, store)}
    except Exception as err:
        raise HTTPException(502, f"Could not find the store page: {err}")


@app.post("/api/check")
async def check(title: str = Form(...), price: float | None = Form(None),
                image_url: str | None = Form(None), image: UploadFile | None = File(None)):
    """Title + (photo file or photo URL) [+ price] -> verdict, matching listings, typical price."""
    data = await image.read() if image is not None and image.filename else None
    if not title.strip() or (data is None and not image_url):
        raise HTTPException(400, "Give a title and a photo (upload or image URL).")
    try:
        return matcher.check_new(title.strip(), image_bytes=data, image_url=image_url, price=price)
    except Exception as err:  # bad image URL, unreadable file
        raise HTTPException(422, f"Could not read that photo: {err}")


# Uploaded photos (online version only) must be reachable by Google Lens.
(config.CACHE_DIR / "uploads").mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=config.CACHE_DIR / "uploads"), name="uploads")
# Everything else (index.html, evidence.html, data/*.json) is plain files.
app.mount("/", StaticFiles(directory=DASHBOARD, html=True), name="site")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # hosting platforms set PORT and need 0.0.0.0
    uvicorn.run(app, host="0.0.0.0" if "PORT" in os.environ else "127.0.0.1", port=port)
