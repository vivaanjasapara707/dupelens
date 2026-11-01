"""DupeLens web app: give it a product name, a store link or a photo; it finds every store selling
that exact item and where it is best to buy.

Serves the dashboard (index.html = live search, evidence.html = how the model was tested) plus:
  POST /api/search  name / link / photo -> offers from many stores, grouped into "same item" by the model
  GET  /api/link    the real product page of one offer at its store
  POST /api/check   one listing (title + photo) -> is it already in the 698-listing test catalog?
Run: python -m src.app   then open http://127.0.0.1:8000
"""
import os
import threading

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from src import config, live
from src.service import Matcher

DASHBOARD = config.ROOT / "dashboard"
app = FastAPI(title="DupeLens")
matcher = Matcher()
# Load the three encoders in the background so the first search does not wait for them.
threading.Thread(target=matcher.encoders, daemon=True).start()


@app.get("/api/health")
def health():
    try:
        live.api_key()
        search_ready = True
    except RuntimeError:
        search_ready = False
    return {"live": True, "search": search_ready, "uploads": bool(os.environ.get("PUBLIC_URL")),
            "catalog": len(matcher.catalog)}


@app.post("/api/search")
async def search(q: str | None = Form(None), image_url: str | None = Form(None),
                 image: UploadFile | None = File(None)):
    """Name, store link, photo address or uploaded photo -> grouped offers (see src/live.py)."""
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
    uvicorn.run(app, host="127.0.0.1", port=8000)
