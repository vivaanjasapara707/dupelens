"""Local web app: search any product across Indian online stores, or check one listing.

Serves the dashboard (index.html = live search, evidence.html = how the model was tested)
plus two endpoints that run the trained models:
  POST /api/search  search words -> offers from many stores, grouped into "same item" by the model
  POST /api/check   one listing (title + photo) -> is it already in the 698-listing catalog?
Run: python -m src.app   then open http://127.0.0.1:8000
"""
import threading

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

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
    return {"live": True, "search": search_ready, "catalog": len(matcher.catalog),
            "category": config.INDIA_CATEGORY["amazon"]}


@app.post("/api/search")
def search(q: str = Form(...)):
    """Any product name -> offers from many stores, grouped by the model, cheapest store first."""
    if len(q.strip()) < 2:
        raise HTTPException(400, "Type a product name.")
    try:
        return live.search(matcher, q.strip())
    except RuntimeError as err:  # no Serper key
        raise HTTPException(503, str(err))
    except Exception as err:  # network or Serper error
        raise HTTPException(502, f"Search failed: {err}")


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


# Everything else (index.html, evidence.html, data/*.json) is plain files.
app.mount("/", StaticFiles(directory=DASHBOARD, html=True), name="site")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
