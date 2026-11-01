"""Price tracker: price history, "buy now or wait" advice, and price-drop alerts.

Tracking a product stores its search words and the item's title. Once a day (`python -m src.tracker --update`,
run by a Windows scheduled task, see scripts/) every tracked search is run again with a fresh Serper answer
for that date; the model finds the same item again (the offer whose title matches best, plus every offer the
model calls the same item) and the day's lowest and median price are stored.

Advice is a simple, explainable rule on the recorded history (not a trained forecaster):
  fewer than 3 days      -> "collecting history"
  at or near the low     -> buy now (lowest price in N days)
  falling trend          -> wait (least-squares slope of the daily lowest price over the last 14 days)
  otherwise              -> how far above the N-day low it is
With 7+ days, a 7-day straight-line projection is shown.
Alerts: price at or below the user's target, or a drop of 5% or more since the last check. Shown in the app;
also sent to Telegram when TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set in .env.
Data: data/tracker.db (SQLite; data/ is not committed).
"""
import datetime as dt
import json
import sqlite3
import sys

import numpy as np
import requests

from src import config

DB = config.DATA_DIR / "tracker.db"


def db():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    con.executescript("""
        CREATE TABLE IF NOT EXISTS tracked (id INTEGER PRIMARY KEY, query TEXT, title TEXT, image TEXT,
                                            target REAL, created TEXT);
        CREATE TABLE IF NOT EXISTS prices (tracked_id INTEGER, day TEXT, lowest REAL, median REAL, sellers INTEGER,
                                           lowest_store TEXT, by_store TEXT, PRIMARY KEY (tracked_id, day));
        CREATE TABLE IF NOT EXISTS alerts (tracked_id INTEGER, day TEXT, message TEXT);""")
    return con


def today():
    return dt.date.today().isoformat()


def add(query, title, image, target, snapshot):
    """Start tracking; snapshot = today's prices from the search the user is looking at."""
    with db() as con:
        cur = con.execute("INSERT INTO tracked (query, title, image, target, created) VALUES (?, ?, ?, ?, ?)",
                          (query, title, image, target, today()))
        record(con, cur.lastrowid, snapshot)
        return cur.lastrowid


def remove(tid):
    with db() as con:
        for table, col in (("tracked", "id"), ("prices", "tracked_id"), ("alerts", "tracked_id")):
            con.execute(f"DELETE FROM {table} WHERE {col} = ?", (tid,))


def record(con, tid, snap):
    con.execute("INSERT OR REPLACE INTO prices VALUES (?, ?, ?, ?, ?, ?, ?)",
                (tid, today(), snap["lowest"], snap.get("median"), snap.get("sellers"), snap.get("lowest_store"),
                 json.dumps(snap.get("by_store", {}))))


def snapshot(res, title):
    """Today's prices of the tracked item in a search result: the best-matching offer and its 'same' group."""
    from difflib import SequenceMatcher
    O = res["offers"]
    if not O:
        return None
    best = max(range(len(O)), key=lambda j: SequenceMatcher(None, O[j]["title"].lower(), title.lower()).ratio())
    V = res.get("verdict") or [["S"]]
    members = [j for j in range(len(O)) if j == best or V[best][j] == "S"]
    priced = [(O[j]["price_value"], O[j]["store"]) for j in members if O[j]["price_value"]]
    if not priced:
        return None
    by_store = {}
    for p, s in priced:
        by_store[s] = min(by_store.get(s, 1e12), p)
    low = min(priced)
    return dict(lowest=low[0], lowest_store=low[1], median=float(np.median([p for p, _ in priced])),
                sellers=len(priced), by_store=by_store)


def advice(history):
    """history = [(day, lowest)] oldest first -> dict(text, kind, projection)."""
    n = len(history)
    if n == 0:
        return dict(kind="none", text="No prices recorded yet.")
    days = [dt.date.fromisoformat(d) for d, _ in history]
    prices = np.array([p for _, p in history], dtype=float)
    cur, low = prices[-1], prices.min()
    span = (days[-1] - days[0]).days + 1
    if n < 3:
        return dict(kind="collecting", text=f"Collecting price history ({n} day{'s' if n > 1 else ''} so far). "
                                             f"Today's best price is Rs {cur:,.0f}.")
    recent = [(d, p) for d, p in zip(days, prices) if (days[-1] - d).days < 14]
    x = np.array([(d - recent[0][0]).days for d, _ in recent], dtype=float)
    slope = float(np.polyfit(x, [p for _, p in recent], 1)[0]) if len(recent) >= 3 and np.ptp(x) > 0 else 0.0
    out = dict(slope_per_day=round(slope, 1))
    if n >= 7:
        out["projection_7d"] = round(max(0.0, cur + 7 * slope), 0)
    if cur <= low * 1.02:
        out.update(kind="buy", text=f"Buy now: Rs {cur:,.0f} is the lowest price in the {span} days we have tracked.")
    elif slope < -0.002 * cur:
        out.update(kind="wait", text=f"Wait: the price has been falling about Rs {-slope:,.0f} per day over the last "
                                     f"{len(recent)} checks. Lowest so far Rs {low:,.0f}.")
    else:
        out.update(kind="hold", text=f"Rs {cur:,.0f} is {100 * (cur / low - 1):.0f}% above the lowest price seen in "
                                     f"{span} days (Rs {low:,.0f}).")
    return out


def listing():
    with db() as con:
        items = []
        for t in con.execute("SELECT * FROM tracked ORDER BY id DESC"):
            hist = [dict(r) for r in con.execute("SELECT day, lowest, median, sellers, lowest_store FROM prices "
                                                 "WHERE tracked_id = ? ORDER BY day", (t["id"],))]
            alerts = [dict(r) for r in con.execute("SELECT day, message FROM alerts WHERE tracked_id = ? "
                                                   "ORDER BY rowid DESC LIMIT 5", (t["id"],))]
            items.append(dict(t) | dict(history=hist, alerts=alerts,
                                        advice=advice([(h["day"], h["lowest"]) for h in hist])))
        return items


def telegram(message):
    token, chat = config.env("TELEGRAM_BOT_TOKEN"), config.env("TELEGRAM_CHAT_ID")
    if token and chat:
        try:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat, "text": message}, timeout=20)
        except Exception as err:
            print("telegram failed:", err)


def update(matcher=None):
    """Check every tracked product once (one fresh Serper search per product per day: 2 credits each)."""
    from src.live import search
    if matcher is None:
        from src.service import Matcher
        matcher = Matcher()
    report = []
    with db() as con:
        for t in con.execute("SELECT * FROM tracked").fetchall():
            snap = snapshot(search(matcher, t["query"], tag=today()), t["title"])
            if snap is None:
                report.append(f"{t['title'][:50]}: no price found today")
                continue
            prev = con.execute("SELECT lowest FROM prices WHERE tracked_id = ? AND day < ? ORDER BY day DESC LIMIT 1",
                               (t["id"], today())).fetchone()
            record(con, t["id"], snap)
            msgs = []
            if t["target"] and snap["lowest"] <= t["target"]:
                msgs.append(f"Price alert: {t['title'][:60]} is Rs {snap['lowest']:,.0f} at {snap['lowest_store']} "
                            f"(your target Rs {t['target']:,.0f}).")
            if prev and snap["lowest"] <= 0.95 * prev["lowest"]:
                msgs.append(f"Price drop: {t['title'][:60]} fell from Rs {prev['lowest']:,.0f} to Rs {snap['lowest']:,.0f} "
                            f"at {snap['lowest_store']}.")
            for msg in msgs:
                con.execute("INSERT INTO alerts VALUES (?, ?, ?)", (t["id"], today(), msg))
                telegram(msg)
            report.append(f"{t['title'][:50]}: Rs {snap['lowest']:,.0f} at {snap['lowest_store']}" + (" | ALERT" if msgs else ""))
    return report


if __name__ == "__main__":
    if "--update" in sys.argv:
        print("\n".join(update()) or "nothing tracked")
    else:
        print(json.dumps(listing(), indent=1, ensure_ascii=False))
