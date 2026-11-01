# DupeLens: rules for every assistant session

Read HANDOFF.md first (it is the source of truth: setup, results, code map, progress log in section 9).
BLUEPRINT.md has the original plan and method.

## Rules (from BLUEPRINT.md, HANDOFF.md and the team)
1. Never invent numbers. Every number in the README, dashboard or a report must come from code that was
   actually run in this repo. Measure first, then write.
2. The Shopee TEST split is used up (cache/test_done.json). Never run `python -m src.train --test`.
   Compare new models on the Shopee VALIDATION split, and evaluate them on a FRESH Indian held-out
   labelled set (pairs never labelled before). Never re-tune on a set you report.
3. Price is never a matching feature (the price comparison would be circular).
4. No hand labelling by the team. Claude labels, blind to model scores, from titles and photos. Every
   report must say: "labelled by Claude, an AI assistant, not humans".
5. Keep code simple and explain each change in plain words (the team are students; avoid jargon or define it).
6. Ask before changing scope, pushing to GitHub, or deploying (Vercel / any hosting).
7. After every milestone, add a line to the progress log in HANDOFF.md section 9 and commit locally.
   Do not push unless the user says so.
8. If a setup or run step fails, stop and show the exact error before trying any workaround.

## Running on Windows
- Python 3.11 venv: `.venv\Scripts\python.exe -m <module>` from the repo root (HANDOFF uses Mac `.venv/bin`).
- Tests: `.venv\Scripts\python.exe -m pytest tests -q` (25 should pass).
- Live app: `.venv\Scripts\python.exe -m src.app` -> http://127.0.0.1:8000 (restart after editing src/service.py).
