# Hero's Journey — benchmark site

A single-page, dependency-free site that (1) lets readers explore how each task
works interactively and (2) shows the model leaderboard (Table 6 of the paper).

## Files

- `index.html` — the page: Overview (landing) + Task Explorer + Leaderboard. No build step.
- `data.js` — generated data: `window.HJ_TASKS` (example episodes for all 8
  tasks × {semantic, nonce}) and `window.HJ_LEADERBOARD` (Table 6 numbers).
- `hj_icon.png`, `hj_fig.png` — anonymized static images used on the Overview page.

## Viewing it

Just open `index.html` in a browser — it loads `data.js` via a `<script>` tag,
so it works from `file://` (double-click) with no server, and unchanged on
GitHub Pages.

To serve locally:

```bash
python -m http.server -d site 8000   # then open http://localhost:8000
```

## Regenerating the data

`data.js` is produced from the live task pipeline:

```bash
python experiments/build_site_data.py
```

Re-run this whenever task definitions, rules, or lexicons change. The leaderboard
numbers are transcribed in that script (`LEADERBOARD_ROWS`) from the anonymous
paper appendix table; edit there if the numbers change.
