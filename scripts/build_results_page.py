"""Wrap the results page body into a deployable static document for GitHub Pages.

`docs/_page.html` holds the title, styles and content, and is also what gets published
as a hosted artifact. This adds the document shell, SEO and Open Graph metadata so the
same source produces both without the two drifting apart.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

TITLE = "Cricket AI — Probabilistic Next-Over Forecasting"
DESCRIPTION = ("Machine-learning system for calibrated next-over run forecasting and event "
               "probabilities in T20 cricket.")
URL = "https://yogeshwardev.github.io/CRICKET-SCORE-PREDICTION-/"

SHELL = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{description}">
<meta name="color-scheme" content="dark">
<meta name="author" content="yogeshwardev">
<link rel="canonical" href="{url}">
<meta property="og:type" content="website">
<meta property="og:site_name" content="Cricket AI">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{description}">
<meta property="og:url" content="{url}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{title}">
<meta name="twitter:description" content="{description}">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' fill='%230C1116'/><text x='16' y='23' font-size='19' text-anchor='middle' fill='%23F0B429' font-family='Georgia,serif'>c</text></svg>">
<style>html{{background:#0C1116}}body{{margin:0}}img{{max-width:100%}}[hidden]{{display:none!important}}</style>
{body}
</head>
<body>
{content}
</body>
</html>
"""


def build(source: Path, destination: Path) -> Path:
    raw = source.read_text(encoding="utf-8")
    # Everything up to the end of the <style> block belongs in <head>; the rest is body.
    split = raw.index("</style>") + len("</style>")
    head, content = raw[:split], raw[split:]
    head = re.sub(r"<title>.*?</title>\s*", "", head, count=1, flags=re.S)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        SHELL.format(title=TITLE, description=DESCRIPTION, url=URL,
                     body=head.strip(), content=content.strip()),
        encoding="utf-8")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    out = build(root / "docs/_page.html", root / "docs/index.html")
    print(f"built {out} ({out.stat().st_size:,} bytes)")
