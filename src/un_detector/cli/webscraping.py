#!/usr/bin/env python3
"""
Safe Adobe Stock Fetcher (preview-only) – Jupyter/Script compatible

Provides a convenience function `download_second_image(page_url, outdir, filename)`
that always selects the second candidate image ([1]) found on the page, downloads it
into `outdir` and saves it under `filename`.

⚠️ This script **only** downloads publicly exposed preview thumbnails or page images.
For licensed/full-resolution assets, license via Adobe Stock or use their official API.

Usage in Jupyter:
  from stock_preview_fetcher import download_second_image
  download_second_image("https://...", outdir="downloads", filename="myname.jpg")

Usage in terminal (unchanged): use --list / --download / --download-all

Dependencies:
  pip install requests beautifulsoup4
"""

import os
import re
import sys
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "image"


def head_content_length(url: str) -> int | None:
    try:
        r = requests.head(url, headers=DEFAULT_HEADERS, allow_redirects=True, timeout=10)
        if r.status_code == 200:
            ct = r.headers.get("Content-Length")
            if ct and ct.isdigit():
                return int(ct)
    except Exception:
        return None
    return None


def download_file(url: str, outdirs: [str]) -> None:
    for outdir in outdirs:
        ensure_outdir(os.path.dirname(outdir))

    with requests.get(url, headers=DEFAULT_HEADERS, stream=True, timeout=30) as r:
        r.raise_for_status()
        for outdir in outdirs:
            with open(outdir, "wb") as f:
                f.write(r.content)


def extract_all_image_candidates(page_url: str, base_url: str | None = None) -> list:
    """Return a list of candidate image dicts found on the page.
    Each dict contains: url (absolute), attrs (dict of HTML attrs), estimated size_bytes
    (from HEAD if available), and filename hint.
    """
    resp = requests.get(page_url, headers=DEFAULT_HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    base = base_url or page_url

    candidates = []
    seen = set()

    # 1) Meta tags (og:image, twitter:image, itemprop image)
    for prop in ("og:image", "twitter:image", "twitter:image:src"):
        # meta may be name= or property=
        meta = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
        if meta and meta.get("content"):
            url = urljoin(base, meta["content"].strip())
            if url not in seen:
                seen.add(url)
                candidates.append({"url": url, "attrs": {"source": prop}})

    # 2) <link rel="image_src">
    link = soup.find("link", rel="image_src")
    if link and link.get("href"):
        url = urljoin(base, link["href"].strip())
        if url not in seen:
            seen.add(url)
            candidates.append({"url": url, "attrs": {"source": "link:image_src"}})

    # 3) All <img> tags (src, data-src, srcset)
    imgs = soup.find_all("img")
    for img in imgs:
        attrs = dict(img.attrs)
        srcs = []
        for key in ("src", "data-src", "data-lazy", "data-srcset", "data-original"):
            if attrs.get(key):
                srcs.append(attrs[key])
        # handle srcset
        if attrs.get("srcset"):
            # pick all entries from srcset
            parts = [p.strip() for p in attrs["srcset"].split(",") if p.strip()]
            for p in parts:
                url_part = p.split()[0]
                srcs.append(url_part)

        for s in srcs:
            if not s:
                continue
            abs_url = urljoin(base, s)
            if abs_url in seen:
                continue
            seen.add(abs_url)
            candidate = {"url": abs_url, "attrs": attrs}
            candidates.append(candidate)

    # 4) Inline CSS background-image in style attributes or <style> blocks (quick regex search)
    for m in re.finditer(r"url\(([^)]+)\)", resp.text):
        raw = m.group(1).strip().strip("\"'")
        if raw.startswith("data:"):
            continue
        abs_url = urljoin(base, raw)
        if abs_url in seen:
            continue
        seen.add(abs_url)
        candidates.append({"url": abs_url, "attrs": {"source": "css_url"}})

    # Enrich with HEAD size if available and filename hint
    enriched = []
    for c in candidates:
        size = head_content_length(c["url"]) or None
        filename = sanitize_filename(os.path.basename(urlparse(c["url"]).path))
        enriched.append(
            {
                "url": c["url"],
                "attrs": c.get("attrs", {}),
                "size_bytes": size,
                "filename": filename,
            }
        )

    return enriched


# Public helpers for Jupyter
def list_images(page_url: str) -> list:
    """Return list of candidate images found on the page."""
    return extract_all_image_candidates(page_url)


def download_image(url: str, outdir: str = "downloads", suggested_name: str | None = None) -> str:
    """Download a single image URL to outdir and return the path."""
    return download_file(url, outdir, suggested_name)


def download_second_image(page_url: str, outdirs: [str]) -> str:
    """Convenience function: find candidate images on `page_url`, select the second one ([1]),
    download it into `outdir` with the provided `filename`, and return the saved path.

    Raises RuntimeError if fewer than 2 candidates are found or download fails.
    """
    imgs = extract_all_image_candidates(page_url)
    if len(imgs) < 2:
        raise RuntimeError(f"Expected at least 2 image candidates, found {len(imgs)}")
    target = imgs[1]
    saved = download_file(target["url"], outdirs)
    return saved


# ---------------------- CLI entry point ----------------------
def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="List or download image candidates from a page (preview-only)."
    )
    parser.add_argument("--url", required=True, help="Page URL to fetch images from")
    parser.add_argument("--outdir", default="downloads", help="Directory to save images")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List candidate images and exit")
    group.add_argument(
        "--download", type=int, help="Download one image by its listed index (0-based)"
    )
    group.add_argument("--download-all", action="store_true", help="Download all candidate images")

    args = parser.parse_args()

    imgs = extract_all_image_candidates(args.url)
    if args.list:
        if not imgs:
            print("No candidate images found.")
            return
        print(f"Found {len(imgs)} candidate images:\n")
        for i, img in enumerate(imgs):
            size = img.get("size_bytes")
            human = f"{size} bytes" if size is not None else "unknown size"
            print(f"[{i}] {img['url']} — {human} — attrs={img['attrs']}\n")
        return

    # downloads
    ensure_outdir(args.outdir)
    if args.download_all:
        for i, img in enumerate(imgs):
            hint = f"{i}_" + img.get("filename", "image")
            try:
                path = download_file(img["url"], [os.path.join(args.outdir, hint)])
                print(f"Saved [{i}] -> {path}")
            except Exception as e:
                print(f"Failed to download [{i}] {img['url']}: {e}")
        return

    # download single index
    idx = args.download
    if idx < 0 or idx >= len(imgs):
        print(f"Index {idx} out of range (0..{len(imgs)-1}).")
        return
    img = imgs[idx]
    hint = f"{idx}_" + img.get("filename", "image")
    try:
        path = download_file(img["url"], [os.path.join(args.outdir, hint)])
        print(f"Saved [{idx}] -> {path}")
    except Exception as e:
        print(f"Failed to download [{idx}] {img['url']}: {e}")


if __name__ == "__main__":
    if sys.argv[0].endswith(("ipykernel_launcher.py", "__main__.py")) and "--url" not in sys.argv:
        print(
            "This module is Jupyter-friendly. "
            + "Use list_images(url) or download_second_image(url, ...) instead of CLI."
        )
    else:
        _cli()
