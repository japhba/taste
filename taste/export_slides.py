"""Screenshot every example panel (.slide) and figure of the built report into PNGs for slides.

  PLAYWRIGHT_BROWSERS_PATH=/var/tmp/jbauer/ms-playwright uv run --no-project --with playwright python -m taste.export_slides
-> <reports>/taste-files/slides/NN_<title>.png (light theme, 2x pixel density)
"""
import re, sys
from pathlib import Path
from playwright.sync_api import sync_playwright

REPORT = Path("/ceph/scratch/jbauer/reports/taste/taste.html")
OUT = REPORT.parent / "taste-files" / "slides"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for p in OUT.glob("*.png"): p.unlink()
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        page = b.new_page(viewport={"width": 1100, "height": 900}, device_scale_factor=2, color_scheme="light")
        page.goto(REPORT.as_uri()); page.wait_for_load_state("networkidle")
        assert page.locator(".katex-error").count() == 0
        for i, el in enumerate(page.locator(".slide").all()):
            title = el.locator("h3").inner_text()
            name = f"{i + 1:02d}_" + re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:60] + ".png"
            el.screenshot(path=str(OUT / name)); print(OUT / name)
        b.close()
    for p in [OUT, *OUT.iterdir()]: p.chmod(0o755 if p.is_dir() else 0o644)


if __name__ == "__main__":
    main()
