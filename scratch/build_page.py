import base64
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
template = ROOT / "scratch" / "page_template.html"
out = ROOT / "scratch" / "figures_page.html"

html = template.read_text(encoding="utf-8")
missing = []


def inline(match):
    name = match.group(1)
    path = ROOT / "figures" / f"{name}.png"
    if not path.exists():
        missing.append(name)
        return ""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


html = re.sub(r"\{\{FIG:([a-z0-9_]+)\}\}", inline, html)
if missing:
    sys.exit(f"missing figures: {missing}")

out.write_text(html, encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
