"""Bundle the offline page from separate calculation/UI/CSS sources. Stdlib only."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
page = (ROOT / 'page.html').read_text(encoding='utf-8')
for marker, filename in [('STYLE', 'style.css'), ('MODEL', 'model.js'), ('UI', 'ui.js'), ('EXTENSION', 'extension.js'), ('LEARNING', 'learning.js'), ('PRESENTATION', 'presentation.js')]:
    page = page.replace(f'/*__{marker}__*/', (ROOT / filename).read_text(encoding='utf-8'))
(ROOT / 'index.html').write_text(page, encoding='utf-8')
print(f'Built {ROOT / "index.html"} ({len(page.encode("utf-8")):,} bytes)')
