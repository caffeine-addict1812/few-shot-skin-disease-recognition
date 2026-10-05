#!/usr/bin/env bash
# report.md -> report.pdf (pandoc to self-contained HTML, then headless Chromium prints it; no LaTeX needed).
#   bash scripts/build_pdf.sh            # CHROME=/path/to/chrome to override the browser
set -euo pipefail
cd "$(dirname "$0")/.."
CHROME="${CHROME:-$(command -v chromium || command -v google-chrome || ls ~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome 2>/dev/null | head -1)}"
[ -x "$CHROME" ] || { echo "no Chromium/Chrome found; set CHROME=" >&2; exit 1; }
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
cat > "$tmp/style.css" <<'CSS'
@page { size: A4; margin: 16mm 15mm; }
body { font: 10.5pt/1.45 "Helvetica Neue", Arial, sans-serif; color: #1d1d1b; max-width: none; margin: 0; }
h1 { font-size: 18pt; margin: 0 0 6pt; } h2 { font-size: 13pt; margin: 14pt 0 4pt; }
table { border-collapse: collapse; margin: 6pt 0; font-size: 9pt; } th, td { border-bottom: 1px solid #ddd; padding: 2pt 6pt; text-align: left; }
th { border-bottom: 1.5px solid #888; } code { font-size: 8.5pt; background: #f3f3f1; padding: 0 2pt; word-break: break-all; }
img { max-width: 100%; max-height: 95mm; display: block; margin: 6pt auto; page-break-inside: avoid; }
li { margin: 2pt 0; } tr, img { page-break-inside: avoid; }
CSS
pandoc report.md -s --embed-resources --resource-path=. --css "$tmp/style.css" \
  --metadata pagetitle="SD-198 few-shot report" -o "$tmp/report.html"
"$CHROME" --headless --no-sandbox --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$PWD/report.pdf" "file://$tmp/report.html" 2>/dev/null
echo "wrote report.pdf ($(du -h report.pdf | cut -f1))"
