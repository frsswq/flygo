#!/bin/sh
set -eu

if [ -n "${BROWSER_BIN:-}" ] \
    || find "$HOME/.cache/ms-playwright" -path '*/chrome-linux64/chrome' -type f -print -quit 2>/dev/null | grep -q . \
    || command -v chromium >/dev/null 2>&1 \
    || command -v google-chrome >/dev/null 2>&1; then
    npm --prefix web run test:browser
else
    echo "browser regression skipped: no Chrome or Chromium found (set BROWSER_BIN)"
fi
