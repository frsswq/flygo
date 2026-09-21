# FlyGo web

This Vite React application runs the 5x5 and 19x19 FlyGo viewer entirely in the browser. A Web Worker loads the frozen graph and policy-value weights, then runs one-second PUCT search without blocking React.

## Develop

Install dependencies from `web/`:

```bash
npm ci
npm run dev
```

Open <http://127.0.0.1:5173>. No backend is required. Vite loads committed bundles from `public/flygo/`.

## Check and build

```bash
npm run check
npm run typecheck
npm run knip
npm test
npm run build
```

The default build writes the FastAPI-mounted artifact to `../src/flygo/static/`. Run `make static` from the repository root to write a root-mounted Cloudflare artifact to `web/dist/`.

`npm run test:browser` runs the browser regression instead of the unit suite. It builds a root-mounted app, serves it with a deliberately slow bundle, and drives a real Chromium through `playwright-core`. It covers self-play and reset during bundle loading, rapid self-play toggles, and a board-size change while loading. Set `BROWSER_BIN` when Chrome or Chromium is not in a standard location; without a browser binary the run fails with that instruction. `make browser` runs the same suite, and `make check` includes it whenever a browser binary is present.

Generate browser weights from Python before a trained release:

```bash
uv run flygo export-web --policy data/models/flygo-19.npz
uv run flygo policy-conformance
```

See the [training pipeline](../docs/pipeline.md) and [deployment guide](../docs/deploy.md) for the full workflow.
