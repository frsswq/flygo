# Deployment

The site is static and all inference runs in the visitor's browser, so no server is required.
Cloudflare serves files only, which is free and unmetered on the free plan.

## Build the artifact

```bash
make static
```

That writes `web/dist/`, the directory to publish.
It serves `/assets/*` (content hashed) and `/flygo/*` (the graph and one policy per board size).
`web/public/_headers` sets the cache policy for both.

## Cloudflare settings

Connect the repository once in the Cloudflare dashboard, then use these settings:

| Setting | Value |
| --- | --- |
| Build command | `cd web && npm ci && npm run build` |
| Output directory | `web/dist` |
| Environment variable | `FLYGO_PUBLIC_BASE` = `/` |
| Environment variable | `FLYGO_OUT_DIR` = `dist` |

`FLYGO_PUBLIC_BASE` switches the built asset URLs from the FastAPI mount at `/static/` to the root.
`FLYGO_OUT_DIR` keeps the deployable build out of the committed FastAPI build.
Cloudflare Pages also accepts the same repository as a Workers project with `npx wrangler deploy`; the artifact is identical.

## Check the artifact locally

```bash
python3 -m http.server 8099 --directory web/dist
```
