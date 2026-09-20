# Deployment

FlyGo is a static site.
The visitor's browser loads the graph and policy-value weights, then runs one-second MCTS in a Web Worker.
Cloudflare serves files but does not execute Go search.

## Verify the model

Inspect `web/public/flygo/manifest.json` before deployment.
The 19x19 policy must have `"trained": true` for a trained release.
The top-level checkpoint metadata must identify the dataset, graph, hyperparameters, and seed.

Run all checks:

```bash
make check
```

## Build the artifact

```bash
make static
```

This writes `web/dist/`.
Content-hashed application and worker files live under `/assets/`.
The graph, manifest, and policy files live under `/flygo/`.
`web/public/_headers` sets their cache policy.

Check the artifact locally:

```bash
uv run python -m http.server 8099 --directory web/dist
```

Open <http://127.0.0.1:8099> and verify that a 19x19 reply reports search simulations after about one second.

## Publish to Cloudflare

```bash
CLOUDFLARE_API_TOKEN=... CLOUDFLARE_ACCOUNT_ID=... \
  npx wrangler@4 pages deploy web/dist --project-name flygo --branch main
```

Keep credentials outside the repository.
The command prints the production or preview URL.

For a repository-connected Pages build, use:

| Setting | Value |
| --- | --- |
| Build command | `cd web && npm ci && npm run build` |
| Output directory | `web/dist` |
| Environment variable | `FLYGO_PUBLIC_BASE` = `/` |
| Environment variable | `FLYGO_OUT_DIR` = `dist` |

`FLYGO_PUBLIC_BASE` changes asset URLs from the FastAPI `/static/` mount to the site root.
`FLYGO_OUT_DIR` keeps deployable output separate from committed FastAPI assets.

Cloudflare currently permits individual Worker static assets up to 25 MiB.
The 19x19 graph and policy total about 2.0 MB.
See the [Cloudflare Workers limits](https://developers.cloudflare.com/workers/platform/limits/) before increasing model size.

## Hostname

Use the default Pages URL.
A custom hostname is out of scope for now.
Revisit it after the trained release.
