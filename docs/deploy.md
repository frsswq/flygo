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

Publish that directory with `wrangler`:

```bash
CLOUDFLARE_API_TOKEN=... CLOUDFLARE_ACCOUNT_ID=... \
  npx wrangler@4 pages deploy web/dist --project-name flygo --branch main
```

The project `flygo` serves the production branch `main` at <https://flygo.pages.dev>.
The command prints the URL of each deployment, including the immutable preview URL for that build.
Keep the token outside the repository, for example in `~/.config/.wrangler/cf-api-token`.

## Custom domain

The project also carries the hostname `gofly.farissaifuddin.com`.
The hostname is attached, but it is not live yet.

Two conditions are missing:

1. `farissaifuddin.com` is under a client hold at its registrar, so the domain does not resolve at all.
   RDAP reports that status, and only the registrar can lift it.
2. The zone must sit in the same Cloudflare account as the Pages project, and a `gofly` record must point at `flygo.pages.dev`.
   Both Cloudflare nameservers published for the domain answer `REFUSED`, which is what Cloudflare returns for a zone it does not host.

The Pages API reports the second condition as `CNAME record not set`:

```bash
curl -sS -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
  "https://api.cloudflare.com/client/v4/accounts/$CLOUDFLARE_ACCOUNT_ID/pages/projects/flygo/domains/gofly.farissaifuddin.com"
```

Until both conditions hold, publish and share `https://flygo.pages.dev`.

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
