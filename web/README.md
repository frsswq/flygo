# FlyGo web

This Vite React application provides the FlyGo research viewer. It uses shadcn components backed by Base UI, Tailwind CSS, and Ultracite with Oxlint and Oxfmt.

Run the FastAPI backend on port 8000 before starting the development server:

```bash
npm ci
npm run dev
```

Vite proxies `/api` to FastAPI during development. Production builds write to `../src/flygo/static/`.

Run the frontend quality gates with:

```bash
npm run check
npm run typecheck
npm test
npm run build
```

Add another Base UI component with `npx shadcn@latest add <component>`.
