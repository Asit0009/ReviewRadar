# ReviewRadar Frontend

React + TypeScript dashboard for the ReviewRadar API. Every number on screen comes from a backend response. Anything the backend cannot provide is shown as "Not available yet".

Stack: Vite, React 19, TypeScript, Tailwind CSS v4, Recharts, Lucide icons, TanStack Query, React Router. Tests use Vitest and Testing Library.

## Two ways to run it

**Production (one process, one port).** The backend serves the built dashboard:

```bash
cd frontend && npm install && npm run build      # writes frontend/dist
cd .. && uvicorn reviewpulse.api.app:app --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000/
```

**Development (hot reload).** Start the backend as above, then:

```bash
cd frontend && npm run dev                       # http://localhost:5173
```

The dev server proxies `/api` to `http://127.0.0.1:8000` (see `vite.config.ts`), so the page and the API share one origin, exactly like production. No `.env` file is needed. To use an API on another host, set `VITE_API_BASE_URL` (and add the page's origin to `CORS_ORIGINS` on the backend).

## Scripts

| Command | What it does |
| :--- | :--- |
| `npm run dev` | Dev server with hot reload and the `/api` proxy |
| `npm run build` | Type-check, then production build into `dist/` |
| `npm run preview` | Serve `dist/` on :4173 with the same proxy (a quick local check, not the production entry point) |
| `npm run lint` | ESLint |
| `npm run typecheck` | TypeScript only |
| `npm test` | Unit and component tests (fetch is mocked; no backend needed) |
| `npm run check:contract` | Compares `src/api/types.ts` with the running backend's `/openapi.json` |
| `npm run e2e` | Real-browser test of the production app through `http://127.0.0.1:8000` (needs `npm run build`, the backend running, and Chrome). Checks four screen sizes, console/network errors, every route, filters, search and pagination, prediction, the assistant and API-down behaviour. Set `E2E_SCREENSHOT_DIR` to save pictures |

## Pages

| Route | Page | Main endpoints |
| :--- | :--- | :--- |
| `/` | Overview | `/overview`, `/sentiment/trend`, `/ratings/distribution`, `/topics`, `/models/metrics`, `/reviews`, `/monitoring/status`, `/health`, `/assistant/chat` |
| `/reviews` | Reviews Analysis (filters live in the URL) | `/reviews`, `/topics` |
| `/sentiment` | Sentiment Analysis | `/overview`, `/sentiment/trend`, `/ratings/distribution`, `/categories`, `/reviews` |
| `/topics` | Topic Modeling | `/topics` |
| `/predictions` | ML Predictions | `/predict`, `/models/metrics`, `/models/metrics/by-category` |
| `/deep-learning` | Deep Learning (NLP) | `/models/metrics`, `/assistant/chat`, `/health` |
| `/assistant` | LLM Assistant | `/assistant/chat`, `/health` |
| `/monitoring` | Deploy & Monitor | `/health`, `/monitoring/status` |

## Filters

The category selector and the From / To dates are shared by every page (`src/context/FilterContext.tsx`). They are sent to the API as `category`, `date_from` and `date_to` with every request that supports them, so the backend recalculates the numbers in SQL. If the start date is after the end date the dates are not sent (the API would answer 422) and the form says why.

## How it is organised

```
src/
  api/          client.ts (fetch + errors), types.ts, endpoints.ts, queries.ts (hooks)
  context/      filters (category + dates), assistant chat history, layout state
  components/   layout/, ui/, charts/, overview/, reviews/, assistant/
  pages/        one file per route
  lib/          formatting, text cleaning, model-table helpers, theme colours
scripts/        check-contract.mjs
```

- `api/types.ts` mirrors `reviewpulse/api/schemas.py`. After changing a schema in the backend, run `npm run check:contract`.
- There is no mock mode. If the API is down, panels show an error with a retry button.
- The category list comes only from `/categories`; nothing is hard-coded.

## Layout

The design targets about 1536 x 1024 (left navigation, main column, tall assistant column). The assistant becomes a slide-over panel below 1280px, and the navigation becomes a drawer below 1024px. Page grids use container queries, so they react to the width of the main column rather than the screen.
