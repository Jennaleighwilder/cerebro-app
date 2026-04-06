# CEREBRO — Deploy to Vercel / Railway / GitHub Pages

## Quick Deploy (Vercel)

1. **Push to GitHub:**
   ```bash
   cd "/Users/jenniferwest/Downloads/files (88)"
   git init
   git add public/ cerebro_export_ui_data.py cerebro_harm_clock_data.csv vercel.json
   git commit -m "CEREBRO v1.0 — Harm Tolerance Clock with GSS Ring B"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/cerebro-app.git
   git push -u origin main
   ```

2. **Deploy on Vercel:**
   - Go to [vercel.com](https://vercel.com) → New Project
   - Import your GitHub repo
   - **Root Directory:** leave default (or set to project root)
   - **Build Command:** `python cerebro_export_ui_data.py` (or leave empty — data is pre-exported)
   - **Output Directory:** `public`
   - Deploy

3. **Before each deploy**, run locally to refresh data:
   ```bash
   ./run_all.sh
   python cerebro_export_ui_data.py
   git add public/cerebro_data.json
   git commit -m "Update clock data"
   git push
   ```

## Railway

1. Create `railway.json` or use `npx serve public -p 3000`
2. Set root to `public/` and serve static files

## GitHub Pages

1. Push to GitHub
2. Settings → Pages → Source: Deploy from branch
3. Branch: main, folder: `/public` (or `/root` if index.html at root)
4. For `/public` as root: use `public/` as the docs folder in Pages settings

## Data Refresh Pipeline

```bash
./run_all.sh                    # Gather + GSS + Phase1
python cerebro_export_ui_data.py  # Export JSON for frontend
# Then push to trigger redeploy
```

### Automated refresh (GitHub Actions)

Workflow **Cerebro data refresh** (`.github/workflows/cerebro-data-refresh.yml`) runs weekly and on demand. It runs `cerebro_export_ui_data.py` and commits `public/cerebro_data.json` when it changes so Vercel can redeploy.

- Add repository secret **`FRED_API_KEY`** (optional) so the job can run `cerebro_phase1_ingest.py` and extend series before export. Without it, the job still re-exports from the CSVs already in the repo.

### Live Oracle on Vercel

The static site calls **`POST /api/oracle`** (serverless function in `api/oracle.py`), with fallback to **`POST /oracle`** for local `python cerebro_api.py`. The router uses `cerebro_oracle_router.route_query` and reads `public/cerebro_data.json`.

Vercel must install Python deps from **`requirements.txt`** (auto-detected for `/api` routes). If the function bundle is too large, trim optional packages in a branch or raise function memory in `vercel.json`.
