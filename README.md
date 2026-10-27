# Mobile & Laptop Sales AI Analyst

A portfolio project that combines a Streamlit web app, sales analytics with Pandas, an OpenAI-powered question-answering interface, and a link to an existing Tableau Public dashboard.

## Included
- `app.py` — Streamlit application
- `mobile_sales_data_public.csv` — public-safe copy of the dataset, excluding customer names, customer locations, and free-text product descriptions
- `requirements.txt` — Python dependencies
- `.gitignore` — avoids committing secrets and local environment files

## Important data assumptions
- Estimated Revenue is calculated as `Price * Quantity Sold`, assuming `Price` is a unit price.
- This is not profit. The dataset has no cost or profit field.
- Row count is described as records, not orders.
- Customer names and detailed customer locations are excluded from the packaged public dataset.
- The source dataset has 50,000 rows and fields include Product, Brand, Product Code, Price, Inward Date, Dispatch Date, Quantity Sold, Region, and device specifications.

## Run locally
1. Install Python 3.10 or newer.
2. Extract this folder.
3. Open a terminal in the project folder and run:
   ```bash
   python -m venv .venv
   ```
4. Activate the environment:
   - Windows PowerShell: `.venv\\Scripts\\Activate.ps1`
   - macOS/Linux: `source .venv/bin/activate`
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
6. Set your API key as an environment variable (never paste it into `app.py`):
   - Windows PowerShell: `$env:OPENAI_API_KEY="your-api-key"`
   - macOS/Linux: `export OPENAI_API_KEY="your-api-key"`
7. Run:
   ```bash
   streamlit run app.py
   ```

## Deploy online with Streamlit Community Cloud
1. Create a GitHub repository and upload the files from this folder.
2. Open Streamlit Community Cloud and deploy `app.py` from the repository.
3. In the app's settings, add a secret:
   ```toml
   OPENAI_API_KEY = "your-api-key"
   ```
4. Redeploy and test the public app URL.

Do not commit API keys to GitHub. Public apps can incur API charges; add usage limits/monitoring to your API account before sharing widely.

## Tableau dashboard
The app links to the published dashboard:
https://public.tableau.com/app/profile/atika.faizah77/viz/MobileandLaptopSalesDashboard/DashboardSummary

The initial version links to Tableau rather than embedding it in the page, to keep deployment simple and reliable. An embedded layout can be added after the basic app is tested.
