# AlonStocks

AlonStocks is a Hebrew, RTL Streamlit portfolio dashboard backed by Google Sheets, Yahoo Finance, and Plotly.

## Run locally

1. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Install the Tesseract executable if screenshot import is needed:

   ```bash
   sudo apt-get update && sudo apt-get install -y tesseract-ocr
   ```

3. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in a Google service account that has access to the spreadsheet.

4. Start the dashboard:

   ```bash
   streamlit run app.py
   ```

If Google Sheets is unavailable, the app loads initial holdings from `portfolio_data.csv`.

## Test

```bash
python -m unittest discover -s tests -v
```

Never commit `.streamlit/secrets.toml`. Any credentials previously committed must be revoked and replaced.