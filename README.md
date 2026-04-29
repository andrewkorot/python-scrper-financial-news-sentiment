# Financial News Sentiment Analyzer

A Python project that scrapes real-time financial news and analyzes market sentiment using FinBERT AI.

## Live Demo
https://financial-sentiment-analyzer-tr85rnwvjo4kzra6fhrnuz.streamlit.app/

## What it does
- Scrapes 300+ live headlines for AAPL, TSLA, and GOOGL from Finviz
- Cleans and structures data using Pandas
- Runs sentiment analysis using FinBERT (transformer AI model) and TextBlob
- Compares both models — FinBERT found 25 more negative headlines than TextBlob
- Interactive Streamlit dashboard with live filtering by ticker and model
- Exports CSV dataset and automated sentiment report

## Tools & Technologies
- Python, BeautifulSoup, Pandas, Matplotlib
- FinBERT (ProsusAI) via Hugging Face Transformers
- Streamlit (deployed on Streamlit Cloud)
- Google Colab, GitHub

## Key Findings (Apr 21-28, 2026)
- 300 headlines analyzed across AAPL, TSLA, GOOGL
- FinBERT detected 57 negative headlines vs TextBlob's 32 — 78% more
- TSLA had the most negative sentiment coverage
- AAPL had the most positive sentiment coverage
- 161/300 headlines labeled differently by the two models

## How to Run Locally
1. Clone this repo
2. pip install -r requirements.txt
3. streamlit run app.py

## Project Structure
- app.py — Streamlit dashboard
- financial_sentiment_data_v2.csv — full labeled dataset
- sentiment_report.txt — automated summary report
- chart1 to chart5 — visualization outputs
- requirements.txt — dependencies
