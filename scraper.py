import requests
from bs4 import BeautifulSoup
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
from textblob import TextBlob
import torch
from datetime import datetime
import os

print('Starting daily scrape...')

# --- Scrape ---
tickers = ['AAPL', 'TSLA', 'GOOGL']
headers = {'User-Agent': 'Mozilla/5.0'}
all_headlines = []

for ticker in tickers:
    url = f'https://finviz.com/quote.ashx?t={ticker}'
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    news_table = soup.find(id='news-table')
    if news_table is None:
        print(f'Skipping {ticker} - no news table')
        continue
    last_date = None
    for row in news_table.find_all('tr'):
        timestamp_td = row.find('td', align='right')
        headline_tag = row.find('a')
        if not headline_tag:
            continue
        headline = headline_tag.text.strip()
        if timestamp_td:
            parts = timestamp_td.text.strip().split()
            if len(parts) == 2:
                last_date = parts[0]
                time = parts[1]
            elif len(parts) == 1:
                time = parts[0]
            else:
                time = None
        else:
            time = None
        all_headlines.append({'ticker': ticker, 'date': last_date, 'time': time, 'headline': headline})

df = pd.DataFrame(all_headlines)
df.dropna(subset=['date'], inplace=True)
df.reset_index(drop=True, inplace=True)

today_str = datetime.today().strftime('%b-%d-%y')
df['date'] = df['date'].replace('Today', today_str)
df['date'] = pd.to_datetime(df['date'], format='%b-%d-%y')
print(f'Scraped {len(df)} headlines')

# --- TextBlob ---
def label_sentiment(score):
    if score > 0.05: return 'positive'
    elif score < -0.05: return 'negative'
    else: return 'neutral'

df['sentiment_score'] = df['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['sentiment_label'] = df['sentiment_score'].apply(label_sentiment)
print('TextBlob done')

# --- FinBERT ---
print('Loading FinBERT...')
tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
model.eval()

def get_finbert_sentiment(headline):
    inputs = tokenizer(headline, return_tensors='pt', truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][predicted_class].item()
    label = model.config.id2label[predicted_class]
    return label, round(confidence, 4)

finbert_labels, finbert_scores = [], []
for i, headline in enumerate(df['headline']):
    label, confidence = get_finbert_sentiment(headline)
    finbert_labels.append(label)
    finbert_scores.append(confidence)
    if (i + 1) % 50 == 0:
        print(f'FinBERT: {i+1}/{len(df)} done')

df['finbert_label'] = finbert_labels
df['finbert_confidence'] = finbert_scores

# --- Save ---
df.to_csv('financial_sentiment_data_v2.csv', index=False)
print(f'Saved {len(df)} rows to financial_sentiment_data_v2.csv')
print('Daily scrape complete!')
