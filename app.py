import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

st.set_page_config(page_title='Financial Sentiment Analyzer', page_icon='📈', layout='wide')

@st.cache_data(ttl=3600)  # Cache expires every 1 hour
def load_data():
    url = 'https://raw.githubusercontent.com/Onunga123/financial-sentiment-analyzer/main/financial_sentiment_data_v2.csv'
    df = pd.read_csv(url)
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

st.title('Financial News Sentiment Analyzer')
st.markdown('Analyzing market sentiment for **AAPL**, **TSLA**, and **GOOGL** using FinBERT AI')
st.markdown('---')

st.sidebar.title('Controls')
selected_ticker = st.sidebar.selectbox('Select Stock Ticker', options=['ALL', 'AAPL', 'TSLA', 'GOOGL'])
selected_model  = st.sidebar.radio('Sentiment Model', options=['FinBERT', 'TextBlob'])
st.sidebar.markdown('---')
st.sidebar.markdown('Last data update: ' + str(df['date'].max().strftime('%b %d, %Y')))
st.sidebar.markdown('Built with Python, FinBERT, Streamlit')

filtered_df = df.copy() if selected_ticker == 'ALL' else df[df['ticker'] == selected_ticker]
label_col = 'finbert_label' if selected_model == 'FinBERT' else 'sentiment_label'

st.subheader(f'Overview — {selected_ticker} ({selected_model})')
col1, col2, col3, col4 = st.columns(4)
total    = len(filtered_df)
positive = len(filtered_df[filtered_df[label_col] == 'positive'])
negative = len(filtered_df[filtered_df[label_col] == 'negative'])
neutral  = len(filtered_df[filtered_df[label_col] == 'neutral'])
col1.metric('Total Headlines', total)
col2.metric('Positive', positive, f'{round(positive/total*100, 1)}%')
col3.metric('Negative', negative, f'{round(negative/total*100, 1)}%')
col4.metric('Neutral',  neutral,  f'{round(neutral/total*100, 1)}%')
st.markdown('---')

left, right = st.columns(2)
colors = {'positive': '#2ecc71', 'neutral': '#95a5a6', 'negative': '#e74c3c'}

with left:
    st.subheader('Sentiment Distribution')
    counts = filtered_df[label_col].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values, color=[colors.get(l, '#bdc3c7') for l in counts.index], edgecolor='white')
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(val), ha='center', fontweight='bold')
    ax.set_ylabel('Number of Headlines')
    ax.set_title(f'{selected_ticker} — {selected_model}')
    st.pyplot(fig)
    plt.close()

with right:
    st.subheader('Daily Sentiment Trend')
    daily = filtered_df.groupby(['date', label_col]).size().unstack(fill_value=0)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    for label, color in colors.items():
        if label in daily.columns:
            ax2.plot(daily.index, daily[label], marker='o', label=label, color=color, linewidth=2)
    ax2.set_ylabel('Headlines per Day')
    ax2.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

st.markdown('---')
st.subheader('Latest Headlines')
display_cols = ['ticker', 'date', 'headline', label_col]
if 'finbert_confidence' in filtered_df.columns and selected_model == 'FinBERT':
    display_cols.append('finbert_confidence')
st.dataframe(filtered_df[display_cols].sort_values('date', ascending=False).head(50), use_container_width=True)

st.markdown('---')
st.caption('Data scraped from Finviz | NLP: FinBERT (ProsusAI) & TextBlob | Built on Google Colab')
