import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="US Airlines Tweet Sentiment", layout="wide", page_icon="✈️")

# --- Sidebar Header ---
st.sidebar.image("https://media.istockphoto.com/id/537126290/photo/american-airlines-a319-taking-off-at-charlotte-douglas-international-airport.jpg?s=612x612&w=0&k=20&c=FXAeZpQEtU9f3xwL45YBZDtnsL5k3Jx9ZfZ0oJUGgUc=", width=80)
st.sidebar.title("US Airlines Sentiment Dashboard")
st.sidebar.markdown("Analyze public sentiment about US airlines using real tweets.")

DATA_URL = "Tweets.csv"

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL)
    data['tweet_created'] = pd.to_datetime(data['tweet_created'])
    # Parse tweet_coord into latitude and longitude
    data['tweet_coord'] = data['tweet_coord'].apply(lambda x: eval(x) if pd.notnull(x) and x != '' else None)
    data['lat'] = data['tweet_coord'].apply(lambda x: x[0] if isinstance(x, list) else None)  # latitude
    data['lon'] = data['tweet_coord'].apply(lambda x: x[1] if isinstance(x, list) else None)  # longitude
    return data

data = load_data()

# --- Main Title ---
st.title("✈️ US Airlines Tweet Sentiment Analysis")
st.markdown("""
This interactive dashboard provides insights into public sentiment about US domestic airlines based on Twitter data.
Explore sentiment trends, tweet locations, airline comparisons, and more!
""")

# --- 1. Show Random Tweet by Sentiment ---
st.sidebar.subheader("Show Random Tweet")
random_tweet = st.sidebar.radio('Sentiment', ['positive', 'neutral', 'negative'])
tweet = data.query('airline_sentiment == @random_tweet')['text'].sample(n=1).iat[0]
st.sidebar.markdown(f"**Random {random_tweet} tweet:**<br><i>{tweet}</i>", unsafe_allow_html=True)

# --- 2. Sentiment Distribution ---
st.sidebar.markdown('### Number of Tweets by Sentiment')
select = st.sidebar.selectbox('Visualization Type', ['Histogram', 'Pie Chart'], key='1')
sentiment_count = data['airline_sentiment'].value_counts()
sentiment_count_df = pd.DataFrame({'Sentiment': sentiment_count.index, 'Tweets': sentiment_count.values})

if not st.sidebar.checkbox('Hide', True, key='hide_sentiment'):
    st.subheader("Number of Tweets by Sentiment")
    col1, col2 = st.columns(2)
    with col1:
        if select == 'Histogram':
            fig = px.bar(sentiment_count_df, x='Sentiment', y='Tweets', color='Sentiment', height=400, text='Tweets')
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.pie(sentiment_count_df, values='Tweets', names='Sentiment', title='Tweets by Sentiment')
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.dataframe(sentiment_count_df, use_container_width=True)

# --- 3. Tweets by Hour and Map ---
st.sidebar.subheader("When and Where Are Users Tweeting?")
hour = st.sidebar.slider('Hour of the day', 0, 23)
modified_data = data[data['tweet_created'].dt.hour == hour]

if not st.sidebar.checkbox('Close', True, key='2'):
    st.subheader(f"Tweet Locations Between {hour}:00 and {(hour+1)%24}:00")
    st.markdown(f"**{len(modified_data)} tweets** between {hour}:00 and {(hour+1)%24}:00")
    map_data = modified_data.dropna(subset=['lat', 'lon'])
    if not map_data.empty:
        st.map(map_data[['lat', 'lon']])
    else:
        st.info("No location data available for this hour.")
    if st.checkbox('Show raw data', False, key='show_raw_data'):
        st.write(modified_data)

# --- 4. Most Tweeted Airlines ---
st.sidebar.subheader("Most Tweeted Airlines")
airline_sentiment = data['airline'].value_counts()
airline_sentiment_df = pd.DataFrame({'Airline': airline_sentiment.index, 'Tweets': airline_sentiment.values})

if not st.sidebar.checkbox('Close', True, key='3'):
    st.subheader("Number of Tweets by Airline")
    fig = px.bar(airline_sentiment_df, x='Airline', y='Tweets', color='Tweets', height=400, text='Tweets')
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    if st.checkbox('Show raw data', False, key='show_raw_data_airline'):
        st.write(airline_sentiment_df)

# --- 5. Airline Sentiment Breakdown ---
st.sidebar.subheader("Breakdown Airline Tweets by Sentiment")
airlines = sorted(data['airline'].unique())
choice = st.sidebar.multiselect('Pick airlines', airlines, key='0')

if choice:
    st.subheader("Sentiment Breakdown by Airline")
    choice_data = data[data.airline.isin(choice)]
    fig_choice = px.histogram(
        choice_data, x='airline', color='airline_sentiment', barmode='group',
        labels={'airline_sentiment': 'Sentiment', 'airline': 'Airline'},
        height=500, width=900, text_auto=True
    )
    st.plotly_chart(fig_choice, use_container_width=True)

# --- 6. Sentiment Over Time ---
if st.checkbox("Show Sentiment Trend Over Time"):
    st.subheader("Sentiment Trend Over Time")
    time_data = data.copy()
    time_data['date'] = time_data['tweet_created'].dt.date
    trend = time_data.groupby(['date', 'airline_sentiment']).size().reset_index(name='count')
    fig_trend = px.line(
        trend, x='date', y='count', color='airline_sentiment',
        labels={'count': 'Number of Tweets', 'date': 'Date', 'airline_sentiment': 'Sentiment'},
        markers=True
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# --- 7. Word Cloud Section ---
st.sidebar.header("Word Cloud")
word_sentiment = st.sidebar.radio('Display word cloud for what sentiment?', ('positive', 'neutral', 'negative'))

if not st.sidebar.checkbox('Close', True, key='5'):
    st.subheader(f"Word Cloud for {word_sentiment.capitalize()} Sentiment")
    df = data[data['airline_sentiment'] == word_sentiment]
    words = ' '.join(df['text'].astype(str))
    processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@')])
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', height=400, width=800).generate(processed_words)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# --- 8. Top Words by Sentiment ---
if st.checkbox("Show Top Words by Sentiment"):
    st.subheader("Top 10 Most Frequent Words by Sentiment")
    from collections import Counter
    cols = st.columns(3)
    for idx, sentiment in enumerate(['positive', 'neutral', 'negative']):
        df = data[data['airline_sentiment'] == sentiment]
        words = ' '.join(df['text'].astype(str)).lower().split()
        words = [w for w in words if w.isalpha() and w not in STOPWORDS and not w.startswith('@')]
        common_words = Counter(words).most_common(10)
        with cols[idx]:
            st.markdown(f"**{sentiment.capitalize()}**")
            st.table(pd.DataFrame(common_words, columns=['Word', 'Count']))

# --- 9. Sentiment by Reason ---
if st.checkbox("Show Sentiment by Reason for Negative Tweets"):
    st.subheader("Negative Tweet Reasons by Airline")
    if 'negativereason' in data.columns:
        neg_data = data[data['airline_sentiment'] == 'negative']
        reason_count = neg_data.groupby(['airline', 'negativereason']).size().reset_index(name='count')
        fig_reason = px.bar(
            reason_count, x='airline', y='count', color='negativereason',
            labels={'count': 'Number of Tweets', 'airline': 'Airline', 'negativereason': 'Reason'},
            height=500
        )
        st.plotly_chart(fig_reason, use_container_width=True)
    else:
        st.info("No 'negativereason' column found in data.")

# --- 10. Download Data ---
st.sidebar.markdown("---")
st.sidebar.download_button(
    label="Download Cleaned Data as CSV",
    data=data.to_csv(index=False).encode('utf-8'),
    file_name='cleaned_tweets.csv',
    mime='text/csv'
)
#finalize the app