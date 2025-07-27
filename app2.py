import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import Counter

# --- Airline accent palettes and SVG icons ---
AIRLINE_THEMES = {
    "Delta": {"primary": "#6384ff", "secondary": "#FF676B", "bg": "#191C28", "logo": "https://1000logos.net/wp-content/uploads/2017/09/Delta-Air-Lines-Logo-640x400.png", "svg": """<svg width="24" height="24"><circle cx="12" cy="12" r="11" stroke="#6384ff" stroke-width="2.5" fill="none" /><polyline points="5,13 12,6 19,13" stroke="#6384ff" stroke-width="2.5" fill="none" /></svg>""" },
    "United": {"primary": "#68a8ff", "secondary": "#FFD944", "bg": "#191C28", "logo": "https://1000logos.net/wp-content/uploads/2017/06/United-Airlines-Logo.png", "svg": """<svg width="24" height="24"><circle cx="12" cy="12" r="11" stroke="#68a8ff" stroke-width="2.5" fill="none"/><rect x="8" y="5" width="8" height="14" rx="4" fill="#68a8ff" opacity="0.26"/></svg>""" },
    "Southwest": {"primary": "#67BCFB", "secondary": "#F95D63", "bg": "#191C28", "logo": "https://1000logos.net/wp-content/uploads/2019/08/southwest-airlines-logo-640x289.png", "svg": """<svg width="24" height="24"><rect x="2" y="9" width="20" height="6" rx="3" fill="#67BCFB"/><circle cx="12" cy="12" r="11" stroke="#67BCFB" stroke-width="2.5" fill="none"/></svg>""" },
    "American": {"primary": "#70CBF4", "secondary": "#B6ECF8", "bg": "#191C28", "logo": "https://1000logos.net/wp-content/uploads/2016/10/American-Airlines-Logo-640x400.png", "svg": """<svg width="24" height="24"><circle cx="12" cy="12" r="11" stroke="#70CBF4" stroke-width="2.5" fill="none"/><polygon points="12,6 18,18 6,18" fill="#70CBF4" opacity="0.36"/></svg>""" },
    "Virgin America": {"primary": "#FA6CA3", "secondary": "#EDE4F0", "bg": "#191C28", "logo": "https://1000logos.net/wp-content/uploads/2023/05/Virgin-America-Logo-768x432.png", "svg": """<svg width="24" height="24"><circle cx="12" cy="12" r="11" stroke="#FA6CA3" stroke-width="2.5" fill="none"/><ellipse cx="12" cy="17" rx="6" ry="3" fill="#FA6CA3" opacity="0.19"/></svg>""" },
    "US Airways": {"primary": "#BCCBD9", "secondary": "#6272A4", "bg": "#191C28", "logo": "https://1000logos.net/wp-content/uploads/2020/06/US-Airways-Logo-640x360.png", "svg": """<svg width="24" height="24"><rect x="4" y="10" width="16" height="4" fill="#BCCBD9"/><circle cx="12" cy="12" r="11" stroke="#BCCBD9" stroke-width="2.5" fill="none"/></svg>""" },
}

st.set_page_config(
    page_title="US Airlines Tweet Sentiment",
    layout="wide",
    page_icon="✈️"
)

# Modern dark-mode CSS plus THIN tab underline
st.markdown("""
<style>
body, .block-container, .stApp { background-color: #191C28 !important; color: #e3e7ef !important; }
.theme-header { color: #87beff !important; font-size:2.0rem; font-weight:900; letter-spacing:1px; margin-bottom:0.09em; display:flex; align-items:center;}
.theme-note { font-size: 1.07rem; color: #c0caf5 !important; font-style: italic;}
.card { background: #23263a !important; color: #e3e7ef !important; border-radius:14px; box-shadow: 0 3px 24px #131522a0; margin-bottom:1.35rem; padding:1.45em 1.1em;}
.st-emotion-cache-1v0mbdj h1, .st-emotion-cache-1v0mbdj h2, .st-emotion-cache-1v0mbdj h3 { color: #8ec7ff !important; font-family: 'Montserrat', 'Roboto', Arial, sans-serif; font-weight: 900;}
a { color: #68a8ff;}
.stSidebar { background-color: #181a20 !important;}
[data-testid="stSidebar"] { background-color: #222436 !important; }
.css-1d391kg { color: #e3e7ef !important; }
.st-bq, .st-cz, .st-eg, .st-et { background: #23263a !important; color: #e3e7ef !important; border-radius:13px;}
.stRadio > div, .stSelectbox > div, .stButton > button, .stTabs, .stSlider > div { padding: 0.49em 1.1em .49em 1.1em !important; margin-bottom:0.19em !important;}
.stRadio label, .stSelectbox label { font-size:1.07em; font-weight: 700; margin-right:1.3em!important;}
.stRadio > div > div { margin-right:1.0em !important; }
.stTabs [data-baseweb="tab-list"] > div[aria-selected="true"] {
    border-bottom: 1px solid #87beff !important;  
    margin-bottom: 0 !important;
    border-radius: 2px 2px 0 0 !important;        
    box-shadow: 0 2px 8px #87beff44;
    padding: 2px;
}
.stTabs [data-baseweb="tab-list"] > div {
    transition: border-bottom 0.3s;
}
.stTabs [data-baseweb="tab-list"] {
    border-bottom: none !important;
}
</style>
""", unsafe_allow_html=True)

airline_options = list(AIRLINE_THEMES.keys())
theme_choice = st.sidebar.selectbox("Select Airline Theme", airline_options)
theme = AIRLINE_THEMES[theme_choice]
st.sidebar.markdown(
    f"""<div style="display:flex;align-items:center;gap:7px;margin-bottom:10px">
    {theme['svg']}
    <img src="{theme['logo']}" width="50" style="border-radius:7px;">
    </div>""", unsafe_allow_html=True)
st.sidebar.title(f"{theme_choice} Dashboard")
st.sidebar.markdown(
    "<span class='theme-note'>Analyze US airline sentiment from millions of real tweets.</span>",
    unsafe_allow_html=True
)
st.sidebar.markdown("---")
st.sidebar.caption("Dashboard built by Roshan.")

DATA_URL = "Tweets.csv"

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL)
    data['tweet_created'] = pd.to_datetime(data['tweet_created'])
    data['tweet_coord'] = data['tweet_coord'].apply(lambda x: eval(x) if pd.notnull(x) and x != '' else None)
    data['lat'] = data['tweet_coord'].apply(lambda x: x[0] if isinstance(x, list) else None)
    data['lon'] = data['tweet_coord'].apply(lambda x: x[1] if isinstance(x, list) else None)
    return data

data = load_data()

st.markdown(
    f"""<div class='theme-header'>{theme['svg']}<span style="margin-left:0.75em">US Airlines Tweet Sentiment Analysis</span></div>
    <div class='theme-note'>Modern, actionable, easy on your eyes. Insights live from real Twitter data.</div>""",
    unsafe_allow_html=True,
)
with st.expander("ℹ️ How to use this dashboard (expand)", expanded=False):
    st.write("""
    - Choose an airline theme (logo/color/icon) on the left.
    - Sentiment page supports Histogram, Pie, Vertical Bar, Donut charts.
    - Tabs include maps, time analysis, text clouds, complaint breakdowns, and more.
    - Download data or focus on segments for in-depth analysis.
    """)

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Sentiment Explorer",
    "Time & Location",
    "Trend & Comparison",
    "Text Insights",
    "Alerts & Drilldown"
])

with tab1:
    st.markdown(f"""<div style='display:flex;align-items:center;gap:10px;font-size:1.2em;margin-bottom:.35em;'>{theme['svg']}<span>Sentiment Explorer</span></div>""", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Sentiment Overview & Distribution")
    chart_types = ["Histogram", "Pie Chart", "Vertical Bar", "Donut"]
    chosen_chart = st.radio(
        "Choose Chart Type",
        chart_types,
        horizontal=True,
        index=0,
        key="new_sent_chart"
    )
    sentiment_count = data['airline_sentiment'].value_counts()
    sentiment_count_df = pd.DataFrame({'Sentiment': sentiment_count.index, 'Tweets': sentiment_count.values})
    accent = [theme['primary'], theme['secondary'], "#848ba3"]
    if chosen_chart == "Histogram":
        fig = px.bar(sentiment_count_df, x='Sentiment', y='Tweets', color='Sentiment',
                     color_discrete_sequence=accent, text='Tweets', height=390,
                     template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    elif chosen_chart == "Pie Chart":
        fig = px.pie(sentiment_count_df, values='Tweets', names='Sentiment',
                     color_discrete_sequence=accent, hole=0, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    elif chosen_chart == "Donut":
        fig = px.pie(sentiment_count_df, values='Tweets', names='Sentiment',
                     color_discrete_sequence=accent, hole=0.48, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    elif chosen_chart == "Vertical Bar":
        fig = px.bar(sentiment_count_df, y='Sentiment', x='Tweets', color='Sentiment',
                     orientation='h', text='Tweets', color_discrete_sequence=accent, height=364,
                     template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Sample Live Opinions")
    sel_sentiment = st.radio(
        "Pick a sentiment for a random tweet:",
        ['positive', 'neutral', 'negative'],
        horizontal=True, key='rand'
    )
    tweet = data.query('airline_sentiment == @sel_sentiment')['text'].sample(n=1).iat[0]
    st.markdown(
        f"<b style='color:{theme['primary']}'>Random {sel_sentiment.title()} tweet:</b><br><i>{tweet}</i>",
        unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown(f"""<div style='display:flex;align-items:center;gap:10px;font-size:1.18em;margin-bottom:.33em;'>{theme['svg']}<span>Time & Location</span></div>""", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("When and Where Are Users Tweeting?")
    hour = st.slider('Select hour of day (local timezone):', 0, 23, value=12)
    by_hour_data = data[data['tweet_created'].dt.hour == hour]
    c3, c4 = st.columns([2, 1])
    with c3:
        st.markdown(
            f"<span style='color:{theme['primary']};'><b>{len(by_hour_data)} tweets</b> between {hour}:00 and {(hour+1)%24}:00</span>",
            unsafe_allow_html=True)
        map_data = by_hour_data.dropna(subset=['lat', 'lon'])
        if not map_data.empty:
            st.map(map_data[['lat', 'lon']])
            fig_heat = px.density_mapbox(
                map_data, lat='lat', lon='lon', z=None, radius=26,
                center=dict(lat=39, lon=-97),
                zoom=3.2, mapbox_style="carto-darkmatter", height=330)
            fig_heat.update_layout(paper_bgcolor=theme['bg'], plot_bgcolor=theme['bg'])
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("No location data for this hour.")
    with c4:
        st.markdown("**Hour/day tweet patterns**")
        timegr = data.groupby([data['tweet_created'].dt.hour, data['tweet_created'].dt.dayofweek]).size().unstack(fill_value=0)
        st.dataframe(timegr, use_container_width=True)
        st.caption("Rows = Hour (0–23); Cols = Day of week (Mon=0, Sun=6)")
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown(f"""<div style='display:flex;align-items:center;gap:10px;font-size:1.18em;margin-bottom:.33em;'>{theme['svg']}<span>Trend & Comparison</span></div>""", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Trends & Airline Comparison")
    airlines = sorted(data['airline'].unique())
    choice = st.multiselect('Compare airlines', airlines, default=[theme_choice] if theme_choice in airlines else [])
    if choice:
        ch_data = data[data.airline.isin(choice)]
        st.subheader("Sentiment Breakdown by Airline")
        fig1 = px.histogram(
            ch_data, x='airline', color='airline_sentiment', barmode='group',
            labels={'airline_sentiment': 'Sentiment', 'airline': 'Airline'},
            color_discrete_sequence=[theme['primary'], theme['secondary'], "#888"],
            height=420, text_auto=True, template='plotly_dark'
        )
        st.plotly_chart(fig1, use_container_width=True)
        if 'negativereason' in data.columns:
            radar_data = ch_data[ch_data.airline_sentiment == 'negative']
            vcount = radar_data.groupby(['airline', 'negativereason']).size().unstack(fill_value=0)
            vcount = vcount.div(vcount.sum(axis=1), axis=0)
            if vcount.shape[1] > 1:
                fig_radar = go.Figure()
                for idx, row in vcount.iterrows():
                    fig_radar.add_trace(go.Scatterpolar(
                        r=row.values.tolist() + [row.values[0]],
                        theta=list(row.index) + [list(row.index)[0]],
                        fill='toself', name=idx))
                fig_radar.update_layout(
                    polar=dict(bgcolor='#23263a', radialaxis=dict(visible=True, gridcolor='#848ba3', color='white')),
                    paper_bgcolor='#191C28', plot_bgcolor='#191C28',
                    font=dict(color='white'),
                    legend=dict(font=dict(color='white')),
                    title="Normalized Complaint Reasons (Radar)",
                    height=440)
                st.plotly_chart(fig_radar, use_container_width=True)
    if st.checkbox("Show Sentiment Trend Over Time", value=True):
        st.subheader("Sentiment Trend Over Time")
        time_data = data.copy()
        time_data['date'] = time_data['tweet_created'].dt.date
        trend = time_data.groupby(['date', 'airline_sentiment']).size().reset_index(name='count')
        fig_trend = px.line(
            trend, x='date', y='count', color='airline_sentiment',
            labels={'count': 'Number of Tweets', 'date': 'Date', 'airline_sentiment': 'Sentiment'},
            color_discrete_sequence=[theme['primary'], theme['secondary'], "#848ba3"],
            markers=True, template='plotly_dark'
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab4:
    st.markdown(f"""<div style='display:flex;align-items:center;gap:10px;font-size:1.18em;margin-bottom:.33em;'>{theme['svg']}<span>Text Insights</span></div>""", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Textual Passenger Insights")
    wc_sentiment = st.radio('Word cloud for sentiment:', ('positive', 'neutral', 'negative'), horizontal=True, key='wcs')
    wc_df = data[data['airline_sentiment'] == wc_sentiment]
    words = ' '.join(wc_df['text'].astype(str))
    processed_words = ' '.join([w for w in words.split() if 'http' not in w and not w.startswith('@')])
    wordcloud = WordCloud(
        stopwords=STOPWORDS, background_color='#23263a', height=340, width=720, colormap="twilight", mode='RGBA'
    ).generate(processed_words)
    st.markdown(f"#### Word Cloud for <span style='color:{theme['primary']}'>{wc_sentiment.capitalize()}</span> Sentiment", unsafe_allow_html=True)
    fig4, ax4 = plt.subplots(figsize=(7, 2.5), facecolor='#23263a')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    fig4.patch.set_facecolor('#23263a')
    st.pyplot(fig4)
    if st.checkbox("Show Top Words by Sentiment"):
        st.subheader("Top 10 Most Frequent Words by Sentiment")
        cols = st.columns(3)
        for idx, sentiment in enumerate(['positive', 'neutral', 'negative']):
            df = data[data['airline_sentiment'] == sentiment]
            words = ' '.join(df['text'].astype(str)).lower().split()
            words = [w for w in words if w.isalpha() and w not in STOPWORDS and not w.startswith('@')]
            common_words = Counter(words).most_common(10)
            with cols[idx]:
                st.markdown(f"<b style='color:{theme['primary']}'>{sentiment.capitalize()}</b>", unsafe_allow_html=True)
                st.table(pd.DataFrame(common_words, columns=['Word', 'Count']))
    st.markdown("</div>", unsafe_allow_html=True)

with tab5:
    st.markdown(f"""<div style='display:flex;align-items:center;gap:10px;font-size:1.18em;margin-bottom:.33em;'>{theme['svg']}<span>Alerts & Drilldown</span></div>""", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Root-Cause, Alerts & Data Download")
    if 'negativereason' in data.columns:
        st.markdown("#### Negative Tweet Reasons by Airline")
        neg_data = data[data['airline_sentiment'] == 'negative']
        reason_count = neg_data.groupby(['airline', 'negativereason']).size().reset_index(name='count')
        fig_reason = px.bar(
            reason_count, x='airline', y='count', color='negativereason',
            labels={'count': 'Number of Tweets', 'airline': 'Airline', 'negativereason': 'Reason'},
            height=500, template='plotly_dark', color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_reason.update_layout(font_color='white')
        st.plotly_chart(fig_reason, use_container_width=True)
        now_naive = pd.Timestamp.now().replace(tzinfo=None)
        airlines = sorted(data['airline'].unique())
        for a in airlines:
            airline_neg = neg_data[neg_data.airline == a]
            if len(airline_neg) > 10 and not airline_neg.empty:
                max_time = airline_neg['tweet_created'].max()
                if isinstance(max_time, pd.Timestamp):
                    max_time = max_time.replace(tzinfo=None)
                if max_time > (now_naive - pd.Timedelta('1 day')):
                    st.warning(f"Spike in negative tweets for {a}: {len(airline_neg)} in the past 24h!")
    else:
        st.info("No 'negativereason' column in this data.")
    with st.expander("Download Cleaned Data"):
        st.download_button(
            label="Download CSV", data=data.to_csv(index=False).encode('utf-8'),
            file_name='cleaned_airline_tweets.csv', mime='text/csv'
        )
        st.caption("Includes airline, sentiment, location/geocoords, and complaint reason if available.")
    st.markdown("</div>", unsafe_allow_html=True)
