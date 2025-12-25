import streamlit as st
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import time
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Tweet Sentiment Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize NLTK
@st.cache_resource
def initialize_nltk():
    try:
        nltk.download('vader_lexicon', quiet=True)
        return SentimentIntensityAnalyzer(), True
    except:
        return None, False

sia, nltk_available = initialize_nltk()

# App title
st.title("📊 Tweet Sentiment Dashboard")
st.write("**Phase 3: Added pandas for data storage**")

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    analysis_method = st.radio("Method:", ["TextBlob", "VADER", "Both"], index=2)
    
    if st.button("🔄 Clear History"):
        if 'analysis_history' in st.session_state:
            st.session_state.analysis_history = pd.DataFrame()
            st.success("History cleared!")

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = pd.DataFrame(columns=[
        'timestamp', 'tweet', 'textblob_sentiment', 'textblob_score'
    ])

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    tweet = st.text_area(
        "Enter tweet:",
        "Just tried the new coffee shop - amazing latte! ☕️",
        height=100
    )
    
    if st.button("🚀 Analyze", type="primary"):
        # TextBlob analysis
        analysis = TextBlob(tweet)
        polarity = analysis.sentiment.polarity
        
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # Store in history
        new_entry = pd.DataFrame([{
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'tweet': tweet[:50] + '...' if len(tweet) > 50 else tweet,
            'textblob_sentiment': sentiment,
            'textblob_score': polarity
        }])
        
        st.session_state.analysis_history = pd.concat(
            [st.session_state.analysis_history, new_entry], ignore_index=True
        )
        
        # Show results
        st.success(f"✅ Analysis added! Sentiment: {sentiment.upper()} (Score: {polarity:.3f})")

with col2:
    st.markdown("### Examples")
    examples = [
        "Great product! Highly recommend.",
        "Terrible service, very disappointed.",
        "It's okay, nothing special.",
        "Loving the new features! 🔥"
    ]
    
    for example in examples:
        if st.button(f"📝 {example[:20]}..."):
            st.session_state.tweet_input = example
            st.rerun()

# Show history if exists
if not st.session_state.analysis_history.empty:
    st.markdown("---")
    st.markdown("## 📋 Analysis History")
    
    # Display as table
    st.dataframe(st.session_state.analysis_history, use_container_width=True)
    
    # Simple statistics
    st.markdown("### 📊 Statistics")
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    
    with col_stat1:
        total = len(st.session_state.analysis_history)
        st.metric("Total Analyses", total)
    
    with col_stat2:
        positive = len(st.session_state.analysis_history[
            st.session_state.analysis_history['textblob_sentiment'] == 'positive'
        ])
        st.metric("Positive", positive)
    
    with col_stat3:
        avg_score = st.session_state.analysis_history['textblob_score'].mean()
        st.metric("Average Score", f"{avg_score:.3f}")
    
    # Download option
    csv = st.session_state.analysis_history.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="sentiment_history.csv",
        mime="text/csv"
    )

# Deployment progress
st.markdown("---")
st.success("✅ **Deployment Strategy:**")
st.write("1. ✅ Basic app with Streamlit")
st.write("2. ✅ Added TextBlob sentiment")
st.write("3. ✅ Added VADER/NLTK")
st.write("4. 🔄 **Adding pandas (current)**")
st.write("5. Next: Add visualizations after pandas works")
