import streamlit as st
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime
import numpy as np

# Set page config
st.set_page_config(
    page_title="Tweet Sentiment Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize NLTK (with error handling)
@st.cache_resource
def initialize_nltk():
    try:
        nltk.download('vader_lexicon', quiet=True)
        return SentimentIntensityAnalyzer(), True
    except:
        return None, False

sia, nltk_available = initialize_nltk()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1DA1F2;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .positive { color: #2ecc71; font-weight: bold; }
    .negative { color: #e74c3c; font-weight: bold; }
    .neutral { color: #3498db; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">📊 Tweet Sentiment Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Dashboard Settings")
    
    analysis_method = st.radio(
        "Analysis Method:",
        ["TextBlob", "VADER", "Compare Both"],
        index=2
    )
    
    show_visualizations = st.checkbox("Show Visualizations", value=True)
    
    st.markdown("---")
    st.markdown("### 📈 Deployment Progress")
    st.progress(0.6, text="Phase 3: Adding visualizations")
    
    if st.button("🔄 Clear History"):
        if 'analysis_history' in st.session_state:
            st.session_state.analysis_history = pd.DataFrame()
            st.success("History cleared!")

# Initialize session state for storing analyses
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = pd.DataFrame(columns=[
        'timestamp', 'tweet', 'textblob_sentiment', 'textblob_score',
        'vader_sentiment', 'vader_score', 'vader_compound'
    ])

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### ✍️ Analyze New Tweet")
    tweet_input = st.text_area(
        "Enter tweet text:",
        "Just tried the new coffee shop downtown - absolutely amazing latte and friendly staff! ☕️😊",
        height=120,
        key="tweet_input"
    )
    
    analyze_btn = st.button("🚀 Analyze & Add to Dashboard", type="primary", use_container_width=True)

with col2:
    st.markdown("### 📝 Example Tweets")
    
    examples = pd.DataFrame({
        'Sentiment': ['Positive', 'Negative', 'Neutral', 'Mixed', 'With Emoji'],
        'Tweet': [
            "Fantastic product! Exceeded all my expectations.",
            "Worst customer service I've ever experienced.",
            "The meeting was scheduled for 3 PM.",
            "The food was great but the service was slow.",
            "Loving the new update! 🔥🚀 #innovation"
        ]
    })
    
    selected_example = st.selectbox(
        "Load example:",
        examples['Tweet'],
        index=0
    )
    
    if st.button("📋 Use This Example"):
        st.session_state.tweet_input = selected_example
        st.rerun()

# Analysis functions
def analyze_textblob(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0.1:
        sentiment = 'positive'
    elif polarity < -0.1:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return sentiment, polarity, analysis.sentiment.subjectivity

def analyze_vader(text):
    if not nltk_available:
        return 'error', 0, {}
    
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    
    if compound >= 0.05:
        sentiment = 'positive'
    elif compound <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return sentiment, compound, scores

# Perform analysis
if analyze_btn and tweet_input:
    with st.spinner("🔍 Analyzing sentiment..."):
        time.sleep(0.5)  # Simulate processing
        
        # Get analyses
        tb_sentiment, tb_score, tb_subjectivity = analyze_textblob(tweet_input)
        
        if nltk_available:
            vader_sentiment, vader_score, vader_scores = analyze_vader(tweet_input)
        else:
            vader_sentiment, vader_score, vader_scores = 'unavailable', 0, {}
        
        # Store in history
        new_entry = pd.DataFrame([{
            'timestamp': datetime.now(),
            'tweet': tweet_input[:100] + '...' if len(tweet_input) > 100 else tweet_input,
            'textblob_sentiment': tb_sentiment,
            'textblob_score': tb_score,
            'vader_sentiment': vader_sentiment if nltk_available else 'N/A',
            'vader_score': vader_score if nltk_available else 0,
            'vader_compound': vader_scores.get('compound', 0) if nltk_available else 0
        }])
        
        st.session_state.analysis_history = pd.concat(
            [st.session_state.analysis_history, new_entry], ignore_index=True
        )
        
        # Display results
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        # Create columns for results
        result_cols = st.columns(2)
        
        with result_cols[0]:
            st.markdown("### 📘 TextBlob")
            st.markdown(f'<div class="{tb_sentiment}">{tb_sentiment.upper()}</div>', unsafe_allow_html=True)
            st.metric("Polarity Score", f"{tb_score:.3f}")
            st.progress((tb_score + 1) / 2, text=f"Subjectivity: {tb_subjectivity:.2f}")
        
        with result_cols[1]:
            if nltk_available:
                st.markdown("### 🎯 VADER")
                st.markdown(f'<div class="{vader_sentiment}">{vader_sentiment.upper()}</div>', unsafe_allow_html=True)
                st.metric("Compound Score", f"{vader_score:.3f}")
                st.progress((vader_score + 1) / 2)
                
                with st.expander("VADER Details"):
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("Positive", f"{vader_scores.get('pos', 0):.3f}")
                    with cols[1]:
                        st.metric("Negative", f"{vader_scores.get('neg', 0):.3f}")
                    with cols[2]:
                        st.metric("Neutral", f"{vader_scores.get('neu', 0):.3f}")
                    with cols[3]:
                        st.metric("Compound", f"{vader_scores.get('compound', 0):.3f}")
            else:
                st.warning("VADER not available")
        
        # Method comparison
        if nltk_available and analysis_method == "Compare Both":
            st.markdown("---")
            st.markdown("### 🤝 Method Comparison")
            
            if tb_sentiment == vader_sentiment:
                st.success(f"✅ Methods agree: **{tb_sentiment.upper()}**")
            else:
                st.warning(f"⚠️ Methods disagree: TextBlob={tb_sentiment.upper()}, VADER={vader_sentiment.upper()}")

# Dashboard Visualizations
if show_visualizations and not st.session_state.analysis_history.empty:
    st.markdown("---")
    st.markdown("## 📈 Analysis Dashboard")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 History", "📈 Trends", "📋 Data"])
    
    with tab1:
        # Display recent analyses
        st.markdown("### Recent Analyses")
        st.dataframe(
            st.session_state.analysis_history.sort_values('timestamp', ascending=False).head(10),
            use_container_width=True
        )
    
    with tab2:
        # Create visualizations
        if len(st.session_state.analysis_history) > 1:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot 1: Sentiment scores over time
            history = st.session_state.analysis_history.copy()
            history['time'] = pd.to_datetime(history['timestamp']).dt.strftime('%H:%M')
            
            axes[0].plot(history['time'], history['textblob_score'], 
                        label='TextBlob', marker='o', color='#3498db')
            
            if nltk_available:
                axes[0].plot(history['time'], history['vader_score'], 
                            label='VADER', marker='s', color='#2ecc71')
            
            axes[0].set_title('Sentiment Scores Over Time')
            axes[0].set_xlabel('Time')
            axes[0].set_ylabel('Score')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].tick_params(axis='x', rotation=45)
            
            # Plot 2: Sentiment distribution
            if len(history) > 0:
                sentiment_counts = history['textblob_sentiment'].value_counts()
                colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#3498db'}
                color_list = [colors.get(s, '#95a5a6') for s in sentiment_counts.index]
                
                axes[1].bar(sentiment_counts.index, sentiment_counts.values, color=color_list)
                axes[1].set_title('Sentiment Distribution')
                axes[1].set_xlabel('Sentiment')
                axes[1].set_ylabel('Count')
                axes[1].grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for i, count in enumerate(sentiment_counts.values):
                    axes[1].text(i, count + 0.1, str(count), ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Analyze more tweets to see trends")
    
    with tab3:
        # Data summary
        st.markdown("### Data Summary")
        
        if len(st.session_state.analysis_history) > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total = len(st.session_state.analysis_history)
                st.metric("Total Analyses", total)
            
            with col2:
                positive = len(st.session_state.analysis_history[
                    st.session_state.analysis_history['textblob_sentiment'] == 'positive'
                ])
                st.metric("Positive", positive, f"{positive/total*100:.1f}%")
            
            with col3:
                avg_score = st.session_state.analysis_history['textblob_score'].mean()
                st.metric("Avg. Score", f"{avg_score:.3f}")
        
        # Download data option
        if not st.session_state.analysis_history.empty:
            csv = st.session_state.analysis_history.to_csv(index=False)
            st.download_button(
                label="📥 Download Analysis History",
                data=csv,
                file_name="sentiment_analysis_history.csv",
                mime="text/csv"
            )

# Deployment progress footer
st.markdown("---")
col_prog1, col_prog2, col_prog3 = st.columns(3)

with col_prog1:
    st.success("✅ Phase 1: TextBlob")
with col_prog2:
    st.success("✅ Phase 2: VADER")
with col_prog3:
    st.info("🔄 Phase 3: Pandas & Visualizations")

# Refresh button
if st.sidebar.button("🔄 Refresh Dashboard", use_container_width=True):
    st.rerun()
