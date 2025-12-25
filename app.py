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
    page_title="Tweet Sentiment Analytics",
    page_icon="📈",
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

# Custom styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 5px;
        text-align: center;
    }
    .positive { color: #2ecc71; }
    .negative { color: #e74c3c; }
    .neutral { color: #3498db; }
</style>
""", unsafe_allow_html=True)

# App header
st.title("📈 Tweet Sentiment Analytics Dashboard")
st.markdown("**Phase 4: Added interactive visualizations**")

# Sidebar controls
with st.sidebar:
    st.markdown("## ⚙️ Dashboard Controls")
    
    view_mode = st.radio(
        "View Mode:",
        ["Live Analysis", "History Dashboard", "Comparison View"],
        index=0
    )
    
    chart_type = st.selectbox(
        "Chart Style:",
        ["Bar Chart", "Line Chart", "Pie Chart", "All Charts"],
        index=0
    )
    
    if st.button("🔄 Reset Dashboard", type="secondary"):
        if 'analysis_history' in st.session_state:
            st.session_state.analysis_history = pd.DataFrame()
            st.success("Dashboard reset!")

# Initialize session state for data storage
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = pd.DataFrame(columns=[
        'id', 'timestamp', 'tweet_short', 'textblob_score', 
        'textblob_sentiment', 'vader_score', 'vader_sentiment'
    ])

# MAIN CONTENT AREA
if view_mode == "Live Analysis":
    # Live analysis tab
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### ✍️ Live Analysis")
        tweet = st.text_area(
            "Enter tweet to analyze:",
            "The customer support team was incredibly helpful and resolved my issue quickly! 👍",
            height=120
        )
        
        if st.button("🚀 Analyze & Visualize", type="primary", use_container_width=True):
            with st.spinner("Analyzing and creating visualizations..."):
                time.sleep(0.5)
                
                # TextBlob analysis
                tb_analysis = TextBlob(tweet)
                tb_score = tb_analysis.sentiment.polarity
                tb_sentiment = 'positive' if tb_score > 0.1 else 'negative' if tb_score < -0.1 else 'neutral'
                
                # VADER analysis if available
                if nltk_available:
                    vader_scores = sia.polarity_scores(tweet)
                    vader_score = vader_scores['compound']
                    vader_sentiment = 'positive' if vader_score >= 0.05 else 'negative' if vader_score <= -0.05 else 'neutral'
                else:
                    vader_score = 0
                    vader_sentiment = 'N/A'
                
                # Store in history
                new_id = len(st.session_state.analysis_history) + 1
                new_entry = pd.DataFrame([{
                    'id': new_id,
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'tweet_short': tweet[:40] + '...' if len(tweet) > 40 else tweet,
                    'textblob_score': tb_score,
                    'textblob_sentiment': tb_sentiment,
                    'vader_score': vader_score,
                    'vader_sentiment': vader_sentiment
                }])
                
                st.session_state.analysis_history = pd.concat(
                    [st.session_state.analysis_history, new_entry], ignore_index=True
                )
                
                st.success(f"✅ Analysis #{new_id} added to dashboard!")
                
                # Show immediate results
                st.markdown("---")
                st.markdown("### 📊 Immediate Results")
                
                result_cols = st.columns(2)
                with result_cols[0]:
                    st.metric("TextBlob Score", f"{tb_score:.3f}", f"Sentiment: {tb_sentiment}")
                with result_cols[1]:
                    if nltk_available:
                        st.metric("VADER Score", f"{vader_score:.3f}", f"Sentiment: {vader_sentiment}")
    
    with col2:
        st.markdown("### 📝 Quick Examples")
        examples = [
            ("😊", "Positive", "Absolutely love this product! Five stars!"),
            ("😠", "Negative", "Worst service ever. Never buying again."),
            ("😐", "Neutral", "The package arrived on time."),
            ("🔥", "Strong", "THIS IS AMAZING! BEST PURCHASE EVER!!!"),
            ("🤔", "Mixed", "Good quality but overpriced.")
        ]
        
        for emoji, label, text in examples:
            if st.button(f"{emoji} {label}", key=f"ex_{label}"):
                st.session_state.example_text = text
                st.rerun()

elif view_mode == "History Dashboard" and not st.session_state.analysis_history.empty:
    # History visualization tab
    st.markdown("## 📈 Analysis History Visualizations")
    
    # Data metrics
    st.markdown("### 📊 Dashboard Metrics")
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        total = len(st.session_state.analysis_history)
        st.metric("Total Analyses", total)
    
    with metric_cols[1]:
        positive = len(st.session_state.analysis_history[
            st.session_state.analysis_history['textblob_sentiment'] == 'positive'
        ])
        st.metric("Positive", positive, f"{positive/total*100:.1f}%")
    
    with metric_cols[2]:
        avg_tb = st.session_state.analysis_history['textblob_score'].mean()
        st.metric("Avg. TextBlob", f"{avg_tb:.3f}")
    
    with metric_cols[3]:
        if nltk_available and 'vader_score' in st.session_state.analysis_history.columns:
            avg_vader = st.session_state.analysis_history['vader_score'].mean()
            st.metric("Avg. VADER", f"{avg_vader:.3f}")
    
    # Create visualizations based on selection
    if chart_type in ["Bar Chart", "All Charts"]:
        st.markdown("### 📊 Sentiment Distribution")
        
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        sentiment_counts = st.session_state.analysis_history['textblob_sentiment'].value_counts()
        
        colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#3498db'}
        bar_colors = [colors.get(s, '#95a5a6') for s in sentiment_counts.index]
        
        bars = ax1.bar(sentiment_counts.index, sentiment_counts.values, color=bar_colors, edgecolor='black')
        ax1.set_title('Sentiment Distribution (TextBlob)', fontweight='bold')
        ax1.set_xlabel('Sentiment')
        ax1.set_ylabel('Count')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, sentiment_counts.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig1)
    
    if chart_type in ["Line Chart", "All Charts"] and len(st.session_state.analysis_history) > 1:
        st.markdown("### 📈 Score Trend Over Time")
        
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        
        # Plot TextBlob scores
        history = st.session_state.analysis_history.copy()
        ax2.plot(range(len(history)), history['textblob_score'], 
                label='TextBlob', marker='o', color='#3498db', linewidth=2)
        
        # Plot VADER scores if available
        if nltk_available and 'vader_score' in history.columns:
            ax2.plot(range(len(history)), history['vader_score'], 
                    label='VADER', marker='s', color='#2ecc71', linewidth=2)
        
        ax2.set_title('Sentiment Score Trend', fontweight='bold')
        ax2.set_xlabel('Analysis Number')
        ax2.set_ylabel('Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([-1.1, 1.1])
        
        st.pyplot(fig2)
    
    if chart_type in ["Pie Chart", "All Charts"]:
        st.markdown("### 🥧 Sentiment Proportions")
        
        fig3, ax3 = plt.subplots(figsize=(8, 8))
        sentiment_counts = st.session_state.analysis_history['textblob_sentiment'].value_counts()
        
        colors = ['#2ecc71', '#e74c3c', '#3498db'][:len(sentiment_counts)]
        wedges, texts, autotexts = ax3.pie(
            sentiment_counts.values, 
            labels=sentiment_counts.index,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            explode=[0.05] * len(sentiment_counts)
        )
        
        ax3.set_title('Sentiment Proportions', fontweight='bold')
        
        # Style the percentages
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        st.pyplot(fig3)
    
    # Data table
    with st.expander("📋 View Raw Data"):
        st.dataframe(st.session_state.analysis_history, use_container_width=True)
        
        # Download option
        csv = st.session_state.analysis_history.to_csv(index=False)
        st.download_button(
            label="📥 Download Full History (CSV)",
            data=csv,
            file_name="sentiment_analytics_full.csv",
            mime="text/csv",
            use_container_width=True
        )

elif view_mode == "Comparison View" and nltk_available and not st.session_state.analysis_history.empty:
    # Method comparison tab
    st.markdown("## 🤝 Method Comparison Analysis")
    
    if 'vader_score' in st.session_state.analysis_history.columns:
        # Create comparison visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Scatter plot comparison
        history = st.session_state.analysis_history
        ax1.scatter(history['textblob_score'], history['vader_score'], 
                   alpha=0.6, color='#9b59b6', edgecolors='black')
        ax1.set_title('TextBlob vs VADER Score Comparison', fontweight='bold')
        ax1.set_xlabel('TextBlob Score')
        ax1.set_ylabel('VADER Score')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # Agreement analysis
        agreement = (history['textblob_sentiment'] == history['vader_sentiment']).mean() * 100
        ax2.bar(['Agree', 'Disagree'], 
                [agreement, 100 - agreement], 
                color=['#2ecc71', '#e74c3c'])
        ax2.set_title('Method Agreement', fontweight='bold')
        ax2.set_ylabel('Percentage (%)')
        ax2.set_ylim([0, 100])
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, value in enumerate([agreement, 100 - agreement]):
            ax2.text(i, value + 2, f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Agreement statistics
        st.info(f"📊 Methods agree on **{agreement:.1f}%** of analyses")
        
        # Show disagreements
        disagreements = history[history['textblob_sentiment'] != history['vader_sentiment']]
        if not disagreements.empty:
            st.markdown("### 🔍 Sample Disagreements")
            st.dataframe(disagreements[['tweet_short', 'textblob_sentiment', 'vader_sentiment']].head(), 
                        use_container_width=True)

else:
    # Empty state
    st.info("📋 No analysis history yet. Switch to 'Live Analysis' to start analyzing tweets!")

# Deployment progress footer
st.markdown("---")
st.markdown("### 🚀 Deployment Progress")
progress_cols = st.columns(5)

with progress_cols[0]:
    st.success("✅ Phase 1\nTextBlob")
with progress_cols[1]:
    st.success("✅ Phase 2\nVADER/NLTK")
with progress_cols[2]:
    st.success("✅ Phase 3\nPandas")
with progress_cols[3]:
    st.info("🔄 Phase 4\nVisualizations")
with progress_cols[4]:
    st.info("Next: ML Model")

# Refresh dashboard
if st.sidebar.button("🔄 Update Charts", use_container_width=True):
    st.rerun()
