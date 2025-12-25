import streamlit as st
from textblob import TextBlob
import nltk
import time

# Download NLTK data (VADER lexicon) - runs once
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk_available = True
except:
    nltk_available = False

st.set_page_config(page_title="Tweet Analyzer", page_icon="📊", layout="wide")
st.title("📊 Advanced Tweet Sentiment Analyzer")
st.write("Now with **TextBlob** AND **VADER** analysis!")

# Sidebar settings
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    analysis_method = st.radio(
        "Choose analysis method:",
        ["TextBlob", "VADER", "Both"],
        index=2
    )
    
    if not nltk_available:
        st.warning("⚠️ NLTK/VADER not available")
        st.info("Make sure nltk==3.8.1 is in requirements.txt")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    # Text input
    tweet = st.text_area("Enter a tweet:", "I love this product! It's amazing!", height=100)
    
    # Analyze button
    analyze_clicked = st.button("🚀 Analyze Sentiment", type="primary")

with col2:
    # Example tweets
    st.markdown("### Quick Examples:")
    examples = [
        ("😊 Positive", "Great service! Will definitely come back."),
        ("😠 Negative", "Terrible experience, never again."),
        ("😐 Neutral", "It was okay, nothing special."),
        ("✈️ Travel", "Flight delayed 3 hours with no explanation!"),
        ("📱 Tech", "The new smartphone features are incredible!")
    ]
    
    for emoji, example in examples:
        if st.button(f"{emoji}", key=f"btn_{example[:5]}"):
            tweet = example

# Perform analysis
if analyze_clicked and tweet:
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    
    results = {}  # Store results from both methods
    
    # TextBlob Analysis
    if analysis_method in ["TextBlob", "Both"]:
        with st.spinner("Analyzing with TextBlob..."):
            time.sleep(0.5)  # Simulate processing
            
            analysis = TextBlob(tweet)
            polarity = analysis.sentiment.polarity
            
            # Determine sentiment
            if polarity > 0.1:
                sentiment = "POSITIVE"
                color = "#2ecc71"
            elif polarity < -0.1:
                sentiment = "NEGATIVE"
                color = "#e74c3c"
            else:
                sentiment = "NEUTRAL"
                color = "#3498db"
            
            # Display TextBlob results
            st.markdown("### 📘 TextBlob Analysis")
            col_tb1, col_tb2 = st.columns([1, 3])
            
            with col_tb1:
                st.markdown(f'<div style="color:{color}; font-size:1.2rem; font-weight:bold;">{sentiment}</div>', 
                          unsafe_allow_html=True)
            
            with col_tb2:
                progress_value = (polarity + 1) / 2
                st.progress(progress_value, text=f"Polarity: {polarity:.3f}")
            
            results['textblob'] = {'sentiment': sentiment, 'score': polarity}
    
    # VADER Analysis (if available)
    if analysis_method in ["VADER", "Both"] and nltk_available:
        with st.spinner("Analyzing with VADER (social media optimized)..."):
            time.sleep(0.5)
            
            try:
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                sia = SentimentIntensityAnalyzer()
                scores = sia.polarity_scores(tweet)
                compound = scores['compound']
                
                # Determine sentiment
                if compound >= 0.05:
                    sentiment = "POSITIVE"
                    color = "#2ecc71"
                elif compound <= -0.05:
                    sentiment = "NEGATIVE"
                    color = "#e74c3c"
                else:
                    sentiment = "NEUTRAL"
                    color = "#3498db"
                
                # Display VADER results
                st.markdown("### 🎯 VADER Analysis")
                col_v1, col_v2 = st.columns([1, 3])
                
                with col_v1:
                    st.markdown(f'<div style="color:{color}; font-size:1.2rem; font-weight:bold;">{sentiment}</div>', 
                              unsafe_allow_html=True)
                
                with col_v2:
                    progress_value = (compound + 1) / 2
                    st.progress(progress_value, text=f"Compound: {compound:.3f}")
                
                # Show detailed scores
                with st.expander("🔍 View VADER Detailed Scores"):
                    st.write(f"**Positive:** {scores['pos']:.3f}")
                    st.write(f"**Negative:** {scores['neg']:.3f}")
                    st.write(f"**Neutral:** {scores['neu']:.3f}")
                    st.write(f"**Compound:** {scores['compound']:.3f}")
                
                results['vader'] = {'sentiment': sentiment, 'score': compound}
                
            except Exception as e:
                st.error(f"VADER analysis failed: {e}")
    
    # Compare methods if both were used
    if analysis_method == "Both" and len(results) >= 2:
        st.markdown("---")
        st.markdown("### 🤝 Method Comparison")
        
        if results.get('textblob') and results.get('vader'):
            tb_sentiment = results['textblob']['sentiment']
            vader_sentiment = results['vader']['sentiment']
            
            if tb_sentiment == vader_sentiment:
                st.success(f"✅ **Both methods agree:** {tb_sentiment}")
            else:
                st.warning(f"⚠️ **Methods disagree:** TextBlob={tb_sentiment}, VADER={vader_sentiment}")
    
    # Show original tweet
    st.markdown("---")
    with st.expander("📝 View Original Tweet"):
        st.info(tweet)

# Footer with deployment status
st.markdown("---")
st.success("✅ **Phase 2: NLTK/VADER added!**")
st.info("Next: Add pandas & visualizations in Phase 3")

# Refresh button in sidebar
if st.sidebar.button("🔄 Refresh App"):
    st.rerun()
