import streamlit as st
from textblob import TextBlob

st.set_page_config(page_title="Tweet Analyzer", page_icon="📊")
st.title("📊 Tweet Sentiment Analyzer")
st.write("Now with TextBlob sentiment analysis!")

# Text input
tweet = st.text_area("Enter a tweet:", "I love this product! It's amazing!", height=100)

# Example tweets
st.markdown("### Quick Examples:")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("😊 Positive"):
        tweet = "Great service! Will definitely come back."
with col2:
    if st.button("😠 Negative"):
        tweet = "Terrible experience, never again."
with col3:
    if st.button("😐 Neutral"):
        tweet = "It was okay, nothing special."

# Analyze button
if st.button("🚀 Analyze Sentiment", type="primary"):
    if tweet:
        # Analyze with TextBlob
        analysis = TextBlob(tweet)
        polarity = analysis.sentiment.polarity  # -1 to 1
        
        # Determine sentiment
        if polarity > 0.1:
            sentiment = "POSITIVE 😊"
            color = "green"
        elif polarity < -0.1:
            sentiment = "NEGATIVE 😠"
            color = "red"
        else:
            sentiment = "NEUTRAL 😐"
            color = "blue"
        
        # Display results
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sentiment", sentiment)
        with col2:
            st.metric("Polarity Score", f"{polarity:.3f}")
        
        # Progress bar visualization
        st.progress((polarity + 1) / 2)
        st.caption(f"Polarity: {polarity:.3f} (Range: -1.0 to 1.0)")
        
        # Show original tweet
        st.markdown("### Original Tweet")
        st.info(tweet)
        
        if polarity > 0.5:
            st.balloons()

st.markdown("---")
st.success("✅ Phase 1: TextBlob sentiment analysis added!")
