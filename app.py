import streamlit as st
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PHASE 1: BASIC APP WORKING ✅
# ============================================
# Set page config
st.set_page_config(
    page_title="Tweet Sentiment Analytics",
    page_icon="📈",
    layout="wide"
)

# ============================================
# PHASE 2: ADD TEXTBLOB ✅
# ============================================
# TextBlob functionality is already integrated

# ============================================
# PHASE 3: ADD NLTK/VADER ✅
# ============================================
# Initialize NLTK resources
@st.cache_resource
def initialize_nltk():
    """Initialize NLTK and download required resources"""
    try:
        # Download required NLTK data
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
        
        # Initialize VADER sentiment analyzer
        sia = SentimentIntensityAnalyzer()
        
        return sia, True
    except Exception as e:
        st.sidebar.warning(f"⚠️ NLTK initialization issue: {str(e)}")
        return None, False

# Initialize NLTK/VADER
sia, nltk_available = initialize_nltk()

# ============================================
# PHASE 4: ADD PANDAS/NUMPY ✅
# ============================================
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
    .stDataFrame { 
        font-size: 0.9em;
        border-radius: 10px;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# App header
st.title("📈 Tweet Sentiment Analytics Dashboard")
st.markdown("**Phase 4: Enhanced pandas/numpy data analytics**")

# Sidebar controls
with st.sidebar:
    st.markdown("## ⚙️ Dashboard Controls")
    
    view_mode = st.radio(
        "View Mode:",
        ["Live Analysis", "History Dashboard", "Data Analytics", "Comparison View"],
        index=0
    )
    
    chart_type = st.selectbox(
        "Chart Style:",
        ["Bar Chart", "Line Chart", "Pie Chart", "Heatmap", "All Charts"],
        index=0
    )
    
    if st.button("🔄 Reset Dashboard", type="secondary"):
        if 'analysis_history' in st.session_state:
            st.session_state.analysis_history = pd.DataFrame()
            st.success("Dashboard reset!")
            st.rerun()

# ============================================
# PANDAS DATA STRUCTURES INITIALIZATION
# ============================================
if 'analysis_history' not in st.session_state:
    # Initialize with proper pandas DataFrame structure
    st.session_state.analysis_history = pd.DataFrame(columns=[
        'id', 
        'timestamp', 
        'tweet', 
        'tweet_short', 
        'textblob_score', 
        'textblob_sentiment', 
        'vader_score', 
        'vader_sentiment',
        'confidence',
        'word_count'
    ])

# ============================================
# NUMPY ANALYTICS FUNCTIONS
# ============================================
def calculate_advanced_metrics(df):
    """Calculate advanced analytics using numpy"""
    if df.empty:
        return {}
    
    metrics = {}
    
    # Basic statistics using numpy
    if 'textblob_score' in df.columns:
        scores = df['textblob_score'].astype(float).values
        metrics.update({
            'mean_score': np.mean(scores),
            'median_score': np.median(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'score_range': np.ptp(scores),  # Peak-to-peak
            'positive_ratio': np.sum(scores > 0.1) / len(scores) * 100,
            'negative_ratio': np.sum(scores < -0.1) / len(scores) * 100,
            'neutral_ratio': np.sum((scores >= -0.1) & (scores <= 0.1)) / len(scores) * 100,
        })
    
    if 'vader_score' in df.columns:
        vader_scores = df['vader_score'].astype(float).values
        metrics.update({
            'vader_mean': np.mean(vader_scores),
            'vader_std': np.std(vader_scores),
        })
    
    # Sentiment distribution using numpy
    if 'textblob_sentiment' in df.columns:
        sentiment_counts = df['textblob_sentiment'].value_counts()
        for sentiment, count in sentiment_counts.items():
            metrics[f'{sentiment}_count'] = count
    
    return metrics

def create_correlation_matrix(df):
    """Create correlation matrix between different metrics"""
    numeric_cols = ['textblob_score', 'vader_score', 'word_count', 'confidence']
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if len(available_cols) < 2:
        return None
    
    # Create correlation matrix using pandas
    corr_matrix = df[available_cols].corr()
    return corr_matrix

# ============================================
# MAIN CONTENT AREA
# ============================================
if view_mode == "Live Analysis":
    # Live analysis tab
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### ✍️ Live Analysis")
        
        # Check for example text
        if 'example_text' in st.session_state:
            default_text = st.session_state.example_text
            del st.session_state.example_text
        else:
            default_text = "The customer support team was incredibly helpful and resolved my issue quickly! 👍"
        
        tweet = st.text_area(
            "Enter tweet to analyze:",
            default_text,
            height=120,
            key="tweet_input"
        )
        
        # Add analysis options
        col_a, col_b = st.columns(2)
        with col_a:
            analyze_textblob = st.checkbox("Use TextBlob", value=True)
        with col_b:
            analyze_vader = st.checkbox("Use VADER", value=True)
        
        if st.button("🚀 Analyze & Store", type="primary", use_container_width=True):
            with st.spinner("Analyzing with advanced metrics..."):
                time.sleep(0.5)
                
                # Word count using numpy
                word_count = len(str(tweet).split())
                
                # TextBlob analysis
                if analyze_textblob:
                    tb_analysis = TextBlob(tweet)
                    tb_score = float(tb_analysis.sentiment.polarity)
                    tb_sentiment = 'positive' if tb_score > 0.1 else 'negative' if tb_score < -0.1 else 'neutral'
                    confidence = float(tb_analysis.sentiment.subjectivity)
                else:
                    tb_score = 0.0
                    tb_sentiment = 'N/A'
                    confidence = 0.0
                
                # VADER analysis if available
                if analyze_vader and nltk_available:
                    vader_scores = sia.polarity_scores(tweet)
                    vader_score = float(vader_scores['compound'])
                    vader_sentiment = 'positive' if vader_score >= 0.05 else 'negative' if vader_score <= -0.05 else 'neutral'
                else:
                    vader_score = 0.0
                    vader_sentiment = 'N/A'
                
                # Create new entry as pandas Series
                new_id = len(st.session_state.analysis_history) + 1
                new_entry = pd.Series({
                    'id': new_id,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'tweet': tweet,
                    'tweet_short': (tweet[:40] + '...') if len(tweet) > 40 else tweet,
                    'textblob_score': tb_score,
                    'textblob_sentiment': tb_sentiment,
                    'vader_score': vader_score,
                    'vader_sentiment': vader_sentiment,
                    'confidence': confidence,
                    'word_count': word_count
                })
                
                # Append to DataFrame using pandas
                st.session_state.analysis_history = pd.concat(
                    [st.session_state.analysis_history, new_entry.to_frame().T], 
                    ignore_index=True
                )
                
                st.success(f"✅ Analysis #{new_id} stored in pandas DataFrame!")
                
                # Show immediate results
                st.markdown("---")
                st.markdown("### 📊 Immediate Results")
                
                result_cols = st.columns(3)
                with result_cols[0]:
                    st.metric("TextBlob Score", f"{tb_score:.3f}", f"Sentiment: {tb_sentiment}")
                with result_cols[1]:
                    if analyze_vader and nltk_available:
                        st.metric("VADER Score", f"{vader_score:.3f}", f"Sentiment: {vader_sentiment}")
                with result_cols[2]:
                    st.metric("Word Count", word_count, f"Confidence: {confidence:.2f}")
    
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
            if st.button(f"{emoji} {label}", key=f"ex_{label}", use_container_width=True):
                st.session_state.example_text = text
                st.rerun()

elif view_mode == "Data Analytics" and not st.session_state.analysis_history.empty:
    # ============================================
    # ADVANCED PANDAS/NUMPY ANALYTICS VIEW
    # ============================================
    st.markdown("## 📊 Advanced Data Analytics")
    
    # Convert to DataFrame for analysis
    df = st.session_state.analysis_history.copy()
    
    # Calculate advanced metrics using numpy
    metrics = calculate_advanced_metrics(df)
    
    # Display key metrics
    st.markdown("### 📈 Statistical Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Analyses", len(df))
        st.metric("Mean Score", f"{metrics.get('mean_score', 0):.3f}")
    
    with col2:
        st.metric("Score Range", f"{metrics.get('score_range', 0):.3f}")
        st.metric("Std Deviation", f"{metrics.get('std_score', 0):.3f}")
    
    with col3:
        st.metric("Positive %", f"{metrics.get('positive_ratio', 0):.1f}%")
        st.metric("Negative %", f"{metrics.get('negative_ratio', 0):.1f}%")
    
    with col4:
        st.metric("Neutral %", f"{metrics.get('neutral_ratio', 0):.1f}%")
        st.metric("Median Score", f"{metrics.get('median_score', 0):.3f}")
    
    # Data Distribution
    st.markdown("### 📊 Data Distribution")
    
    # Create distribution plot using matplotlib
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    
    if 'textblob_score' in df.columns:
        # Create histogram with numpy bins
        scores = df['textblob_score'].astype(float).values
        hist, bins = np.histogram(scores, bins=20, range=(-1, 1))
        
        # Plot histogram
        ax1.hist(scores, bins=20, alpha=0.7, color='#3498db', edgecolor='black')
        ax1.axvline(x=metrics.get('mean_score', 0), color='red', linestyle='--', 
                   label=f'Mean: {metrics.get("mean_score", 0):.3f}')
        ax1.axvline(x=metrics.get('median_score', 0), color='green', linestyle='--',
                   label=f'Median: {metrics.get("median_score", 0):.3f}')
        
        ax1.set_title('Sentiment Score Distribution', fontweight='bold')
        ax1.set_xlabel('Sentiment Score')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        st.pyplot(fig1)
    
    # Correlation Matrix
    st.markdown("### 🔗 Correlation Analysis")
    corr_matrix = create_correlation_matrix(df)
    
    if corr_matrix is not None:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        
        # Create heatmap using matplotlib
        im = ax2.imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add annotations
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                text = ax2.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        # Set labels
        ax2.set_xticks(np.arange(len(corr_matrix.columns)))
        ax2.set_yticks(np.arange(len(corr_matrix.columns)))
        ax2.set_xticklabels(corr_matrix.columns, rotation=45)
        ax2.set_yticklabels(corr_matrix.columns)
        ax2.set_title('Correlation Matrix', fontweight='bold')
        
        # Add colorbar
        plt.colorbar(im, ax=ax2)
        
        st.pyplot(fig2)
    
    # Data Export Options
    st.markdown("### 💾 Export Data")
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        # CSV Export
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="sentiment_analytics.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_exp2:
        # JSON Export
        json_data = df.to_json(orient='records')
        st.download_button(
            label="📥 Download as JSON",
            data=json_data,
            file_name="sentiment_analytics.json",
            mime="application/json",
            use_container_width=True
        )

elif view_mode == "History Dashboard" and not st.session_state.analysis_history.empty:
    # History visualization tab
    st.markdown("## 📈 Analysis History Visualizations")
    
    # Data metrics using pandas
    df = st.session_state.analysis_history.copy()
    
    st.markdown("### 📊 Dashboard Metrics")
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        total = len(df)
        st.metric("Total Analyses", total)
    
    with metric_cols[1]:
        if 'textblob_sentiment' in df.columns:
            positive = len(df[df['textblob_sentiment'] == 'positive'])
            st.metric("Positive", positive, f"{positive/total*100:.1f}%" if total > 0 else "0%")
    
    with metric_cols[2]:
        if 'textblob_score' in df.columns:
            avg_tb = df['textblob_score'].mean()
            st.metric("Avg. TextBlob", f"{avg_tb:.3f}")
    
    with metric_cols[3]:
        if nltk_available and 'vader_score' in df.columns:
            avg_vader = df['vader_score'].mean()
            st.metric("Avg. VADER", f"{avg_vader:.3f}")
    
    # Create visualizations
    if chart_type in ["Bar Chart", "All Charts"]:
        st.markdown("### 📊 Sentiment Distribution")
        
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        if 'textblob_sentiment' in df.columns:
            sentiment_counts = df['textblob_sentiment'].value_counts()
            
            colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#3498db'}
            bar_colors = [colors.get(s, '#95a5a6') for s in sentiment_counts.index]
            
            bars = ax1.bar(sentiment_counts.index, sentiment_counts.values, 
                          color=bar_colors, edgecolor='black')
            ax1.set_title('Sentiment Distribution (TextBlob)', fontweight='bold')
            ax1.set_xlabel('Sentiment')
            ax1.set_ylabel('Count')
            ax1.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, count in zip(bars, sentiment_counts.values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                        str(count), ha='center', va='bottom', fontweight='bold')
            
            st.pyplot(fig1)
    
    # Data table with pandas styling
    with st.expander("📋 View Raw Data with pandas"):
        st.dataframe(df.style.background_gradient(subset=['textblob_score', 'vader_score'], 
                                                 cmap='RdYlGn'),
                    use_container_width=True)

elif view_mode == "Comparison View" and nltk_available and not st.session_state.analysis_history.empty:
    # Method comparison tab
    st.markdown("## 🤝 Method Comparison Analysis")
    
    df = st.session_state.analysis_history.copy()
    
    if 'vader_score' in df.columns:
        # Create comparison visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Scatter plot comparison
        ax1.scatter(df['textblob_score'], df['vader_score'], 
                   alpha=0.6, color='#9b59b6', edgecolors='black')
        ax1.set_title('TextBlob vs VADER Score Comparison', fontweight='bold')
        ax1.set_xlabel('TextBlob Score')
        ax1.set_ylabel('VADER Score')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # Agreement analysis using pandas
        agreement = (df['textblob_sentiment'] == df['vader_sentiment']).mean() * 100
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
        
        # Show disagreements using pandas filtering
        disagreements = df[df['textblob_sentiment'] != df['vader_sentiment']]
        if not disagreements.empty:
            st.markdown("### 🔍 Sample Disagreements")
            st.dataframe(disagreements[['tweet_short', 'textblob_sentiment', 'vader_sentiment']].head(), 
                        use_container_width=True)

else:
    # Empty state
    st.info("📋 No analysis history yet. Switch to 'Live Analysis' to start analyzing tweets!")

# ============================================
# DEPLOYMENT PROGRESS FOOTER
# ============================================
st.markdown("---")
st.markdown("### 🚀 Deployment Progress")

progress_cols = st.columns(5)

with progress_cols[0]:
    st.success("✅ Phase 1\nBasic App")
with progress_cols[1]:
    st.success("✅ Phase 2\nTextBlob")
with progress_cols[2]:
    st.success("✅ Phase 3\nNLTK/VADER")
with progress_cols[3]:
    st.info("🔄 Phase 4\npandas/numpy")
with progress_cols[4]:
    st.info("Next: Phase 5\nVisualizations")

# Add numpy operations demo in sidebar
with st.sidebar.expander("🧮 numpy Operations Demo"):
    if not st.session_state.analysis_history.empty:
        df = st.session_state.analysis_history.copy()
        if 'textblob_score' in df.columns:
            scores = df['textblob_score'].astype(float).values
            
            st.write("**numpy Operations:**")
            st.write(f"Array shape: {scores.shape}")
            st.write(f"Sorted scores: {np.sort(scores)[:5]}...")
            st.write(f"Unique values: {np.unique(scores).size}")
            
            # Show some numpy calculations
            if len(scores) > 1:
                st.write(f"Variance: {np.var(scores):.4f}")
                st.write(f"25th percentile: {np.percentile(scores, 25):.3f}")
                st.write(f"75th percentile: {np.percentile(scores, 75):.3f}")

# Refresh dashboard
if st.sidebar.button("🔄 Update Analytics", use_container_width=True):
    st.rerun()
