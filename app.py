import streamlit as st

st.set_page_config(page_title="Tweet Analyzer", page_icon="📊")
st.title("📊 Simple Tweet Analyzer")
st.write("This is a test deployment to verify Streamlit Cloud works.")

user_input = st.text_input("Enter a tweet:", "Hello, this app is deployed!")
if user_input:
    st.success(f"You entered: {user_input}")
    if "good" in user_input.lower() or "great" in user_input.lower():
        st.balloons()
        st.success("😊 Positive sentiment detected!")
