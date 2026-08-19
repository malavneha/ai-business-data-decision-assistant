
import streamlit as st
import pandas as pd
from google import genai
client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)

st.set_page_config(
    page_title="AI Business Data Decision Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Business Data Decision Assistant")
st.write(
    "Ask questions about Citi Bike data and receive "
    "evidence-based business insights."
)

# Load data
df = pd.read_csv("citibike_data.csv")

# Standardize column names
df.columns = df.columns.str.strip().str.lower()

# Match the column names used by the app
df = df.rename(columns={
    "trip duration": "tripduration",
    "user type": "usertype"
})

# Basic analysis
station_demand = (
    df["start station name"]
    .value_counts()
    .head(10)
    .reset_index()
)

station_demand.columns = ["Station", "Number of Trips"]

# Dashboard
st.subheader("📊 Data Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Trips", len(df))

with col2:
    st.metric(
        "Average Trip",
        f"{df['tripduration'].mean():.0f} sec"
    )

with col3:
    st.metric(
        "Top Station",
        station_demand.iloc[0]["Station"]
    )

st.subheader("🚲 Top 10 Starting Stations")

st.bar_chart(
    station_demand.set_index("Station")
)

st.subheader("🤖 Ask the AI Assistant")

question = st.text_input(
    "Ask a business question about the data:"
)

if question:

    summary = f"""
    Total trips analyzed: {len(df)}

    Average trip duration:
    {df["tripduration"].mean():.1f} seconds

    Top 10 starting stations:
    {station_demand.to_string(index=False)}

    User types:
    {df["usertype"].value_counts().to_string()}
    """

    prompt = f"""
    You are an AI Business Data Decision Assistant.

    Use ONLY the following data:

    {summary}

    User question:
    {question}

    Give:
    1. A clear answer.
    2. Evidence from the data.
    3. One business insight.
    4. One practical recommendation.

    Do not invent numbers or facts.
    If the available data cannot answer the question,
    clearly say so.
    """

    try:
        client = genai.Client(
            api_key=st.secrets["GEMINI_API_KEY"]
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        st.subheader("💡 AI Decision Support")
        st.write(response.text)

    except Exception as e:
        st.error(
            "The AI service could not be reached. "
            "Please check the app's Gemini API configuration."
        )
