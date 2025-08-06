# app.py

import streamlit as st
import os
import sys
import pandas as pd

# Add project root to sys.path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))


from models.llm import get_chat_model
from utils.predict import predict_disease
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Load symptom options
symptom_options = pd.read_csv("data/symptoms.csv")["Symptom"].tolist()


# 🔁 Function to get Groq response
def get_chat_response(chat_model, messages, system_prompt):
    try:
        formatted_messages = [SystemMessage(content=system_prompt)]

        for msg in messages:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                formatted_messages.append(AIMessage(content=msg["content"]))

        response = chat_model.invoke(formatted_messages)
        return response.content

    except Exception as e:
        return f"Error getting response: {str(e)}"


# 🧭 Page: Instructions
def instructions_page():
    st.title("The Chatbot Blueprint")
    st.markdown("""
    ## 🔧 Installation
    ```bash
    pip install -r requirements.txt
    ```

    ## API Key Setup
    - Get your Groq key here: [https://console.groq.com/keys](https://console.groq.com/keys)

    ## How to Use
    1. Go to the Chat page
    2. Select symptoms
    3. Get disease prediction + ask questions

    _Note: This app is for educational purposes. Not a substitute for real medical advice._
    """)


# 💬 Page: Chat
def chat_page():
    st.title("🩺 AI Health Assistant")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Section 1: Select symptoms
    st.subheader("Select your symptoms")
    user_symptoms = st.multiselect("Symptoms:", symptom_options)

    if st.button("🔍 Predict Disease"):
        if not user_symptoms:
            st.warning("Please select at least one symptom.")
        else:
            with st.spinner("Predicting disease..."):
                result = predict_disease(user_symptoms)

                response = f"""
                Based on your symptoms, you may have **{result['disease']}**.

                **Description:** {result['description']}

                **Medications:** {result['medications']}

                **Precautions:**
                {chr(10).join(['- ' + str(p) for p in result['precautions'] if pd.notna(p)])}

                _Note: This is a prediction. Please consult a real doctor for confirmation._
                """

                st.session_state.messages.append({"role": "assistant", "content": response})

    # Section 2: Chat
    st.subheader("💬 Chat")
    chat_model = get_chat_model()
    system_prompt = "You are a helpful and cautious medical assistant. Always provide safe and polite answers."

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a follow-up question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_chat_response(chat_model, st.session_state.messages, system_prompt)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


# 🔁 Main app entry
def main():
    st.set_page_config(
        page_title="AI Health Assistant",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to:", ["Chat", "Instructions"], index=0)
        if page == "Chat":
            st.divider()
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    if page == "Instructions":
        instructions_page()
    elif page == "Chat":
        chat_page()


if __name__ == "__main__":
    main()

