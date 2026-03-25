import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()

# Initialize model
model = ChatMistralAI(
    model="mistral-small",
    api_key=os.getenv("MISTRAL_API_KEY")
)

# Page config
st.set_page_config(page_title="AI Chatbot", page_icon="💬", layout="centered")

st.title("💬AI Chatbot")

# ------------------------------
# Reset chat without experimental_rerun
# ------------------------------
if "reset_flag" not in st.session_state:
    st.session_state.reset_flag = False

with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🧹 Reset Chat"):
        st.session_state.messages = [SystemMessage(content="You are a helpful AI assistant")]
        st.session_state.chat_display = []
        st.session_state.reset_flag = True

# Initialize session
if "messages" not in st.session_state or st.session_state.reset_flag:
    st.session_state.messages = [SystemMessage(content="You are a helpful AI assistant")]
    st.session_state.chat_display = []
    st.session_state.reset_flag = False

if "chat_display" not in st.session_state:
    st.session_state.chat_display = []

# ------------------------------
# Display chat messages
# ------------------------------
for chat in st.session_state.chat_display:
    role = chat["role"]
    content = chat["content"]
    st.chat_message(role).markdown(content)

# ------------------------------
# User input
# ------------------------------
prompt = st.chat_input("Type your message...")

if prompt:
    # Append user message
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.session_state.chat_display.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # Typing spinner for UX
    with st.spinner("🤖 Thinking..."):
        time.sleep(0.5)  # short delay for effect

    # Get AI response
    try:
        response = model.invoke(st.session_state.messages)
        reply = response.content
    except Exception as e:
        reply = f"Error: {str(e)}"

    # Append AI message
    st.session_state.messages.append(AIMessage(content=reply))
    st.session_state.chat_display.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").markdown(reply)
