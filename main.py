import streamlit as st
from langchain.memory import ConversationBufferMemory
from utils import qa_agent

st.title("PDF智能问答助手")

with st.sidebar:
    openai_api_key = st.text_input("请输入OpenAI API Key", type="password")
    st.markdown("[若无秘钥，请点此获取](https://api.aigc369.com/register)")

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer")
