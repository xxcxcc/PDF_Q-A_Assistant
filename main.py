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

uploaded_file = st.file_uploader("选择 PDF 文件", type="pdf")
question = st.text_input("请输入 PDF 相关问题", disabled=not uploaded_file)
st.button("提交")

if uploaded_file and question and openai_api_key:
    with st.spinner("正在处理..."):
        response = qa_agent(uploaded_file, question, st.session_state["memory"],
                            openai_api_key)
        st.write("### 答案")
        st.write(response["answer"])
        st.session_state["chat_history"] = response["chat_history"]

if "chat_history" in st.session_state:
    with st.expander("查看历史记录"):
        for i, message in enumerate(st.session_state["chat_history"]):
            if i % 2 == 0:
                st.write(f"### 问题：{message.content}")
            else:
                st.write(f"### 答案：{message.content}")
            if i != len(st.session_state["chat_history"]) - 1:
                st.divider()
