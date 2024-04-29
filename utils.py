from langchain.chains import ConversationalRetrivalChain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


def qa_agent(openai_api_key, memory, uploaded_file, question):
    """
    使用OpenAI API和PDF内容来回答问题的智能代理。

    参数:
    - openai_api_key (str): 访问OpenAI API所需的密钥。
    - memory (list): 前一次对话的记忆，用于上下文连续性。
    - uploaded_file (IO): 包含要处理的PDF内容的文件对象。
    - question (str): 需要回答的问题。

    返回:
    - response (dict): 包含问题回答的字典。
    """

    # 初始化OpenAI聊天模型
    model = ChatOpenAI(model_name="gpt-3.5-turbo",
                       openai_api_key=openai_api_key,
                       base_url="https://api.aigc369.com/v1")

    # 从上传的文件中读取内容
    file_content = uploaded_file.read()

    # 将文件内容临时保存到本地
    temp_file_path = "temp.pdf"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(file_content)

    # 加载PDF内容
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    # 将PDF文本分割成小块
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=50,
                                                   separators=["\n\n", "\n", "。", "！", "？", "；", "，", "：", ""])
    texts = text_splitter.split_documents(docs)

    # 使用嵌入模型对文本进行嵌入，以便于检索
    embeddings_model = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings_model)

    # 设置检索器和对话链
    retriever = db.as_retriever()
    qa = ConversationalRetrivalChain.from_llm(llm=model, retriever=retriever, memory=memory)

    # 使用设定好的模型和检索器来回答问题
    response = qa.invoke({"chat_history": memory, "question": question})

    return response
