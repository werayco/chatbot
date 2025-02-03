import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import warnings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

warnings.filterwarnings("ignore")
groq_api_key = st.secrets["API_KEY"]

hf_token = st.secrets["HF_TOKEN"]

pdf_doc = PyPDFLoader("./Pdf_for_rag_chatbot.pdf").load()
splitter = RecursiveCharacterTextSplitter(chunk_overlap=20, chunk_size=400)
docs = splitter.split_documents(pdf_doc)

# savipersistingng the vector database [in our cwd] by 
persist_directory = "./chroma_db"
vector_db = Chroma.from_documents(docs, HuggingFaceEmbeddings(), persist_directory=persist_directory)
db = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chat_bot = ChatGroq(model="gemma2-9b-it", api_key=groq_api_key, temperature=0.2)

template = """You are an AI language model Assistant. Your task is to generate different versions of the given
user question to retrieve relevant documents from a vector database. Provide these alternative questions separated by newlines.
Original question: {question}"""

query_prompt = PromptTemplate.from_template(template)
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=db, llm=chat_bot, prompt=query_prompt
)

template = """Answer the question based ONLY on the following context: {context}
Question: {question}.
Also, if asked who you are, you are a chatbot created by Helpwritingresumes to help users
Any question asked is about Helpwritingresumes """
ptmp = ChatPromptTemplate.from_template(template)
chain = (
    {"context": retriever_from_llm, "question": RunnablePassthrough()}
    | ptmp
    | chat_bot
    | StrOutputParser()
)

st.title("helpwritingresume Chatbot")
st.write("This is a Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type your message...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    response = chain.invoke(user_input)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
