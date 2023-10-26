import openai
import streamlit as st
import pinecone
import plotly.express as px
import os
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA

P_API_KEY =st.secrets["pincone_key"]
API_KEY=st.secrets["openAI_key"]
st.set_page_config(layout="centered")

with st.container():
    video_html = """
    <style>
    .video-container {
    width: 100%; 
    height: auto;
    }

    video {
    width: 100%; 
    height: auto;
    }

    .content {
    background: rgba(0, 0, 0, 0.5);
    color: #f1f1f1;
    width: 100%;
    padding: 20px;
    }
    </style>    
    <div class="video-container">
    <video autoplay muted loop id="myVideo">
        <source src="https://static.streamlit.io/examples/star.mp4">
        Your browser does not support HTML5 video.
    </video>
    </div>
    """  
    st.markdown(video_html, unsafe_allow_html=True) 

loader = WebBaseLoader("https://medium.com/swlh/an-ultimate-guide-to-creating-a-startup-3b310f41d7e7")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 1000,
    chunk_overlap  = 100,
    length_function = len,
    add_start_index = True,
)
texts = text_splitter.split_documents(data)

embeddings = OpenAIEmbeddings(openai_api_key =API_KEY) # set openai_api_key = 'your_openai_api_key' # type: ignore
pinecone.init(api_key=P_API_KEY, environment="gcp-starter")
index_name = pinecone.Index('index-1')

#llm = ChatOpenAI(model_name='gpt-3.5-turbo-0301', temperature=0,openai_api_key =API_KEY )
#vectordb = Pinecone.from_documents(texts, embeddings, index_name='index-1')
#retriever = vectordb.as_retriever()
#memory = ConversationBufferMemory(memory_key="chat_history", return_messages= True)
#chain = ConversationalRetrievalChain.from_llm(llm, retriever= retriever, memory= memory)

def semantic_search(prompt):
  MODEL = "text-embedding-ada-002"
  messages = []

  index = pinecone.Index('index-1')

  search_response = index.query(
  top_k=5,
    vector=tuple(openai.Embedding.create(
    input=[
        prompt
    ], engine=MODEL
  )["data"][0]["embedding"]),
    include_metadata=True,
  )['matches']

  for response in search_response:
    messages.append(response['metadata']['description'])

  return messages

with st.sidebar:
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    openai.api_key = API_KEY
    st.session_state.messages.append({"role": 'system', 'content': 'the following is your memory: '+str(semantic_search(prompt))})
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    msg = response.choices[0].message
    print(msg)
    st.session_state.messages.append(msg)
    st.chat_message("assistant").write(msg.content)
    






