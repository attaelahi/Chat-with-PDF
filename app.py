import streamlit as st
import os
from PyPDF2 import PdfReader
import docx
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub
from langchain.callbacks import get_openai_callback

def main(): 
    load_dotenv()
    st.set_page_config(page_title="Chat With PDF")
    st.header("💬 Chat with PDF")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx'], accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")
    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        files_text = get_files_text(uploaded_files)
        # get text chunks
        text_chunks = get_text_chunks(files_text)
        # create vector store
        vectorstore = get_vectorstore(text_chunks)
        # create conversation chain
        st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)  # for OpenAI
        # st.session_state.conversation = get_conversation_chain(vectorstore)  # for HuggingFace

        st.session_state.processComplete = True

    if st.session_state.processComplete:
        user_question = st.text_input("Ask a question about your files.")
        search_button = st.button("Search")  # Add search button
        if user_question or search_button:  # Process question when the search button is clicked
            handle_user_input(user_question)

def get_files_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        split_tup = os.path.splitext(uploaded_file.name)
        file_extension = split_tup[1]
        if file_extension == ".pdf":
            text += get_pdf_text(uploaded_file)
        elif file_extension == ".docx":
            text += get_docx_text(uploaded_file)
    return text

def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_docx_text(file):
    doc = docx.Document(file)
    all_text = []
    for docpara in doc.paragraphs:
        all_text.append(docpara.text)
    text = ' '.join(all_text)
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,  # Reduced chunk size
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    knowledge_base = FAISS.from_texts(text_chunks, embeddings)
    return knowledge_base

def get_conversation_chain(vectorstore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input(user_question):
    with get_openai_callback() as cb:
        response = st.session_state.conversation({'question': user_question})
    
    st.session_state.chat_history.append((user_question, response['answer']))

    response_container = st.container()
    with response_container:
        for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
            st.markdown(f"**User:** {user_msg}")
            st.markdown(f"**Bot:** {bot_msg}")

        st.write(f"Total Tokens: {cb.total_tokens}, Prompt Tokens: {cb.prompt_tokens}, Completion Tokens: {cb.completion_tokens}, Total Cost (USD): ${cb.total_cost}")

if __name__ == '__main__':
    main()
