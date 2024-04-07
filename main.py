import streamlit as st
from dotenv import load_dotenv
import pdfplumber
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import concurrent.futures
import tempfile
import os
from pathlib import Path


def process_pdf(pdf_path):
    text_content = ""
    table_content = []

    with pdf_path.open("rb") as file:
        pdf_reader = pdfplumber.open(file)
        for page in pdf_reader.pages:
            text = page.extract_text()
            text_content += text + "\n"

            tables = page.extract_tables()
            for table in tables:
                if table is not None:
                    table_content.append(table)

    return text_content, table_content


def extract_text_and_tables(pdf_docs):
    text_content = ""
    table_content = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_pdf, pdf_path) for pdf_path in pdf_docs]
        for future in concurrent.futures.as_completed(futures):
            pdf_text, pdf_tables = future.result()
            text_content += pdf_text
            table_content.extend(pdf_tables)

    return text_content, table_content


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    if st.session_state.conversation is not None:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        st.error("Conversation not initialized. Please upload PDFs and process them first.")


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        with st.spinner("Processing user question..."):
            handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing PDFs..."):
                with tempfile.TemporaryDirectory() as temp_dir:
                    if pdf_docs is not None:
                        pdf_paths = [Path(temp_dir) / f.name for f in pdf_docs]
                        for pdf, pdf_path in zip(pdf_docs, pdf_paths):
                            with open(pdf_path, "wb") as f:
                                f.write(pdf.getbuffer())

                        text_content, table_content = extract_text_and_tables(pdf_paths)

                        # Convert table content to text
                        table_text_content = ""
                        for table in table_content:
                            if table is not None:
                                for row in table:
                                    if row is not None:
                                        # Ensure each element in the row is not None
                                        filtered_row = [item if item is not None else "" for item in row]
                                        table_text_content += ' '.join(filtered_row) + '\n'

                        # Combine text and table text
                        combined_text = text_content + table_text_content

                        # Get the text chunks
                        text_chunks = get_text_chunks(combined_text)
                        print(text_chunks)

                        # Create vector store
                        vectorstore = get_vectorstore(text_chunks)

                        # Create conversation chain
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                    else:
                        st.error("Please upload PDF files before processing.")


if __name__ == '__main__':
    main()