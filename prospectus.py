import os
import streamlit as st

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm():
    llm = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name="llama3-8b-8192",
        temperature=0.5,
        max_tokens=512
    )
    return llm


# --- Add this at the top of your file, after imports ---
st.markdown("""
    <style>
    body, .main {
        background-color: #181a1b;
        font-family: 'Segoe UI', 'Arial', sans-serif;
    }
    .chat-container {
        max-width: 700px;
        margin: 0 auto;
        padding-bottom: 80px;
    }
    .chat-bubble {
        padding: 16px 20px;
        border-radius: 18px;
        margin-bottom: 12px;
        max-width: 80%;
        font-size: 1.1em;
        line-height: 1.6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        word-break: break-word;
        border: none;
    }
    .user-bubble {
        background: linear-gradient(90deg, #23272f 80%, #23272f 100%);
        color: #fff;
        margin-left: auto;
        margin-right: 0;
        border-bottom-right-radius: 6px;
        text-align: right;
    }
    .assistant-bubble {
        background: #f4f6fa;
        color: #222;
        margin-right: auto;
        margin-left: 0;
        border-bottom-left-radius: 6px;
        text-align: left;
    }
    .header-title {
        font-size: 2.2em;
        font-weight: 700;
        margin-bottom: 0.2em;
        color: #fff;
        text-align: center;
        letter-spacing: -1px;
    }
    .header-subtitle {
        font-size: 1.1em;
        color: #bdbdbd;
        text-align: center;
        margin-bottom: 2em;
    }
    .source-card {
        background: #e9ecef;
        color: #222;
        border-radius: 8px;
        padding: 10px;
        margin-top: 8px;
        font-size: 0.97em;
    }
    /* Hide Streamlit's default hamburger and footer */
    #MainMenu, footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)


def main():
    st.markdown('<div class="header-title">Ask Chatbot!</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Your AI-powered NED prospectus assistant. Ask any NED University related question!</div>', unsafe_allow_html=True)

    # Center chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Input at the bottom
    prompt = st.chat_input("Type your question here...")

    if prompt:
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                You are a helpful NED University prospectus assistant.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context
                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """
        
        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain=RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response=qa_chain.invoke({'query':prompt})

            result=response["result"]
            source_documents=response["source_documents"]

            # Format the answer (without sources)
            result_to_show = result.strip()

            st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

    # Render chat history in order, alternating bubbles
    for message in st.session_state.messages:
        role = message['role']
        content = message['content']
        bubble_class = "user-bubble" if role == "user" else "assistant-bubble"
        st.markdown(f'<div class="chat-bubble {bubble_class}">{content}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # Close chat-container

    # Sidebar for info
    st.sidebar.title("About")
    ##st.sidebar.info("This chatbot is for informational purposes only and does not provide medical advice. Always consult a healthcare professional.")

if __name__ == "__main__":
    main()
    