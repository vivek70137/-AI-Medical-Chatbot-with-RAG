import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Path to vectorstore
DB_FAISS_PATH = "vectorstore/db_faiss"

# Load the vectorstore with embeddings
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# Custom prompt format
def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# Load LLM from Hugging Face
def load_llm(huggingface_repo_id, HF_TOKEN):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",  # Important for endpoint to understand the task
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.5,
        max_length=512
    )

# Main Streamlit app
def main():
    st.set_page_config(page_title="MediBot - AI Health Assistant", page_icon="🧠", layout="centered")
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🤖 MediBot - Your AI Health Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Ask health-related questions powered by AI</p>", unsafe_allow_html=True)
    st.divider()

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    prompt = st.chat_input("Ask MediBot a health-related question...")

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Prompt template
        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, just say that you don't know—do not make up an answer.
        Only use the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk.
        """

        # Load secrets and components
        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("❌ Failed to load the vector store")

            # Retrieval chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=False,  # ✅ No source docs
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            # Query response
            response = qa_chain.invoke({'query': prompt})
            result = response["result"]

            with st.chat_message("assistant"):
                st.markdown(result)

            st.session_state.messages.append({"role": "assistant", "content": result})

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
