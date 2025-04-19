import os
from dotenv import load_dotenv
from huggingface_hub import login

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Step 1: Load environment variables and login
load_dotenv()  # Loads from .env file

HF_TOKEN = os.getenv("HF_TOKEN")  # Make sure .env contains HF_TOKEN=hf_...
if not HF_TOKEN:
    raise ValueError("Hugging Face token not found. Make sure HF_TOKEN is set in .env")

login(HF_TOKEN)  # Authenticate with Hugging Face

HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Step 2: Setup the LLM
def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        max_length=512,
        task="text-generation",
        huggingfacehub_api_token=HF_TOKEN
    )
    return llm

# Step 3: Prompt template
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer. 
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# Step 4: Load vector database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 5: Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Step 6: Run a query
user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})

print("\nRESULT:\n", response["result"])
print("\nSOURCE DOCUMENTS:\n", response["source_documents"])
