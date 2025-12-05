# RAG QA Bot using LangChain, Chroma, and Gradio

import os
from langchain_community.chat_models.openai import ChatOpenAI
from dotenv import find_dotenv, load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from huggingface_hub import HfFolder
import gradio as gr

# Suppress warnings generated
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

load_dotenv(find_dotenv())
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

llm_model = "qwen/qwen3-235b-a22b:free"

# Use Qwen as the single primary model for this app (no failover).
chat = ChatOpenAI(
    model_name=llm_model,
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.0
)

# Load the Document
def document_loader(file_path):
    loader = PyPDFLoader(file_path)
    loaded_documents = loader.load()
    return loaded_documents

# Define the text splitter
def text_splitter(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

# Define the Embedding Model
def get_embedding_model():
    hf_token = HfFolder.get_token()
    from langchain_community.embeddings import HuggingFaceEmbeddings
    # Some versions of the community HuggingFaceEmbeddings expect the
    # huggingface hub token to be provided via the environment variable
    # `HUGGINGFACEHUB_API_TOKEN` rather than as a constructor kwarg. Passing
    # the token directly can cause Pydantic validation errors like
    # "extra fields not permitted". Set the env var here if we have a token.
    if hf_token:
        os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", hf_token)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    return embedding_model

# Define the Vector Store
def get_vector_store(chunks, embedding_model):
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name="rag_qa_bot_collection"
    )
    return vector_store

# Define the Retriever
def get_retriever(vector_store):
    # `vector_store` parameter is actually a file path passed from the Gradio UI.
    # Rename it to `file_path` for clarity and use that file to build the retriever.
    file_path = vector_store
    documents = document_loader(file_path)
    chunks = text_splitter(documents)
    embedding_model = get_embedding_model()
    vector_store = get_vector_store(chunks, embedding_model)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    return retriever

# QA Chain
def get_qa_chain(file, query):
    # Build a retriever from the uploaded file path
    retriever_obj = get_retriever(file)

    qa_chain = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=True
    )

    # Single-model flow: call the RetrievalQA chain using Qwen and return
    # the exact model result in the `Answer` field and the raw chain
    # output in the `Full LLM Output` field.
    try:
        raw_result = qa_chain({"query": query})
    except Exception as e:
        err = f"Model error: {e}"
        return err, err

    # If the chain returns a dict with a `result` key, return that exact
    # value verbatim as the concise Answer. Otherwise, if it's a string,
    # use it; else fall back to repr for the short answer. The full
    # output is the raw representation of the chain return value.
    try:
        if isinstance(raw_result, dict) and "result" in raw_result:
            short_answer = raw_result["result"]
        elif isinstance(raw_result, str):
            short_answer = raw_result
        else:
            short_answer = repr(raw_result)
    except Exception:
        short_answer = repr(raw_result)

    try:
        full_raw = raw_result if isinstance(raw_result, str) else repr(raw_result)
    except Exception:
        full_raw = repr(raw_result)

    return short_answer, full_raw

# Gradio Interface
rag_application = gr.Interface(
    fn=get_qa_chain,
    allow_flagging="never",
    inputs=[
        gr.File(label="Upload PDF Document", file_count="single", file_types=[".pdf"], type="filepath"),
        gr.Textbox(label="Enter your question here", lines=4, placeholder="Type your question here...")
    ],
    outputs=[
        gr.Textbox(label="Answer", lines=6),
        gr.Textbox(label="Full LLM Output", lines=20),
    ],
    title="RAG QA Bot with Gradio",
    description="Upload a PDF document and ask questions about its content."
)

# Launch the app
rag_application.launch(server_name="127.0.0.1", server_port=7870)