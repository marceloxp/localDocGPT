import os
from typing import List
from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from chromadb.config import Settings
from pathlib import Path

ROOT_DIRECTORY = Path().resolve()
# Define the folder for storing database
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/documents"
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/db"
# Define the Chroma settings
CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
)

def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf8")
    elif file_path.endswith(".pdf"):
        loader = PDFMinerLoader(file_path)
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path)
    return loader.load()[0]

def load_documents(source_dir: str) -> List[Document]:
    # Loads all documents from source documents directory
    all_files = os.listdir(source_dir)
    return [load_single_document(f"{source_dir}/{file_path}") for file_path in all_files if file_path[-4:] in ['.txt', '.pdf', '.csv'] ]

device_type = "cuda"
# load the instructorEmbeddings
print(f"Running on: {device_type}")
# Load documents and split in chunks
print(f"Loading documents from {SOURCE_DIRECTORY}")
documents = load_documents(SOURCE_DIRECTORY)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
print(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
print(f"Split into {len(texts)} chunks of text")

# Create embeddings
embeddings = HuggingFaceInstructEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": device_type})

db = Chroma.from_documents(texts, embeddings, persist_directory=PERSIST_DIRECTORY, client_settings=CHROMA_SETTINGS)
db.persist()
db = None