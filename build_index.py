
import os

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


DATA_PATH = "data"
DB_FAISS_PATH = "vectorstore/faiss_index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents():

    documents = []

    # Load text files

    text_loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents.extend(text_loader.load())

    # Load Markdown files

    markdown_loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents.extend(markdown_loader.load())

    # Load PDF files

    pdf_loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )

    documents.extend(pdf_loader.load())

    return documents


def create_vectorstore():

    print("Loading portfolio documents...")

    documents = load_documents()

    if not documents:
        raise ValueError(
            "No documents found inside the data folder."
        )

    print(f"Loaded {len(documents)} document pages/files.")

    # Split documents into smaller chunks

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    # Generate embeddings

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    # Create FAISS index

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    # Save index

    os.makedirs(
        os.path.dirname(DB_FAISS_PATH),
        exist_ok=True
    )

    vectorstore.save_local(DB_FAISS_PATH)

    print("FAISS index created successfully!")


if __name__ == "__main__":
    create_vectorstore()
