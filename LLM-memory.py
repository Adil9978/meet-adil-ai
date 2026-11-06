from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Load raw text files
DATA_PATH="data/"
def load_text_files(data):
    loader = DirectoryLoader(data,
                             glob='*.txt',
                             loader_cls=TextLoader,
                             loader_kwargs={"encoding": "utf-8"})
    
    documents=loader.load()
    return documents

documents=load_text_files(data=DATA_PATH)
print("Length of text documents: ", len(documents))


# Step 2: Create Chunks
def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                 chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(extracted_data=documents)
print("Length of Text Chunks: ", len(text_chunks))


def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model=get_embedding_model()
db=FAISS.from_documents(text_chunks, embedding_model)
db.save_local("vectorstore/faiss_index")