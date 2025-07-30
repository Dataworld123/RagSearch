import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


#import api key

from config import PINECONE_API_KEY ,PINECONE_ENVIRONMENT,PINE_CONE_INDEX_NAME

# Set Environemnt variable for pinecone

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Initialize pinecone
pc = Pinecone (api_key= PINECONE_API_KEY)

# define huggingface embedding model

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLm-L6-v2")

# define pinecone index name
INDEX_NAME =PINE_CONE_INDEX_NAME


# define retriever


def get_retriever():
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"creating new pinecone index:{INDEX_NAME}...")
        pc.create_index(
            name =INDEX_NAME,
            dimension =384,
            metric ="cosine",
            spec=ServerlessSpec(cloud="aws",region="us-east-1")


        )
        print("index create successfully")

    vectorstore =PineconeVectorStore(index_name=INDEX_NAME,embedding=embeddings)
    return vectorstore.as_retriever()


def ensure_index_exists():
    """Ensure the Pinecone index exists, create if it doesn't"""
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new pinecone index: {INDEX_NAME}...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print("Index created successfully")


# Add document to the vectorestore
def add_document_to_vectorstore(text_content: str):
    if not text_content:
        raise ValueError("Document content can not be empty")
    
    # Ensure index exists before adding documents
    ensure_index_exists()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )

    # Create Langchain Document from raw text
    documents = text_splitter.create_documents([text_content])
    print(f"Splitting document into {len(documents)} chunks for indexing...")

    # Get vector store instance
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    # Add documents to the vectorstore
    vectorstore.add_documents(documents)
    print(f"Successfully added {len(documents)} chunks to pinecone index {INDEX_NAME}.")




