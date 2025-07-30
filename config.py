import os
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

#Pinecone

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT","us-east-1")
PINE_CONE_INDEX_NAME= os.getenv("PINECONE_INDEX_NAME","rag-index")

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#Groq_Api_Key = GROQ_API_KEY  # Keep for backward compatibility

# tavily
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# embedding model

EMBED_MODEL = os.getenv("EMBED_MODEL","openai:text-embedding-3-small")

# paths

DOC_SOURCE_DIR=os.getenv("DOC_SOURCE_DIR","data")


