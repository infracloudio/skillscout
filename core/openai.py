from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

def get_openai_llm():
    return ChatOpenAI(
        model="gpt-3.5-turbo-0125",
        streaming=True,
    )

def get_openai_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-large")