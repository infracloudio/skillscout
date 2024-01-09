import boto3
import os

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


BEDROCK_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v1"
BEDROCK_LLM_MODEL_ID = "amazon.titan-text-express-v1"
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")


def get_bedrock_runtime_client():
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_REGION,
    )


def get_bedrock_embeddings():
    return BedrockEmbeddings(
        model_id=BEDROCK_EMBEDDING_MODEL_ID,
        client=get_bedrock_runtime_client(),
    )


def get_bedrock_llm():
    return Bedrock(
        model_id=BEDROCK_LLM_MODEL_ID,
        model_kwargs={
            "maxTokenCount": 512,
            "stopSequences": [],
            "temperature": 0,
            "topP": 0.9,
        },
        client=get_bedrock_runtime_client(),
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )
