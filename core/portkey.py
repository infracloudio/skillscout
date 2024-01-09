import os

from portkey_ai.llms.langchain import ChatPortkey

PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
PORTKEY_VIRTUAL_KEY = os.getenv("PORTKEY_VIRTUAL_KEY")

def get_portkey_llm():
    return ChatPortkey(
        api_key=PORTKEY_API_KEY,
        virtual_key=PORTKEY_VIRTUAL_KEY,
        model="gpt-3.5-turbo-0125",
    )
