import os
import gradio
import click

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from core import get_llm, get_embeddings


# from langchain.globals import set_verbose, set_debug
# set_verbose(True)
# set_debug(True)

DEFAULT_RESUME_DIR = "./data"
DEFAULT_VECTOR_STORE = "vector_store"


def load_text_docs(resume_summary_path):
    loader = DirectoryLoader(resume_summary_path, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    return documents


def split_docs_with_merge(documents):
    source = documents[0].metadata["source"]
    text = "\n\n".join([d.page_content for d in documents])
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        # Set a really small chunk size, just to show.
        chunk_size=1000,
        chunk_overlap=100,
    )
    splited_text = text_splitter.split_text(text)
    splited_docs = []
    for i, t in enumerate(splited_text):
        splited_docs.append(
            Document(page_content=t, metadata={"chunk": i, "source": source})
        )
    return splited_docs


def split_docs(documents):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        # Set a really small chunk size, just to show.
        chunk_size=1000,
        chunk_overlap=100,
    )
    docs = text_splitter.split_documents(documents)
    return docs


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [
                f"Document {i+1} {d.metadata}:\n\n" + d.page_content
                for i, d in enumerate(docs)
            ]
        )
    )


def format_docs(docs):
    print("-" * 30)
    print("Total docs", len(docs))
    for d in docs:
        print(d.metadata, len(d.page_content))
    print("-" * 30)
    return "\n\n".join([d.page_content for d in docs])


def load_store(vector_store, embeddings):
    return FAISS.load_local(vector_store, embeddings, allow_dangerous_deserialization=True)


def get_chain(vectore_store, llm):
    prompt = """
Context has resume data of candidates. Answer the question only based on the following context. If you don't know the answer, just say that you don't know. Be concise.

Context: {context}
Question: {question}
Answer:
    """
    contextualize_q_chain = lambda input: (
        contextualized_question_chain(llm)
        if input["chat_history"]
        else input["question"]
    )
    prompt = ChatPromptTemplate.from_template(prompt)
    retriever = vectore_store.as_retriever(k=3)
    chain = (
        RunnablePassthrough.assign(
            context=contextualize_q_chain | retriever | format_docs
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def messages_to_chat(messages: list[str]):
    human = True
    history = []
    for m in messages:
        # print(f"Message history {m}")
        if human:
            history.append(f"Question: {m}\n")
            human = False
        else:
            history.append(f"Answer: {m}\n")
            human = True
    return "".join(history)


def contextualized_question_chain(llm):
    prompt = """
Chat History:
{chat_history}

Question: {question}

Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
"""
    context_q_prompt = ChatPromptTemplate.from_template(prompt)
    chain = (
        {
            "chat_history": lambda input: messages_to_chat(input["chat_history"]),
            "question": lambda input: input["question"],
        }
        | context_q_prompt
        | llm
        | StrOutputParser()
    )
    return chain


def summarize_pdf(pdf_path, llm):
    pdf_without_extension = os.path.splitext(pdf_path)[0]
    summary_path = f"{pdf_without_extension}.txt"
    # if summary already exists, skip
    if os.path.exists(summary_path):
        print(f"Summary already exists for {pdf_path}")
        return
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splited_docs = split_docs_with_merge(documents)
    question_prompt = ChatPromptTemplate.from_messages(
        messages=[
            ("system", "Hello, I am Skillscout. I can help you summarize resumes."),
            (
                "user",
                """
Please extract the following data from the resume text provided in triple quotes below and provide them only in format specified:
Name of the candidate:
Professional Summary:
Total years of experience:
Skills (comma separated):
Companies worked for (comma separated):
Education:
Certifications (comma separated):

When extracting the candidate's total years of experience if available in the resume text or calculate the total years of experience based on the dates provided in the resume text. If you are calculating then assume current month and year as Jan 2024.

Resume Text:
```
{text}
```""",
            ),
        ],
    )

    refine_prompt = ChatPromptTemplate.from_messages(
        messages=[
            ("system", "Hello, I am Skillscout. I can help you summarize resumes."),
            (
                "user",
                """
Please extract the following data from the existing and additional resume text provided in triple quotes below and provide them only in format specified:
Name of the candidate:
Professional Summary:
Total years of experience:
Education:
Skills (comma separated):
Companies worked for (comma separated):
Certifications (comma separated):

When extracting the candidate's total years of experience if available in the resume text or calculate the total years of experience based on the dates provided in the resume text. If you are calculating then assume current month and year as Jan 2024.

Existing Resume Text:
```
{existing_answer}
```

Additional Resume Text:
```
{text}
```
""",
            ),
        ]
    )

    chain = load_summarize_chain(
        llm,
        chain_type="refine",
        question_prompt=question_prompt,
        refine_prompt=refine_prompt,
        verbose=False,
        input_key="input_text",
        output_key="output_text",
    )
    summary = chain.invoke({"input_text": splited_docs}, return_only_outputs=True)
    output = summary["output_text"]
    # Write the output to a file
    # replace pdf extension with txt
    with open(summary_path, "w") as f:
        f.write(output)
    print(f"Summary written to {summary_path} for {pdf_path}")


@click.command(name="summarize")
@click.option("--resume_path", default=DEFAULT_RESUME_DIR, help="Path to resume pdfs")
def summarize(resume_path):
    llm = get_llm()
    for file in os.listdir(resume_path):
        if file.endswith(".pdf"):
            summarize_pdf(f"{resume_path}/{file}", llm)


@click.command(name="vector_store")
@click.option(
    "--resume_summary_path", default=DEFAULT_RESUME_DIR, help="Path to resume summaries"
)
@click.option(
    "--vector_store", default=DEFAULT_VECTOR_STORE, help="Path to vector store"
)
def gen_vector_store(resume_summary_path, vector_store):
    documents = load_text_docs(resume_summary_path)
    docs = split_docs(documents)
    print("Found {} documents".format(len(docs)))
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(
        documents=docs,
        embedding=embeddings,
    )
    vectorstore.save_local(vector_store)


@click.command(name="app")
@click.option(
    "--vector_store", default=DEFAULT_VECTOR_STORE, help="Path to vector store"
)
def app(vector_store):
    llm = get_llm()
    embeddings = get_embeddings()
    store = load_store(vector_store, embeddings)
    chain = get_chain(store, llm)

    gradio.ChatInterface(
        fn=lambda message, history: (
            chain.invoke({"question": message, "chat_history": history[0]})
            if len(history) > 0
            else chain.invoke({"question": message, "chat_history": []})
        ),
        title="Skillscout - Resume Chatbot",
        description="Ask questions about resumes",
    ).queue().launch(share=False)


@click.group()
def cli():
    pass


cli.add_command(summarize)
cli.add_command(gen_vector_store)
cli.add_command(app)

if __name__ == "__main__":
    cli()
