import os

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_chroma import Chroma


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=API_KEY
)


def create_database(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.create_documents([text])

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    return db


def ask_question(db, question):

    docs = db.similarity_search(question, k=3)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    prompt = f"""
Answer the question using ONLY the information
provided in the context below.

Context:
{context}

Question:
{question}

If the answer is not available in the context,
say:

"I could not find the answer in the uploaded document."
"""

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=API_KEY
    )

    response = llm.invoke(prompt)

    # Gemini 3 can return content as structured blocks, for example:
    # [{"type": "text", "text": "...", "extras": {...}}].
    # Return only the visible text instead of Streamlit rendering the metadata.
    if isinstance(response.content, str):
        return response.content

    text_blocks = [
        block["text"]
        for block in response.content
        if isinstance(block, dict) and block.get("type") == "text"
    ]

    return "\n".join(text_blocks)
