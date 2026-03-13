"""Create a local Chroma FAQ index for the Agentic CX Support Router demo.

This script generates a small dummy FAQ corpus for a fictional sportswear
company, loads it with LangChain document loaders, chunks it, and stores the
embeddings in a local Chroma database using Google Generative AI embeddings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FAQ_PATH = DATA_DIR / "sportswear_faq.txt"
CHROMA_DIR = BASE_DIR / "chroma_db"
EMBEDDING_MODEL = "gemini-embedding-001"


FAQ_TEXT = """# SwiftStride Sportswear Customer Support FAQ

## Returns and refunds
Customers can return unworn apparel and footwear within 30 days of delivery.
Items must include the original tags and packaging.
Refunds are issued to the original payment method within 5 to 7 business days after the return is inspected.
Final sale items and personalized products cannot be returned unless they arrive damaged or defective.

## Shipping timelines
Standard shipping within the continental United States usually takes 3 to 5 business days.
Expedited shipping usually takes 1 to 2 business days.
Orders placed after 3 PM Eastern Time are processed on the next business day.
Customers receive a tracking email as soon as the order ships.

## Exchanges
We currently do not offer direct exchanges.
Customers should return the original item and place a new order for the replacement size or color.

## Damaged or incorrect items
If a package arrives damaged or contains the wrong item, customers should contact support within 7 days of delivery.
The support team will provide a prepaid return label and prioritize a replacement or refund review.

## International shipping
International shipping is available to select countries and usually takes 7 to 14 business days.
Customs delays may extend the final delivery time.
"""


def ensure_google_api_key() -> str:
    """Return the Gemini API key or raise a helpful error."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. Export your Google Gemini API key before "
            "running data_setup.py."
        )
    return api_key


def write_dummy_faq(path: Path = FAQ_PATH) -> Path:
    """Write the dummy FAQ corpus to disk if it does not already exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(FAQ_TEXT, encoding="utf-8")
    return path


def load_documents(path: Path = FAQ_PATH) -> list[Document]:
    """Load the FAQ text as LangChain documents."""
    loader = TextLoader(str(path), encoding="utf-8")
    return loader.load()


def split_documents(documents: Iterable[Document]) -> list[Document]:
    """Chunk the FAQ into retrieval-friendly pieces."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=60,
        separators=["\n## ", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(list(documents))


def build_vector_store(persist_directory: Path = CHROMA_DIR) -> Chroma:
    """Build and persist a local Chroma vector store from the dummy FAQ data."""
    ensure_google_api_key()
    faq_path = write_dummy_faq()
    raw_documents = load_documents(faq_path)
    chunked_documents = split_documents(raw_documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )

    vector_store = Chroma.from_documents(
        documents=chunked_documents,
        embedding=embeddings,
        persist_directory=str(persist_directory),
        collection_name="sportswear_faq",
    )
    return vector_store


def main() -> None:
    """CLI entrypoint for building the local FAQ vector database."""
    vector_store = build_vector_store()
    print(f"Indexed {vector_store._collection.count()} FAQ chunks into {CHROMA_DIR}")


if __name__ == "__main__":
    main()
