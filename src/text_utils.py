from typing import List, TypedDict, Literal, Annotated

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

def get_docs_from_url(url: str) -> List[Document]:
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs

def split_text(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 0) -> List[Document]:
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_docs = text_splitter.split_documents(docs)
    return split_docs

def get_split_page_content_from_url(url: str, chunk_size: int = 1000, chunk_overlap: int = 0) -> List[str]:
    docs = get_docs_from_url(url)
    split_docs = split_text(docs, chunk_size, chunk_overlap)
    page_content = [doc.page_content for doc in split_docs]
    return page_content

