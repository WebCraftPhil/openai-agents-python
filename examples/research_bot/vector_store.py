from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from typing import Iterable, Sequence

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

DEFAULT_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "tech-trends")
DEFAULT_DIMENSION = int(os.getenv("PINECONE_DIMENSION", "1536"))
DEFAULT_METRIC = os.getenv("PINECONE_METRIC", "cosine")
DEFAULT_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
DEFAULT_REGION = os.getenv("PINECONE_REGION", "us-east-1")
EMBEDDING_MODEL = os.getenv("PINECONE_EMBEDDING_MODEL", "text-embedding-3-small")


@lru_cache(maxsize=1)
def _pinecone_client() -> Pinecone:
    """Return a cached Pinecone client configured from environment variables."""

    api_key = os.environ["PINECONE_API_KEY"]
    return Pinecone(api_key=api_key)


@lru_cache(maxsize=1)
def _openai_client() -> OpenAI:
    """Return a cached OpenAI client for embedding generation."""

    return OpenAI()


@lru_cache(maxsize=1)
def get_index():
    """Ensure the Pinecone index exists and return a handle to it."""

    client = _pinecone_client()
    existing_indexes = {idx["name"] for idx in client.list_indexes()}
    if DEFAULT_INDEX_NAME not in existing_indexes:
        client.create_index(
            name=DEFAULT_INDEX_NAME,
            dimension=DEFAULT_DIMENSION,
            metric=DEFAULT_METRIC,
            spec=ServerlessSpec(cloud=DEFAULT_CLOUD, region=DEFAULT_REGION),
        )
    return client.Index(DEFAULT_INDEX_NAME)


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Generate embeddings for the provided texts using the configured model."""

    response = _openai_client().embeddings.create(model=EMBEDDING_MODEL, input=list(texts))
    return [item.embedding for item in response.data]


def upsert_snippets(snippets: Iterable[dict[str, str]]) -> None:
    """Upsert the supplied news snippets into Pinecone."""

    snippets_list = list(snippets)
    if not snippets_list:
        return

    texts = [snippet["summary"] for snippet in snippets_list]
    embeddings = embed_texts(texts)
    index = get_index()

    vectors = []
    for snippet, embedding in zip(snippets_list, embeddings, strict=True):
        url = snippet["url"]
        identifier = hashlib.sha1(url.encode("utf-8"), usedforsecurity=False).hexdigest()
        metadata = {
            "title": snippet.get("title", ""),
            "url": url,
            "summary": snippet.get("summary", ""),
            "source": snippet.get("source", ""),
            "published_at": snippet.get("published_at", ""),
        }
        vectors.append((identifier, embedding, metadata))

    index.upsert(vectors=vectors)


def query_snippets(query: str, top_k: int = 5) -> list[dict[str, str]]:
    """Query Pinecone for snippets matching the user's prompt."""

    [embedding] = embed_texts([query])
    results = get_index().query(vector=embedding, top_k=top_k, include_metadata=True)
    matches: list[dict[str, str]] = []
    for match in results.get("matches", []):
        metadata = match.get("metadata", {}) or {}
        matches.append({
            "title": metadata.get("title", ""),
            "url": metadata.get("url", ""),
            "summary": metadata.get("summary", ""),
            "source": metadata.get("source", ""),
            "published_at": metadata.get("published_at", ""),
        })
    return matches