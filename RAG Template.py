"""
RAG (Retrieval-Augmented Generation) Pipeline
=============================================
Stack:
  - sentence-transformers  →  local embeddings
  - FAISS                  →  vector store
  - OpenAI GPT             →  answer generation

Install:
  pip install sentence-transformers faiss-cpu openai numpy

Set your key:
  export OPENAI_API_KEY="sk-..."
"""

import os
from typing import List, Dict
from dataclasses import dataclass, field

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# ── Config ────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL       = "gpt-4o-mini"
TOP_K           = 3
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Document:
    content:  str
    metadata: Dict = field(default_factory=dict)


# ── 1. Loader ─────────────────────────────────────────────────────────────────

def load_documents(file_paths: List[str]) -> List[Document]:
    docs = []
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            docs.append(Document(content=f.read(), metadata={"source": path}))
    print(f"[Loader] Loaded {len(docs)} file(s).")
    return docs


# ── 2. Chunker ────────────────────────────────────────────────────────────────

def chunk_documents(docs: List[Document], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[Document]:
    chunks = []
    for doc in docs:
        text, start = doc.content, 0
        while start < len(text):
            chunks.append(Document(content=text[start:start+chunk_size], metadata=doc.metadata))
            start += chunk_size - overlap
    print(f"[Chunker] Created {len(chunks)} chunk(s).")
    return chunks


# ── 3. Vector Store ───────────────────────────────────────────────────────────

class VectorStore:
    def __init__(self, model_name=EMBEDDING_MODEL):
        self.embedder = SentenceTransformer(model_name)
        self.index:  faiss.Index    = None
        self.chunks: List[Document] = []

    def build(self, chunks: List[Document]):
        self.chunks = chunks
        emb = self._embed([c.content for c in chunks])
        self.index = faiss.IndexFlatL2(emb.shape[1])
        self.index.add(emb)
        print(f"[VectorStore] Indexed {len(chunks)} chunk(s).")

    def search(self, query: str, top_k=TOP_K) -> List[Document]:
        _, idxs = self.index.search(self._embed([query]), top_k)
        return [self.chunks[i] for i in idxs[0] if i < len(self.chunks)]

    def _embed(self, texts):
        return self.embedder.encode(texts, convert_to_numpy=True).astype(np.float32)


# ── 4. Generator ──────────────────────────────────────────────────────────────

def generate_answer(query: str, context_chunks: List[Document], model=LLM_MODEL) -> str:
    client  = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    context = "\n\n---\n\n".join(c.content for c in context_chunks)
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": (
                "You are a helpful assistant. "
                "Answer ONLY using the context below. "
                "If the answer isn't there, say \"I don't know\"."
            )},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    )
    return response.choices[0].message.content.strip()


# ── 5. Pipeline ───────────────────────────────────────────────────────────────

class RAGPipeline:
    def __init__(self):
        self.store = VectorStore()

    def ingest(self, file_paths: List[str]):
        """Load → chunk → index."""
        self.store.build(chunk_documents(load_documents(file_paths)))

    def query(self, question: str) -> str:
        """Retrieve → generate → return."""
        print(f"\n[Query] {question}")
        chunks = self.store.search(question)
        print(f"[Retriever] Using {len(chunks)} chunk(s) as context.")
        return generate_answer(question, chunks)


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with open("sample.txt", "w") as f:
        f.write(
            "Retrieval-Augmented Generation (RAG) combines information retrieval "
            "with a large language model. It fetches relevant passages from an "
            "external knowledge base and feeds them as context to the LLM.\n\n"
            "This reduces hallucinations because the model grounds its answer in "
            "real source material rather than its training weights alone."
        )

    rag = RAGPipeline()
    rag.ingest(["sample.txt"])
    print("\n[Answer]")
    print(rag.query("Why does RAG reduce hallucinations?"))