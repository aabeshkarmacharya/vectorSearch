import uuid

from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from utils import chunk_text

COLLECTION_NAME = "documents"
app = FastAPI(title="BGE Embeddings")

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

qdrant = QdrantClient(url="http://qdrant:6333")


class EmbedRequest(BaseModel):
    document: str
    doc_id: str | None = None
    metadata: dict = dict()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/embed")
def embed(req: EmbedRequest):
    doc_id = req.doc_id or str(uuid.uuid4())
    chunks: list[str] = chunk_text(req.document)
    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        batch_size=32
    )
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        points.append({
            "id": uuid.uuid4(),
            "vector": embedding.tolist(),
            "payload": {
                "doc_id": doc_id,
                "chunk_id": i,
                "text": chunk,
                **req.metadata
            }
        })
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

    return {
        "status": "ingested",
        "doc_id": doc_id,
        "chunks": len(points)
    }


@app.post("/query")
def query(req: QueryRequest):
    query_embedding = model.encode(
        req.query,
        normalize_embeddings=True
    )

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=req.top_k
    )

    return {
        "query": req.query,
        "results": [
            {
                "score": r.score,
                "text": r.payload["text"],
                "doc_id": r.payload["doc_id"],
                "chunk_id": r.payload["chunk_id"],
            }
            for r in results.points
        ]
    }
