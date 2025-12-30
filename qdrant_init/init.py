from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

COLLECTION = "documents"
VECTOR_SIZE = 768  # bge-base-en-v1.5

client = QdrantClient(url="http://qdrant:6333")

collections = [c.name for c in client.get_collections().collections]

if COLLECTION not in collections:
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"Created collection: {COLLECTION}")
else:
    print(f"Collection already exists: {COLLECTION}")

client.collection_exists(
    collection_name=COLLECTION,
)
