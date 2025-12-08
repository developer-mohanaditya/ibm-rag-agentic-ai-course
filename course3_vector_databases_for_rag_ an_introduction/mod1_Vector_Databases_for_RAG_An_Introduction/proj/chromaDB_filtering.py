import chromadb
from chromadb.utils import embedding_functions

import warnings
warnings.filterwarnings('ignore')

ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Create a Chroma client
client = chromadb.Client()

# Create a collection with the embedding function
# collection = client.create_collection(
#     name="filtering_demo", 
#     metadata={"description": "A demo collection for filtering in ChromaDB"},
#     configuration={"embedding_function": ef})

collection = client.create_collection(
    name="filtering_demo",
    metadata={"description": "A demo collection for filtering in ChromaDB"},
    embedding_function=ef
)

print(f"Collection created with name: {collection.name}")

# Add documents with metadata
collection.add(
    documents=[
         "This is a document about LangChain",
        "This is a reading about LlamaIndex",
        "This is a book about Python",
        "This is a document about pandas",
        "This is another document about LangChain"
    ],
    metadatas=[
         {"source": "langchain.com", "version": 0.1},
        {"source": "llamaindex.ai", "version": 0.2},
        {"source": "python.org", "version": 0.3},
        {"source": "pandas.pydata.org", "version": 0.4},
        {"source": "langchain.com", "version": 0.5},
    ],
    ids=["id1", "id2", "id3", "id4", "id5"]
)

# Filter using Metadata
# print(collection.get(where={"source": {"$eq": "langchain.com"}}))

# print(collection.get(where={"$and": [
#     {"source": {"$eq": "langchain.com"}},
#     {"version": {"$lt": 0.3}}
# ]}))


# Filter using Document Content
# print(collection.get(where_document={"$contains": "pandas"}))


# Combined Filtering
print(collection.get(
    where={"version": {"$gt": 0.1}},
    where_document={"$or": [
                    {"$contains": "LangChain"},
                    {"$contains": "Python"}]}
))

