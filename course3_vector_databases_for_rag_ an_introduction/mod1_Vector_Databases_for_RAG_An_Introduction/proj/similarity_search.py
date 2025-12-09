# Import necessary Modules from ChromaDB package
import chromadb
from chromadb.utils import embedding_functions

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

# Define the embedding function using SentenceTransformer
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create a new instance of ChromaClient to interact with the chromaDB
client = chromadb.Client()

# Define the name for the collection to be created and retrieved

collection_name = "myGroceryCollection"

# Define the main function to interact with ChromaDB

def main():
    try:
        # Database Operations will be  inside this block
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "A collection for storing grocery data"},
            embedding_function=ef
        )
        
        pass
    
    except Exception as error: # Catch any errors and log them to the console
        print(f"Error occurred: {error}")