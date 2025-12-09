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
            metadata={"description": "A collection for storing grocery data",
                      "hnsw:space": "cosine"},
            embedding_function=ef
        )
        
        print(f"Collection created with name: {collection.name}")
        
        # Array of grocery-realted items in text
        texts = [
            'fresh red apples',
            'organic bananas',
            'ripe mangoes',
            'whole wheat bread',
            'farm-fresh eggs',
            'natural yogurt',
            'frozen vegetables',
            'grass-fed beef',
            'free-range chicken',
            'fresh salmon fillet',
            'aromatic coffee beans',
            'pure honey',
            'golden apple',
            'red fruit'
        ]
        
        # Create a list of unique IDs for each item
        ids = [f"food_{index + 1}" for index, _ in enumerate(texts)]
        
        # Add documents and their corresponding IDs to the collection
        collection.add(
            documents=texts,
            metadatas=[{"source": "grocery_store", "category": "food"} for _ in texts],
            ids=ids
        )
        
        # Retrieve all the items (documents) stored in the collection
        all_items = collection.get()
        
        # Log the retrieved items to the console
        print("Collection contents:")
        print(f"Number of documents: {len(all_items['documents'])}")
        
        # Perform a similarity search for a sample query
        def perform_similarity_search(collection, all_items):
            try:
                # Similarity search code inside this block
                
                # Define the query term you want to search for in the collection
                # query_term = "apple"
                query_term = ["red", "fresh"]
                if isinstance(query_term, str):
                    query_term = [query_term]
                
                # Perform a query to search for the most similar documents to the 'query_term'
                results = collection.query(
                    query_texts=query_term,
                    n_results=3 # Retrieve top 3 similar results
                )
                
                print(f"Query results for '{query_term}':")
                print(results)
                
                # Check if no results are returned or if the results array is empty
                if not results or not results['ids'] or len(results['ids'][0]) == 0:
                    print(f"No similar items found for query '{query_term}'.")
                    return
                
                # Display the top documents whose distance is closest to the query compare to other text data
                for q in range(len(query_term)):
                    print(f"Top 3 similar documents to '{query_term[q]}':")    
                    
                    # Access the nested array in 'results["ids"]' and 'results["distances"]'
                    for i in range(min(3, len(results['ids'][q]))):
                        doc_id = results['ids'][q][i] # Get ID from 'ids' array
                        score = results['distances'][q][i] # Get distance score from 'distances' array
                        
                        # Retrieve text data from the results
                        text = results['documents'][q][i]
                        if not text:
                            print(f' - ID: {doc_id}, Text: "Text not available", Score: {score:.4f}')
                        else:
                            print(f' - ID: {doc_id}, Text: "{text}", Score: {score:.4f}')
            except Exception as error:
                print(f"Error in similarity search: {error}")
        
        perform_similarity_search(collection, all_items)
    
    except Exception as error: # Catch any errors and log them to the console
        print(f"Error occurred: {error}")
        
if __name__ == "__main__":
    main()