import chromadb
from chromadb.utils import embedding_functions

import warnings
warnings.filterwarnings("ignore")

import json
import re
import numpy as np
from typing import List, Dict, Any, Optional


# Initialize ChromaDB client
chroma_client = chromadb.Client()

# Create the data loading function
def load_food_data(file_path: str) -> List[Dict]:
    """Load Food data from a JSON file."""
    
    try:
        
        with open (file_path, 'r', encoding='utf-8') as file:
            food_data = json.load(file)
        
        # Ensure each item has required fields and normalize the structure
        for i, item in enumerate(food_data):
            # Normalize food_id to string
            if 'food_id' not in item:
                item['food_id'] = str(i + 1)
            else:
                item['food_id'] = str(item['food_id'])
                
            # Ensure required fields exist
            if 'food_ingredients' not in item:
                item['food_ingredients'] = []
            if 'food_description' not in item:
                item['food_description'] = ''
            if 'cuisine_type' not in item:
                item['cuisine_type'] = 'Unknown'
            if 'food_calories_per_serving' not in item:
                item['food_calories_per_serving'] = 0
                
            # Extract taste features from nested food_features if available
            if 'food_features' in item and isinstance(item['food_features'], dict):
                taste_features = []
                for key, value in item['food_features'].items():
                    if value:
                        taste_features.append(str(value))
                item['tast_profile'] = ', '.join(taste_features)
            else:
                item['tast_profile'] = ''
        
        print(f"Successfully loaded {len(food_data)} food items from {file_path}")
        return food_data
    
    except Exception as e:
        print(f"Error loading food data: {e}")
        return []
    
# Create the collection setup function
def create_similarity_search_collection(
    collection_name: str,
    collection_metadata: dict = None):
    """Create a ChromaDB collection with sentense transformer embeddings for similarity search."""
    
    try:
        # Try to delete existing collection if it exists to start fresh
        try:
            chroma_client.delete_collection(name=collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass  # Collection does not exist, no need to delete
        
        # Create embedding function using SentenceTransformer
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create new collection
        return chroma_client.create_collection(
            name = collection_name,
            metadata = collection_metadata,
            embedding_function = sentence_transformer_ef
            )
        
        
    except Exception as e:
        print(f"Error creating collection: {e}")
        return None
    
# Create the data population function
def populate_similarity_collection(collection, food_items: List[Dict]):
    """Populate collection with food data and generate embeddings."""
    documents = []
    ids = []
    metadatas = []
    
    # Create unique IDs to avoid duplication
    used_ids = set()
    
    for i, food in enumerate(food_items):
        # Create comprehensive text for embedding using rich JSON structure
        text = f"Name: {food.get('food_name')}."
        text += f"Description: {food.get('food_description')}."
        text += f"Ingredients: {', '.join(food.get('food_ingredients', []))}."
        text += f"Cuisine Type: {food.get('cuisine_type', 'Unknown')}."
        text += f"Cooking Method: {food.get('cooking_method')}."
        
        # Add taste profile from food_features
        taste_profile = food.get('tast_profile', '')
        if taste_profile:
            text += f" Taste and features: {taste_profile}."
            
        # Add health benefits if available
        health_benefits = food.get('health_benefits', '')
        if health_benefits:
            text += f" Health Benefits: {health_benefits}."
        
        # Add Nutritional Information if available
        if 'food_nutritional_factors' in food:
            nutrition = food['food_nutritional_factors']
            if isinstance(nutrition, dict):
                nutrition_info = ', '.join([f"{k}: {v}" for k, v in nutrition.items()])
                text += f" Nutritional Information: {nutrition_info}."
        
        # Generate unique ID to avoid duplication
        base_id = str(food.get('food_id', i))
        unique_id = base_id
        counter = 1
        while unique_id in used_ids:
            unique_id = f"{base_id}_{counter}"
            counter += 1
        used_ids.add(unique_id)
        
        documents.append(text)
        ids.append(unique_id)
        metadatas.append({
            "name": food["food_name"],
            "cuisine_type": food.get("cuisine_type", "Unknown"),
            "ingredients": ", ".join(food.get("food_ingredients", [])),
            "calories": food.get("food_calories_per_serving", 0),
            "description": food.get("food_description", ""),
            "cooking_method": food.get("cooking_method", ""),
            "health_benefits": food.get("health_benefits", ""),
            "taste_profile": food.get("tast_profile", "")
        })
        
    # Add all data to collection
    collection.add(
        documents = documents,
        ids = ids,
        metadatas = metadatas
    )
    
    print(f"Added {len(documents)} food items to the collection.")
    

# Create the Basic Similarity Search function
def perform_similarity_search(collection, query: str, n_results: int = 5) -> List[Dict]:
    """Perform similarity search in the collection for the given query."""
    try:
        results = collection.query(
            query_texts = [query],
            n_results = n_results
        )
        
        if not results or not results['ids'] or len(results['ids'][0]) == 0:
            print("No results found.")
            return []
        
        formatted_results = []
        for i in range(len(results['ids'][0])):
            # Callculate similarity score (assuming cosine similarity)
            similarity_score = 1 - results['distances'][0][i]
            
            result = {
                "food_id": results['ids'][0][i],
                "food_name": results['metadatas'][0][i].get('name'),
                "food_description": results['metadatas'][0][i]['description'],
                "cuisine_type": results['metadatas'][0][i]['cuisine_type'],
                "food_calories_per_serving": results['metadatas'][0][i]['calories'],
                "similarity_score": similarity_score,
                "distance": results['distances'][0][i]
            }
            
            formatted_results.append(result)
        
        return formatted_results
    
    except Exception as e:
        print(f"Error performing similarity search: {e}")
        return []
    
# Create the Filtered Similarity Search function
def perform_filtered_similarity_search(
    collection,
    query: str,
    cuisine_filter: str = None,
    max_calories: int = None,
    n_results: int = 5) -> List[Dict]:
    """Perform filtered similarity search with metadata constraints."""
    
    where_clauses = None
    
    # Build Filters List
    filters = []
    if cuisine_filter:
        filters.append({
            "cuisine_type": cuisine_filter})
        
    if max_calories:
        filters.append({
            "calories": {"$lte": max_calories}
        })
        
    # Construct where_clauses based on number of filters
    if len(filters) == 1:
        where_clauses = filters[0]
    elif len(filters) > 1:
        where_clauses = {"$and": filters}
        
    try:
        results = collection.query(
            query_texts = [query],
            n_results = n_results,
            where = where_clauses
        )
        
        if not results or not results['ids'] or len(results['ids'][0]) == 0:
            print("No results found with the given filters.")
            return []
        
        formatted_results = []
        for i in range(len(results['ids'][0])):
            # Callculate similarity score (assuming cosine similarity)
            similarity_score = 1 - results['distances'][0][i]
            
            result = {
                "food_id": results['ids'][0][i],
                "food_name": results['metadatas'][0][i]['name'],
                "food_description": results['metadatas'][0][i]['description'],
                "cuisine_type": results['metadatas'][0][i]['cuisine_type'],
                "food_calories_per_serving": results['metadatas'][0][i]['calories'],
                "similarity_score": similarity_score,
                "distance": results['distances'][0][i]
            }
            formatted_results.append(result)
        
        return formatted_results
    
    except Exception as e:
        print(f"Error performing filtered similarity search: {e}")
        return []
        
        