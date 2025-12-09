import math
import numpy as np
import scipy
import torch
from  sentence_transformers import SentenceTransformer

import warnings
warnings.filterwarnings('ignore')

# Example documents
documents = [
    'Bugs introduced by the intern had to be squashed by the lead developer.',
    'Bugs found by the quality assurance engineer were difficult to debug.',
    'Bugs are common throughout the warm summer months, according to the entomologist.',
    'Bugs, in particular spiders, are extensively studied by arachnologists.'
]

# Load pre-trained sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Generate embeddings
embeddings = model.encode(documents)


# print("Embeddings:")
# for i, emb in enumerate(embeddings):
#     print(f"Document {i+1} embedding: {emb}\n")

# Manual implementation of L2 distance calculation
def euclidean_distance_function(vec1, vec2):
    square_sum = sum((x-y) ** 2 for x, y in zip(vec1, vec2))
    return math.sqrt(square_sum)

# print("Euclidean Distance of Embedding 0 and 1: \n")
# print(euclidean_distance_function(embeddings[0], embeddings[1]))

# print("Euclidean Distance of all Embedding: \n")

# l2_distance_manual = np.zeros((len(embeddings), len(embeddings)))
# for i in range(embeddings.shape[0]):
#     for j in range(embeddings.shape[0]):
#         l2_distance_manual[i][j] = euclidean_distance_function(embeddings[i], embeddings[j])

# print(l2_distance_manual) # Output the full distance matrix
# print(l2_distance_manual[1,0]) # Output the distance between embedding 1 and 0

# Make the manual calculation more efficient
# l2_distance_manual_improved = np.zeros((len(embeddings), len(embeddings)))
# for i in range(embeddings.shape[0]):
#     for j in range(embeddings.shape[0]):
#         l2_distance_manual_improved[i][j] = np.linalg.norm(embeddings[i] - embeddings[j])

# manual method
# l2_dist_manual_improved = np.zeros([4,4])
# for i in range(embeddings.shape[0]):
#     for j in range(embeddings.shape[0]):
#         if j > i: # Calculate the upper triangle only
#             l2_dist_manual_improved[i,j] = euclidean_distance_function(embeddings[i], embeddings[j])
#         elif i > j: # Copy the uper triangle to the lower triangle
#             l2_dist_manual_improved[i,j] = l2_dist_manual_improved[j,i]
            
# print("L2 Distance Matrix (Improved Manual Calculation): \n")
# print(l2_dist_manual_improved)

# Using scipy to calculate L2 distance
# l2_distance_scipy = scipy.spatial.distance.cdist(embeddings, embeddings, metric='euclidean')
# print("L2 Distance Matrix (SciPy Calculation): \n")
# print(l2_distance_scipy)

# Dot Product Similarity and Distance
def dot_product_function(vec1, vec2):
    return sum(x*y for x, y in zip(vec1, vec2))

# print("\nDot Product Similarity of Embedding 0 and 1:")
# print(dot_product_function(embeddings[0], embeddings[1]))

# Dot Product Manual Calculation
dot_product_manual = np.zeros((len(embeddings), len(embeddings)))
for i in range(embeddings.shape[0]):
    for j in range(embeddings.shape[0]):
        dot_product_manual[i][j] = dot_product_function(embeddings[i], embeddings[j])
        
# print("\nDot Product Similarity Matrix (Manual Calculation): ")
# print(dot_product_manual)        

# Calculating Dot Product using matrix multiplication
dot_product_operator = embeddings @ embeddings.T

# print(dot_product_operator)

# print(np.allclose(dot_product_manual, dot_product_operator, atol=1e-05))  # Verify both methods give the same result

# print(np.matmul(embeddings, embeddings.T))

# print("\n")

# print(np.dot(embeddings, embeddings.T))

# Calculate Dot Product Distance
dot_product_distance = 1 - dot_product_manual
# print("\nDot Product Distance Matrix: ")
# print(dot_product_distance)

# Manual implementation of Cosine Similarity calculation

# Calculate the L2 norm
l2_norms = np.sqrt(np.sum(embeddings**2, axis=1))
# print (l2_norms)

l2_norms_reshaped = l2_norms.reshape(-1, 1)
# print (l2_norms_reshaped)

# Normalize the embeddings vectors
normalized_embeddings_manual = embeddings / l2_norms_reshaped
# print (normalized_embeddings_manual)

# Verify the vectos are normalized
# np.sqrt(np.sum(normalized_embeddings_manual**2, axis=1))
# print("\nAre the vectors normalized? ", np.allclose(np.linalg.norm(normalized_embeddings_manual, axis=1), 1))

# Normalize Embeddings using PyTorch

normalized_embeddings_torch = torch.nn.functional.normalize(torch.from_numpy(embeddings)).numpy()
# print(normalized_embeddings_torch)

# np.allclose(normalized_embeddings_manual, normalized_embeddings_torch)
# print("\nAre the vectors normalized? ", np.allclose(np.linalg.norm(normalized_embeddings_torch, axis=1), 1))

# Calculate Cosine Similarity manually

dot_product_function(normalized_embeddings_manual[0], normalized_embeddings_manual[1])
# print("\nCosine Similarity of Embedding 0 and 1 (Manual Calculation):")
# print(dot_product_function(normalized_embeddings_manual[0], normalized_embeddings_manual[1]))

# calculate cosine similarity between all normalized vectors
cosine_similarity_manual = np.empty((len(embeddings), len(embeddings)))
for i in range(normalized_embeddings_manual.shape[0]):
    for j in range(normalized_embeddings_manual.shape[0]):
        cosine_similarity_manual[i][j] = dot_product_function(normalized_embeddings_manual[i], normalized_embeddings_manual[j])
        
# print("\nCosine Similarity Matrix (Manual Calculation):")
# print(cosine_similarity_manual)

# calculate cosine similarity using matrix multiplication
cosine_similarity_operator = normalized_embeddings_manual @ normalized_embeddings_manual.T

# print("\nCosine Similarity Matrix (Operator Calculation):")
# print(cosine_similarity_operator)

# print("\nAre the matrices equal? ", np.allclose(cosine_similarity_manual, cosine_similarity_operator, atol=1e-05))  

# Calculate Cosine Distance
cosine_distance = 1 - cosine_similarity_operator
# print("\nCosine Distance Matrix:")
# print(cosine_distance)

# SIMILARITY SEARCH Using a Query

documents = [
    'Bugs introduced by the intern had to be squashed by the lead developer.',
    'Bugs found by the quality assurance engineer were difficult to debug.',
    'Bugs are common throughout the warm summer months, according to the entomologist.',
    'Bugs, in particular spiders, are extensively studied by arachnologists.',
]

# embed the query
query_embedding = model.encode([input("Enter your query: ")])

# normalize the query embedding
normalized_query_embedding = torch.nn.functional.normalize(torch.from_numpy(query_embedding)).numpy()

# calculate the cosine similarity between the query and all document embeddings by using the dot product
cosine_similarity_query = normalized_embeddings_manual @ normalized_query_embedding.T

# find the position of the vector with the highest cosine similarity
highest_cosine_position = cosine_similarity_query.argmax()

# find the document in that position in the `documents` array
most_similar_document = documents[highest_cosine_position]
print("Most similar document:", most_similar_document)
