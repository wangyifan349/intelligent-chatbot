from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F

# 1. 加载预训练模型
model = SentenceTransformer('all-MiniLM-L6-v2')
# 2. 保存到当前目录下的子文件夹，比如 "./all-MiniLM-L6-v2"
model.save('./all-MiniLM-L6-v2')
# 3. 验证：从本地加载
model_local = SentenceTransformer('./all-MiniLM-L6-v2')


# Biomedical QA dictionary using triple quotes for better formatting and to avoid escape issues
QA_dict = {
    """What is DNA?""": """DNA is a molecule that carries genetic instructions in living organisms.""",
    """What causes diabetes?""": """Diabetes is caused by high blood sugar due to insulin issues.""",
    """How does vaccination work?""": """Vaccination trains the immune system to recognize and fight pathogens.""",
    """What is a virus?""": """A virus is a microscopic infectious agent that replicates inside living cells.""",
    """What are antibiotics?""": """Antibiotics are drugs that kill or stop the growth of bacteria.""",
    """What is the human genome?""": """The human genome is the complete set of genetic information in humans.""",
    """How do neurons communicate?""": """Neurons communicate via electrical and chemical signals.""",
}

questions = list(QA_dict.keys())
questions_emb = model.encode(questions, convert_to_tensor=True)

while True:
    query = input("Your question (type 'exit' to quit): ")
    if query.lower() == 'exit':
        print("Exiting the system. Goodbye!")
        break

    query_emb = model.encode(query, convert_to_tensor=True)
    cos_scores = F.cosine_similarity(query_emb.unsqueeze(0), questions_emb, dim=1)

    top_k = 1
    top_k_indices = torch.topk(cos_scores, top_k).indices
    for idx in top_k_indices:
        best_question = questions[idx]
        best_answer = QA_dict[best_question]
        best_score = cos_scores[idx].item()

        print(f"\nMatching Question: {best_question}")
        print(f"Answer: {best_answer}")
        print(f"Similarity Score: {best_score:.4f}\n")



"""
# Required libraries:
# pip install sentence-transformers faiss-gpu torch numpy
#This is a version that uses GPU to run the model, theoretically faster and supporting larger scale data vector retrieval.

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
# Device setup: Use GPU if available for both encoding and FAISS search
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Initialize sentence-transformers model (loads into specified device)
model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
# Example QA dataset: dictionary of question -> answer
QA_dict = {
    "What is DNA?": "DNA is a molecule that carries genetic instructions in living organisms.",
    "What causes diabetes?": "Diabetes is caused by high blood sugar due to insulin issues.",
    "How does vaccination work?": "Vaccination trains the immune system to recognize and fight pathogens.",
    "What is a virus?": "A virus is a microscopic infectious agent that replicates inside living cells.",
    "What are antibiotics?": "Antibiotics are drugs that kill or stop the growth of bacteria.",
    "What is the human genome?": "The human genome is the complete set of genetic information in humans.",
    "How do neurons communicate?": "Neurons communicate via electrical and chemical signals."
}
questions = list(QA_dict.keys())
# Step 1: Encode all questions to vectors on GPU
# Output tensor shape: (num_questions, embedding_dim)
questions_emb = model.encode(questions, convert_to_tensor=True, device=device)
# Step 2: Convert embeddings from torch tensor (GPU) to numpy float32 array on CPU for FAISS
questions_np = questions_emb.cpu().numpy().astype('float32')
# Step 3: Normalize vectors to unit length (L2 norm = 1)
# Why normalize?
# Cosine similarity between two vectors a, b is defined as:
# cos_sim = (a · b) / (||a|| * ||b||)
#
# If a and b are normalized to unit vectors (||a||=||b||=1),
# cosine similarity simplifies to their inner (dot) product:
# cos_sim = a · b
#
# Therefore, by normalizing all vectors,
# we can use FAISS's Inner Product (Dot Product) index to retrieve
# based on cosine similarity.
faiss.normalize_L2(questions_np)

# Dimension of embeddings
d = questions_np.shape[1]
# Step 4: Build a FAISS CPU index that uses inner product similarity
cpu_index = faiss.IndexFlatIP(d)  # "Flat" index: brute-force search with inner product metric
cpu_index.add(questions_np)       # Add all question vectors to the index
# Step 5: Move the index from CPU to GPU
res = faiss.StandardGpuResources()             # Initialize GPU resources (one per GPU)
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)  # 0: GPU device ID (adjust if necessary)
# Define a search function for incoming user query(string)
def search(query, top_k=3):
    """
    Encodes query text and performs top_k nearest neighbor search in the FAISS GPU index.
    Returns list of tuples (question, answer, similarity_score).
    """
    # Encode the query sentence (single example) into a GPU tensor
    query_emb = model.encode([query], convert_to_tensor=True, device=device)
    # Convert to numpy float32, move to CPU
    query_np = query_emb.cpu().numpy().astype('float32')
    # Normalize the query vector so that inner product matches cosine similarity
    faiss.normalize_L2(query_np)
    # Search top_k nearest neighbors using GPU index
    distances, indices = gpu_index.search(query_np, top_k)   # distances shape: (1, top_k)
    # Prepare results by mapping indices back to QA pairs
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        question = questions[idx]
        answer = QA_dict[question]
        # dist is cosine similarity score after normalization
        results.append((question, answer, dist))
    return results
# Simple interactive loop for demonstration (remove if embedding in larger application)
print("QA search system. Type 'exit' to quit.")
while True:
    user_query = input("Enter your question: ").strip()
    if user_query.lower() == "exit":
        print("Exiting...")
        break
    top_results = search(user_query, top_k=3)
    print("\nTop matching Q&A results:")
    for rank, (q, a, score) in enumerate(top_results, start=1):
        print(f"Rank {rank}:")
        print(f"Question: {q}")
        print(f"Answer: {a}")
        print(f"Similarity: {score:.4f}\n")
    print("-" * 60)
"""
