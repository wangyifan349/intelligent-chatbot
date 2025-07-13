from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Biomedical QA dictionary
QA_dict = {
    "What is DNA?": "DNA is a molecule that carries genetic instructions in living organisms.",
    "What causes diabetes?": "Diabetes is caused by high blood sugar due to insulin issues.",
    "How does vaccination work?": "Vaccination trains the immune system to recognize and fight pathogens.",
    "What is a virus?": "A virus is a microscopic infectious agent that replicates inside living cells.",
    "What are antibiotics?": "Antibiotics are drugs that kill or stop the growth of bacteria.",
    "What is the human genome?": "The human genome is the complete set of genetic information in humans.",
    "How do neurons communicate?": "Neurons communicate via electrical and chemical signals.",
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
