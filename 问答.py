from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
# 加载社区微调的句子编码器
model = SentenceTransformer('all-MiniLM-L6-v2')
# 生物医学领域的简单QA字典
QA_dict = {
    "What is DNA?": "DNA is a molecule that carries genetic instructions in living organisms.",
    "What causes diabetes?": "Diabetes is caused by high blood sugar due to insulin issues.",
    "How does vaccination work?": "Vaccination trains the immune system to recognize and fight pathogens.",
    "What is a virus?": "A virus is a microscopic infectious agent that replicates inside living cells.",
    "What are antibiotics?": "Antibiotics are drugs that kill or stop the growth of bacteria.",
    "What is the human genome?": "The human genome is the complete set of genetic information in humans.",
    "How do neurons communicate?": "Neurons communicate via electrical and chemical signals.",
}
query = "What do vaccines do?"
# 编码输入查询
query_emb = model.encode(query, convert_to_tensor=True)
# 把所有问题先放进列表
questions = []
for q in QA_dict:
    questions.append(q)
# 批量编码所有问题
questions_emb = model.encode(questions, convert_to_tensor=True)
# 计算余弦相似度
cos_scores = F.cosine_similarity(query_emb.unsqueeze(0), questions_emb, dim=1)
# 打印每个候选问题的相似度
for i in range(len(questions)):
    print("Question:", questions[i])
    print("Similarity score:", cos_scores[i].item())
    print()
# 找出最高相似度的答案
best_score = -1.0
best_answer = "Sorry, I don't know the answer."
for i in range(len(questions)):
    if cos_scores[i] > best_score:
        best_score = cos_scores[i]
        best_answer = QA_dict[questions[i]]
print("Query:", query)
print("Best answer:", best_answer)
print("Best score:", best_score.item())
