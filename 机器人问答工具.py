# -*- coding: utf-8 -*-
import json
import os
import re
# --------------------------------------------------
# 配置区
# --------------------------------------------------
LOCAL_QA_PATH = "external_qa.json"
W_LCS = 0.8
W_EDIT = 0.2
# --------------------------------------------------
# 内置问答字典
# --------------------------------------------------
QA_DICT = {
    "你好": """你好！有什么可以帮您的吗？""",

    "你叫什么名字": """我是智能客服小助手。""",

    "怎么写一个 for 循环": """
下面给你一个简单的 Python for 循环示例：

for i in range(5):
    # 打印当前 i 的值
    print(f"当前 i = {i}")

print("循环结束")
""",

    "谢谢": """不客气，很高兴为您服务！""",

    # —— 新增内容开始 —— #

    "联系方式": """
您可以通过以下方式与我们取得联系：
1. 邮箱：contact@example.com
2. 电话：+86-10-1234-5678
3. 微信：my_wechat_id
4. LinkedIn：linkedin.com/in/your-profile
""",

    "通信方式": """
我们支持邮件、电话、微信和在线即时消息。
如需及时响应，建议添加微信或在工作时间内拨打电话。
""",

    "商业合作": """
非常欢迎各类商业合作机会！
请发送商务需求至邮箱 contact@example.com，
我们会在 1-2 个工作日内与您取得联系。
""",

    "技术合作": """
在技术研发、产品集成、创新项目等方面均可洽谈合作。
详情请联系技术负责人：
    邮箱：tech_lead@example.com
    微信：my_wechat_id
""",

    "可以科技上合作吗": """当然可以！请告诉我们您的项目需求，我们会安排相关技术团队与您对接。"""
    # —— 新增内容结束 —— #
}

QA_DICT.update({
    # —— 联系方式与沟通 —— #
    "电话是多少": """
我们的客服电话：+86-10-1234-5678  
服务时间：周一至周五 09:00–18:00（法定节假日除外）  
如遇紧急事务，请在服务时间外微信联系。  
""",

    "微信号是多少": """
请添加我们的微信进行即时沟通：  
微信号：my_wechat_id  
添加后会有专人尽快与您对接。  
""",

    "邮箱是多少": """
· 商务合作／咨询：contact@example.com  
· 技术支持：tech_lead@example.com  
我们会在1-2个工作日内回复。  
""",

    "工作时间": """
· 周一—周五：09:00–18:00  
· 周末和法定节假日：休息  
如需24小时支持，请提前预约或使用微信联系。  
""",

    # —— 价值观与技术立场 —— #
    "你支持开源吗": """
我们支持开源精神，鼓励社区协作与代码共享；  
同时也理解商业闭源模型的价值和必要性。  
无论是开源项目还是闭源产品，只要能为客户创造价值，  
我们都愿意提供技术支持和合作。  
""",

    "你支持闭源吗": """
我们认可闭源软件在商业模式、知识产权保护和持续盈利方面的优势；  
也倡导在合适场景下开放API或模块化接口，兼顾安全与可扩展。  
""",

    "你反对垄断吗": """
我们反对滥用市场支配地位的垄断行为，  
主张公平竞争、开放生态与多方共赢。  
""",

    "你支持可持续发展吗": """
可持续发展是我们的核心理念之一，  
我们在产品和运营中践行环保、节能减排和社会责任。  
""",

    "你反对数据滥用吗": """
我们坚决反对未经同意的数据采集和滥用，  
严格遵守《个人信息保护法》等相关法规，  
以技术和制度保障用户隐私和数据安全。  
""",

    "你们的核心价值观是什么": """
我们的核心价值观：  
1. 客户至上：倾听需求，超越期望  
2. 创新驱动：技术+场景，持续迭代  
3. 诚信合作：透明沟通，信守承诺  
4. 开放共享：支持开源也支持闭源，共创生态  
5. 社会责任：关注环保、数据安全与公益  
"""
})

# 这是你“append”想要添加的内容，改成字典
addition_dict = {
    # 日常工作
    "如何提高工作效率": """要提高工作效率，可以从以下几方面入手：
1. 制定清晰的工作目标和计划，确定优先级；
2. 使用番茄工作法（25 分钟专注、5 分钟休息）来保持专注；
3. 减少会议频率，简化沟通流程；
4. 善用自动化工具（如脚本、宏、流程管理软件）来处理重复性任务；
5. 保持良好的作息，保证充足的睡眠与运动。""",

    "怎样做好时间管理": """高效的时间管理可以考虑：
1. 记录并分析每天的时间花费，找出低效环节；
2. 使用日历和待办清单工具（如 Google Calendar、Todoist）；
3. 把大任务拆分为小步骤，逐步完成；
4. 设置缓冲时间，预留应对突发状况；
5. 坚持“先做最重要的事”，避免拖延。""",

    # 经济学定律
    "需求定律是什么": """需求定律（Law of Demand）是微观经济学的基本定律，指出在其他条件不变的情况下，商品价格与消费者对该商品的需求量呈反向关系：
- 当价格上升时，需求量下降；
- 当价格下降时，需求量上升。
这一现象的背后原因包括替代效应和收入效应。""",

    "供给定律是什么": """供给定律（Law of Supply）表明，在其他条件不变的情况下，商品价格与生产者愿意提供的数量呈正向关系：
- 价格上升时，生产者愿意提供更多；
- 价格下降时，生产者提供量减少。
这是因为更高的价格提高了生产利润，吸引更多资源投入。""",

    "边际效用递减": """边际效用递减法则（Law of Diminishing Marginal Utility）指出，消费者对同一种商品的边际效用（额外获得的满足）会随着消费数量的增加而递减。当边际效用下降到零或负值时，消费者就不会继续增加消费了。""",

    # 黄金
    "黄金有什么投资价值": """黄金作为贵金属资产具有以下投资价值：
1. 保值与避险：对冲通胀和货币贬值风险；
2. 流动性高：全球市场对黄金需求稳定，买卖方便；
3. 资产多元化：与股票、债券等资产相关性低；
4. 长期增值：历史上黄金价格具备上涨趋势。""",

    "如何选购实物黄金": """购买实物黄金时应注意：
1. 选择正规渠道：银行、知名交易所或品牌金店；
2. 看纯度：常见有 24K、Au999 足金；
3. 检验真伪：关注重量、印记、包装，必要时用专业仪器检测；
4. 考虑存储和保险成本；
5. 关注买卖价差（点差），选择点差较小的时机。""",

    # 比特币
    "什么是比特币": """比特币（Bitcoin，BTC）是一种去中心化的加密数字货币，由“中本聪”在 2008 年提出，并于 2009 年开源运行。它基于区块链技术，所有交易记录公开透明、不可篡改。总量固定为 2100 万枚，因此具备稀缺性和抗通胀属性。""",

    "如何安全保管比特币": """安全保管比特币的常见方式：
1. 硬件钱包：Ledger、Trezor 等；
2. 冷钱包（离线签名）+ 多重签名方案；
3. 备份助记词/私钥，妥善保管于离线安全环境；
4. 避免使用不知名的软件钱包，谨防钓鱼及木马攻击；
5. 启用多重身份验证和加密存储。""",

    # 密码学
    "什么是对称加密": """对称加密（Symmetric Encryption）是指加密和解密使用同一把密钥的加密方式。
优点：算法成熟、加解密速度快，适合大数据量场景。
缺点：密钥分发与管理困难，一旦密钥泄露，消息安全性丧失。""",

    "什么是非对称加密": """非对称加密（Asymmetric Encryption）使用一对密钥：公钥和私钥。
公钥用于加密，私钥用于解密。常见算法有 RSA、ECC。
优点：无需事先共享私钥，密钥管理更加安全；
缺点：加解密速度较慢，通常用于小数据量或密钥交换。""",

    "什么是哈希函数": """哈希函数（Hash Function）将任意长度的输入映射到固定长度的输出（哈希值）。
特点：
1. 单向性：难以从哈希值推回原文；
2. 敏感性：输入微小变化会导致输出大幅变化；
3. 碰撞抗性：找到两个不同输入产生相同哈希值非常困难。
常见算法有 SHA-256、SHA-3、MD5（已不再安全）。""",

    # 医学
    "高血压如何预防": """预防高血压的建议：
1. 均衡饮食：低盐低脂，多吃水果、蔬菜和全谷物；
2. 适量运动：每周至少 150 分钟中等强度有氧运动；
3. 控制体重：保持 BMI 在合理范围；
4. 戒烟限酒：烟酒均可升高血压；
5. 定期体检：早期发现并干预。""",

    "糖尿病症状": """糖尿病常见症状包括：
1. 多尿、口渴；
2. 体重无故下降；
3. 乏力、视力模糊；
4. 愈合缓慢、易感染；
5. 严重时可出现酮症酸中毒或低血糖昏迷。
如有上述症状，应及早就医并检查血糖。""",

    "急救创伤止血方法": """常用止血方法：
1. 直接压迫：用干净纱布或手直接按压伤口；
2. 绑扎止血：压力绷带或干净布条加压包扎；
3. 体位止血：抬高患肢，必要时抬高心脏水平；
4. 静脉止血带：仅在生命危险时短时间使用；
5. 专业救援：尽快送往医院，由医务人员处理。"""
}





```python
QA_DICT = {
    # —— 协同过滤算法 —— #
    "协同过滤算法实现": """
下面是一个基于用户–物品矩阵的协同过滤（UserCF）简单实现，使用 NumPy 计算用户相似度并给出推荐：

import numpy as np

# 示例用户–物品评分矩阵，行：用户，列：物品
R = np.array([
    [5, 0, 3, 0, 2],
    [4, 0, 4, 3, 0],
    [1, 2, 0, 5, 4],
    [0, 3, 5, 4, 0],
    [2, 4, 1, 0, 3]
], dtype=float)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a)==0 or np.linalg.norm(b)==0:
        return 0.0
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

def usercf_recommend(R: np.ndarray, user_id: int, top_k: int = 2, n_rec: int = 3):
    # 计算用户相似度矩阵
    n_users = R.shape[0]
    sim = np.zeros((n_users, n_users))
    for i in range(n_users):
        for j in range(n_users):
            sim[i,j] = cosine_similarity(R[i], R[j])
    # 找到 top_k 最相似用户
    neighbors = np.argsort(-sim[user_id])[:top_k]
    # 预测未评分物品的分数
    scores = np.zeros(R.shape[1])
    for item in range(R.shape[1]):
        if R[user_id, item] == 0:
            # 加权平均
            scores[item] = np.dot(sim[user_id, neighbors], R[neighbors, item]) / (np.sum(sim[user_id, neighbors]) + 1e-9)
    # 推荐分数最高的 n_rec 个物品
    rec_items = np.argsort(-scores)[:n_rec]
    return rec_items, scores[rec_items]

# 示例：为用户 0 推荐
items, scores = usercf_recommend(R, user_id=0, top_k=2, n_rec=3)
print("推荐物品索引：", items, "预测分数：", scores)
""",

    # —— 余弦相似度 —— #
    "怎么计算余弦相似度": """
下面是使用 NumPy 计算向量余弦相似度的函数示例：

import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    \"\"\"计算两个向量 a、b 的余弦相似度\"\"\"
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

# 示例
v1 = np.array([1.0, 2.0, 3.0])
v2 = np.array([2.0, 3.0, 4.0])
print("余弦相似度：", cosine_similarity(v1, v2))
""",

    # —— L2 距离 —— #
    "怎么计算L2距离": """
下面是使用 NumPy 计算欧氏（L2）距离的函数示例：

import numpy as np

def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    \"\"\"计算两个向量 a、b 的欧氏距离\"\"\"
    return float(np.linalg.norm(a - b))

# 示例
v1 = np.array([1.0, 2.0, 3.0])
v2 = np.array([2.0, 3.0, 4.0])
print("L2 距离：", l2_distance(v1, v2))
""",

    # —— 基于 BERT 的问答示例 —— #
    "BERT问答示例": """
下面演示如何使用 Hugging Face Transformers 中的 BERT 模型做简单问答（QA）。  
请先安装依赖：pip install transformers torch

from transformers import BertForQuestionAnswering, BertTokenizer
import torch

# 选择预训练模型
MODEL_NAME = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForQuestionAnswering.from_pretrained(MODEL_NAME)

def bert_qa(question: str, context: str) -> str:
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    # 模型前向
    outputs = model(input_ids, attention_mask=attention_mask)
    start_scores, end_scores = outputs.start_logits, outputs.end_logits
    # 获取答案片段
    start = torch.argmax(start_scores)
    end = torch.argmax(end_scores) + 1
    answer_tokens = input_ids[0][start:end]
    return tokenizer.decode(answer_tokens, skip_special_tokens=True)

# 示例调用
context = (
    "Transformers provides thousands of pretrained models to perform tasks "
    "on different modalities such as text, vision, and audio."
)
question = "What does Transformers provide?"
print("Answer:", bert_qa(question, context))
""",

    # —— 综合智能问答流程示例 —— #
    "智能问答完整流程": """
下面示例整合了字典问答、余弦相似度检索、BERT 本地问答与 OpenAI API 调用：  

```python
import os
import numpy as np
from transformers import BertForQuestionAnswering, BertTokenizer
import torch
import openai

# 1. 问答字典
QA_DICT = {...}  # 上面所有条目

# 2. 简易 Embedding：字符均值
def fake_embedding(text: str) -> np.ndarray:
    arr = np.array([ord(c) for c in text], dtype=float)
    return np.array([arr.mean()]) if arr.size else np.zeros(1)

# 3. 余弦相似度
def cosine_similarity(a, b):
    if np.linalg.norm(a)==0 or np.linalg.norm(b)==0: return 0.0
    return float(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))

# 4. BERT QA 初始化
bert_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = BertTokenizer.from_pretrained(bert_name)
bert_model = BertForQuestionAnswering.from_pretrained(bert_name)

def bert_qa(question, context):
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
    s, e = bert_model(**inputs).start_logits, bert_model(**inputs).end_logits
    start, end = torch.argmax(s), torch.argmax(e) + 1
    return tokenizer.decode(inputs["input_ids"][0][start:end], skip_special_tokens=True)

# 5. OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")
def query_openai(prompt: str) -> str:
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.6, max_tokens=200
    )
    return resp.choices[0].message.content.strip()

# 6. 问答主流程
keys = list(QA_DICT.keys())
embs = [fake_embedding(k) for k in keys]

def get_answer(query: str, threshold=0.8) -> str:
    # 字典精确匹配
    if query in QA_DICT:
        return QA_DICT[query]
    # 相似度检索
    q_emb = fake_embedding(query)
    sims = [cosine_similarity(q_emb, e) for e in embs]
    idx, score = max(enumerate(sims), key=lambda x: x[1])
    if score >= threshold:
        return f"(匹配到「{keys[idx]}」、相似度={score:.2f})\n" + QA_DICT[keys[idx]]
    # 本地 BERT 问答
    example_context = "这里填写上下文，用于 BERT 问答示例。"
    return bert_qa(query, example_context)

# 7. 测试
for q in ["怎么计算余弦相似度？", "协同过滤算法", "What does Transformers provide?"]:
    print("Q:", q)
    print("A:", get_answer(q))
    print("-"*40)
```  
"""
}





# 用 update 合并追加
QA_DICT.update(addition_dict)

# 如果想排序（按key排序）
QA_DICT_sorted = dict(sorted(QA_DICT.items(), key=lambda item: item[0]))

# --------------------------------------------------
# 加载外部问答（仅本地 JSON）
# --------------------------------------------------
def load_external_qa():
    qa = {}
    if not os.path.isfile(LOCAL_QA_PATH):
        return qa

    try:
        f = open(LOCAL_QA_PATH, "r", encoding="utf-8")
        data = json.load(f)
        f.close()
    except Exception:
        return qa

    # 只保留键和值都是字符串的条目
    for key in data:
        value = data[key]
        if isinstance(key, str) and isinstance(value, str):
            qa[key] = value
    return qa
# --------------------------------------------------
# 计算最长公共子序列长度
# --------------------------------------------------
def lcs_length(s1, s2):
    n = len(s1)
    m = len(s2)
    # 初始化 dp 数组
    dp = []
    for i in range(n+1):
        row = []
        for j in range(m+1):
            row.append(0)
        dp.append(row)

    # 填表
    for i in range(1, n+1):
        for j in range(1, m+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                if dp[i-1][j] >= dp[i][j-1]:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = dp[i][j-1]
    return dp[n][m]
# --------------------------------------------------
# 计算编辑距离（Levenshtein Distance）
# --------------------------------------------------
def edit_distance(s1, s2):
    n = len(s1)
    m = len(s2)
    # 初始化 dp 数组
    dp = []
    for i in range(n+1):
        row = []
        for j in range(m+1):
            row.append(0)
        dp.append(row)

    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j

    # 填表
    for i in range(1, n+1):
        for j in range(1, m+1):
            if s1[i-1] == s2[j-1]:
                cost = 0
            else:
                cost = 1

            delete = dp[i-1][j] + 1
            insert = dp[i][j-1] + 1
            replace = dp[i-1][j-1] + cost

            # 取三者最小值
            v = delete
            if insert < v:
                v = insert
            if replace < v:
                v = replace
            dp[i][j] = v

    return dp[n][m]

# --------------------------------------------------
# 计算匹配分数
# --------------------------------------------------
def compute_score(query, template):
    max_len = max(len(query), len(template), 1)
    lcs_len = lcs_length(query, template)
    ed = edit_distance(query, template)

    score_lcs = lcs_len / max_len
    score_edit = 1 - ed / max_len

    return W_LCS * score_lcs + W_EDIT * score_edit

# --------------------------------------------------
# 找到 Top-K 回答
# --------------------------------------------------
def find_top_k_answers(query, k):
    scored = []
    # 计算每个模板的分数
    for tpl in QA_DICT:
        sc = compute_score(query, tpl)
        if sc > 0:
            scored.append((sc, QA_DICT[tpl]))

    # 按分数降序排序
    for i in range(len(scored)):
        for j in range(i+1, len(scored)):
            if scored[j][0] > scored[i][0]:
                temp = scored[i]
                scored[i] = scored[j]
                scored[j] = temp

    # 取前 k 条
    topk = []
    cnt = 0
    while cnt < k and cnt < len(scored):
        topk.append(scored[cnt])
        cnt += 1

    return topk
# --------------------------------------------------
# 组织回复
def compose_reply(query, k):
    topk = find_top_k_answers(query, k)
    if len(topk) == 0:
        return "抱歉，我不太明白您的意思。"

    # 用空行分隔多条回答
    reply = ""
    for index in range(len(topk)):
        item = topk[index]
        ans = item[1]
        if index > 0:
            reply += "\n\n"
        reply += ans
    return reply
# --------------------------------------------------
# 主程序
print("输入“退出”结束对话。")
top_k = 1   # 这里预定义top_k的值，不接受用户输入
while True:
    user_input = input("用户: ").strip()
    if user_input in ("退出", "再见", "bye"):
        print("助手: 再见！祝您生活愉快！")
        break

    query = user_input  # 把用户输入赋给query变量

    reply = compose_reply(query, top_k)
    print("助手:\n" + reply)







# -*- coding: utf-8 -*-
import json
import os
from flask import Flask, request, jsonify, render_template_string

# --------------------------------------------------
# 配置区
# --------------------------------------------------
LOCAL_QA_PATH = "external_qa.json"
W_LCS = 0.8
W_EDIT = 0.2

# --------------------------------------------------
# 内置问答字典
# --------------------------------------------------
QA_DICT = {
    "你好": "你好！有什么可以帮您的吗？",
    "你叫什么名字": "我是智能客服小助手。",
    "怎么写一个 for 循环": (
        "下面给你一个简单的 Python for 循环示例：\n\n"
        "for i in range(5):\n"
        "    # 打印当前 i 的值\n"
        "    print(f\"当前 i = {i}\")\n\n"
        "# 循环结束后的空行也会保留\n\n"
        "print(\"循环结束\")"
    ),
    "谢谢": "不客气，很高兴为您服务！"
}

# --------------------------------------------------
# 加载外部问答（仅本地 JSON）
# --------------------------------------------------
def load_external_qa():
    qa = {}
    if not os.path.isfile(LOCAL_QA_PATH):
        return qa
    try:
        with open(LOCAL_QA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return qa
    for key, value in data.items():
        if isinstance(key, str) and isinstance(value, str):
            qa[key] = value
    return qa

# --------------------------------------------------
# LCS & 编辑距离函数
# --------------------------------------------------
def lcs_length(s1, s2):
    n, m = len(s1), len(s2)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(1, m+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[n][m]

def edit_distance(s1, s2):
    n, m = len(s1), len(s2)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )
    return dp[n][m]

# --------------------------------------------------
# 计算分数 & 找 Top-K 回答
# --------------------------------------------------
def compute_score(query, template):
    max_len = max(len(query), len(template), 1)
    lcs_len = lcs_length(query, template)
    ed = edit_distance(query, template)
    score_lcs = lcs_len / max_len
    score_edit = 1 - ed / max_len
    return W_LCS * score_lcs + W_EDIT * score_edit

def find_top_k_answers(query, k=2):
    scored = []
    for tpl, ans in QA_DICT.items():
        sc = compute_score(query, tpl)
        if sc > 0:
            scored.append((sc, ans))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ans for _, ans in scored[:k]]

def compose_reply(query, k=2):
    answers = find_top_k_answers(query, k)
    if not answers:
        return "抱歉，我不太明白您的意思。"
    return "\n\n".join(answers)

# --------------------------------------------------
# Flask 应用
# --------------------------------------------------
app = Flask(__name__)

# 加载并合并本地外部问答（如果有）
external = load_external_qa()
if external:
    QA_DICT.update(external)
    print(f"[系统] 已加载并合并 {len(external)} 条本地外部问答。")

# HTML 模板
HTML_PAGE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>智能客服小助手</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body { background: #f0f2f5; }
    .chat-container {
      max-width: 720px;
      margin: 40px auto;
      background: #fff;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      padding: 20px;
    }
    #chat-window {
      height: 400px;
      overflow-y: auto;
      padding: 10px;
      border: 1px solid #ddd;
      border-radius: 5px;
      background: #fafafa;
    }
    .message { margin-bottom: 12px; display: flex; }
    .message.user   { justify-content: flex-end; }
    .message.bot    { justify-content: flex-start; }
    .bubble {
      max-width: 75%;
      padding: 10px 14px;
      border-radius: 18px;
      white-space: pre-wrap;
      word-wrap: break-word;
    }
    .bubble.user { background: #0d6efd; color: #fff; }
    .bubble.bot  { background: #e9ecef; color: #212529; }
    .input-group { margin-top: 16px; }
  </style>
</head>
<body>
<div class="chat-container">
  <h4 class="mb-4 text-center">智能客服小助手</h4>
  <div id="chat-window"></div>
  <div class="input-group">
    <input type="text" id="user-input" class="form-control" placeholder="请输入内容…">
    <button id="send-btn" class="btn btn-primary">发送</button>
  </div>
</div>
<script>
  const chatWindow = document.getElementById("chat-window");
  const userInput  = document.getElementById("user-input");
  const sendBtn    = document.getElementById("send-btn");

  function appendMessage(text, who) {
    const wrapper = document.createElement("div");
    wrapper.className = "message " + who;
    const bubble = document.createElement("div");
    bubble.className = "bubble " + who;
    bubble.textContent = text;
    wrapper.appendChild(bubble);
    chatWindow.appendChild(wrapper);
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  async function sendMessage() {
    const msg = userInput.value.trim();
    if (!msg) return;
    appendMessage(msg, "user");
    userInput.value = "";
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg })
      });
      const data = await res.json();
      appendMessage(data.reply, "bot");
      if (data.end) {
        sendBtn.disabled = true;
        userInput.disabled = true;
      }
    } catch (err) {
      appendMessage("网络错误，请稍后再试。", "bot");
    }
  }

  sendBtn.addEventListener("click", sendMessage);
  userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendMessage();
  });
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True)
    user_msg = data.get("message", "").strip()
    if not user_msg:
        return jsonify({"reply": "消息为空，请重新输入。", "end": False}), 400
    if user_msg in ("退出", "再见", "bye"):
        return jsonify({"reply": "再见！祝您生活愉快！", "end": True})
    reply = compose_reply(user_msg, k=2)
    return jsonify({"reply": reply, "end": False})

if __name__ == "__main__":
    # host="0.0.0.0" 可根据需要开放局域网访问
    app.run(debug=True, port=5000)




