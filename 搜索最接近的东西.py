import os
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ─── 配置区 ────────────────────────────────────────────────────────────────────

# 模型名称（Hugging Face 预训练模型）
TEXT_MODEL_NAME = "all-MiniLM-L6-v2"
CODE_MODEL_NAME = "microsoft/codebert-base"

# 当前工作目录下的 models 子目录
BASE_MODEL_DIR = "./models"
LOCAL_TEXT_MODEL_PATH = os.path.join(BASE_MODEL_DIR, "all-MiniLM-L6-v2")
LOCAL_CODE_MODEL_PATH = os.path.join(BASE_MODEL_DIR, "codebert-base")

# ─── 语料库 ────────────────────────────────────────────────────────────────────

corpus = [
    "The quick brown fox jumps over the lazy dog.",
    '''def fib(n):
    """
    Compute the nth Fibonacci number
    """
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
''',
    '''class Person:
    def __init__(self, name):
        self.name = name

    def greet(self):
        print(f"Hello, {self.name}")''',
    "I love watching sci-fi movies."
]

# ─── 工具函数 ──────────────────────────────────────────────────────────────────

def download_and_save_model(model_name: str, local_path: str) -> SentenceTransformer:
    """
    如果本地不存在模型文件夹，则下载保存；否则直接加载本地模型。
    """
    if not os.path.exists(local_path):
        print(f"Downloading model '{model_name}' …")
        model = SentenceTransformer(model_name)
        os.makedirs(local_path, exist_ok=True)
        model.save(local_path)
        print(f"Saved to '{local_path}'")
    else:
        print(f"Loading model from '{local_path}' …")
        model = SentenceTransformer(local_path)
    return model

def is_code(text: str) -> bool:
    """
    简单启发式判定：若包含典型代码结构，则视为代码。
    """
    code_markers = [
        r"\bdef\s+\w+\(",    # Python 函数定义
        r"\bclass\s+\w+",    # Python 类定义
        r"{.*}",             # 大括号
        r";\s*$",            # 行尾分号
        r"=\s*\w+\(",        # 赋值调用
        r"#"                 # 注释
    ]
    pattern = "|".join(code_markers)
    return re.search(pattern, text, re.MULTILINE) is not None

def encode_and_normalize(
    model: SentenceTransformer,
    texts: list,
    batch_size: int = 16
) -> np.ndarray:
    """
    批量编码并做 L2 归一化，返回 numpy 数组。
    """
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True
    )

def build_faiss_index(embs: np.ndarray) -> faiss.IndexFlatIP:
    """
    基于内积构建 FAISS 索引（内积等价于已归一化向量的余弦相似度）。
    """
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    return index

# ─── 主流程 ────────────────────────────────────────────────────────────────────

# 确保 models 根目录存在
os.makedirs(BASE_MODEL_DIR, exist_ok=True)

# 1) 下载或加载本地模型
text_model = download_and_save_model(TEXT_MODEL_NAME, LOCAL_TEXT_MODEL_PATH)
code_model = download_and_save_model(CODE_MODEL_NAME, LOCAL_CODE_MODEL_PATH)

# 2) 按类型拆分语料并记录原始索引
text_corpus, text_idx_map = [], []
code_corpus, code_idx_map = [], []
for idx, entry in enumerate(corpus):
    if is_code(entry):
        code_corpus.append(entry)
        code_idx_map.append(idx)
    else:
        text_corpus.append(entry)
        text_idx_map.append(idx)

# 3) 编码并归一化
text_embeddings = encode_and_normalize(text_model, text_corpus)
code_embeddings = encode_and_normalize(code_model, code_corpus)

# 4) 构建 FAISS 索引
text_index = build_faiss_index(text_embeddings)
code_index = build_faiss_index(code_embeddings)

# ─── 交互式查询循环 ────────────────────────────────────────────────────────────

print("Enter your query (code or text). Type 'exit' or 'quit' to stop.")

while True:
    # 读取多行输入直到空行
    print("\nEnter your query (finish with an empty line):")
    lines = []
    while True:
        line = input()
        if not line.strip():
            break
        lines.append(line)
    query = "\n".join(lines).strip()

    # 退出判断
    if query.lower() in ("exit", "quit"):
        print("Bye!")
        break

    # 5) 类型判定 & 模型、索引选择
    if is_code(query):
        model = code_model
        index = code_index
        idx_map = code_idx_map
        print("→ Detected code snippet; using code model.")
    else:
        model = text_model
        index = text_index
        idx_map = text_idx_map
        print("→ Detected natural language; using text model.")

    # 6) 编码查询并检索
    q_emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    distances, indices = index.search(q_emb, 1)
    top_sub = int(indices[0][0])
    score = float(distances[0][0])

    # 7) 映射回原始语料
    orig_idx = idx_map[top_sub]
    best = corpus[orig_idx]

    # 8) 打印结果
    print(f"\nBest match (score = {score:.4f}):\n{best}")
