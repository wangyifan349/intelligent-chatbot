# qa_whole_ktrapeznikov_fixed.py
# requirements:
# pip install transformers torch numpy
"""
BERT 是一种基于双向 Transformer 编码器的语言表示模型：它通过在大规模文本上联合训练两种自监督任务——掩码语言建模（Masked Language Modeling，随机遮蔽输入中的若干词并让模型预测被遮蔽的词）和下一句预测（Next Sentence Prediction，用以学习句子间关系）——来捕捉上下文信息。
与传统单向或浅层上下文表示不同，BERT 的每一层自注意力机制允许每个 token 同时看见左右两侧的上下文，从而得到深层且上下文敏感的词向量；
在下游任务（如问答）中，BERT 在预训练后通过在任务特定数据上微调，将输入的 question 与 context 拼接输入模型，利用模型输出的 start/end logits 来定位答案的起止 token，整个过程保留了预训练学到的语言知识并借微调适配具体任务。
"""
from typing import List, Dict, Any, Tuple
import torch
import numpy as np
from transformers import BertTokenizerFast, BertForQuestionAnswering
# -------------------- 配置 --------------------
MODEL_NAME = "ktrapeznikov/bert-base-uncased-whole-word-masking-squad2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tokenization / sliding window / answer constraints
MAX_SEQ_LENGTH = 384
DOC_STRIDE = 128
MAX_ANSWER_LENGTH = 30
N_BEST_SIZE = 50   # 内部候选数量（取 top N_BEST_SIZE from logits），最后按 top_k 返回
ALLOW_NO_ANSWER = True  # 是否允许返回空答案（SQuAD2 style）
NO_ANSWER_BIAS = 0.0    # 若需调节空答案倾向，改变此值（加到 cls score）
# -------------------- 加载模型 & tokenizer --------------------
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
model = BertForQuestionAnswering.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()
# -------------------- 工具函数 --------------------
def prepare_features(question: str, context: str):
    """
    对 question+context 做滑动窗口编码，返回 encodings（包含 tensors 和 offset mappings）
    """
    enc = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=MAX_SEQ_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt",
    )
    return enc
def compute_softmax(scores: List[float]) -> List[float]:
    """对一组分数做 softmax，数值稳定实现"""
    if not scores:
        return []
    a = np.array(scores, dtype=np.float64)
    a -= np.max(a)
    exp = np.exp(a)
    return (exp / exp.sum()).tolist()
def get_top_indices(logits: np.ndarray, top_k: int) -> List[int]:
    """返回 logits 的 top_k 索引（从大到小），处理 top_k > len(logits) 的情况"""
    length = logits.shape[0]
    k = min(top_k, length)
    return np.argsort(logits)[-k:][::-1].tolist()
def get_candidates_from_feature(
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    offsets: List[Tuple[Any, Any]],
    token_type_ids: List[int],
    context_text: str,
    feature_index: int,
) -> List[Dict[str, Any]]:
    """
    从一个 feature（分片）中生成候选(answer spans)。
    offsets: list of (start_char, end_char) per token (may contain None)
    token_type_ids: list indicating token belongs to which sequence (0=question,1=context)
    返回 list of dict with keys: text, start_token, end_token, start_char, end_char, score
    """
    candidates: List[Dict[str, Any]] = []

    # Ensure numpy arrays
    start_logits = np.asarray(start_logits)
    end_logits = np.asarray(end_logits)

    seq_len = start_logits.shape[0]
    # context token indices
    context_token_indices = [i for i, t in enumerate(token_type_ids) if t == 1]

    if not context_token_indices:
        return candidates

    # pick top indices
    start_top_inds = get_top_indices(start_logits, N_BEST_SIZE)
    end_top_inds = get_top_indices(end_logits, N_BEST_SIZE)
    for si in start_top_inds:
        if si < 0 or si >= seq_len:
            continue
        if si not in context_token_indices:
            continue
        for ei in end_top_inds:
            if ei < 0 or ei >= seq_len:
                continue
            if ei not in context_token_indices:
                continue
            if ei < si:
                continue
            length = ei - si + 1
            if length > MAX_ANSWER_LENGTH:
                continue
            s_char, e_char = offsets[si] if si < len(offsets) else (None, None)
            # offsets may be None or integers; ensure ints
            if s_char is None or e_char is None:
                continue
            if not isinstance(s_char, int) or not isinstance(e_char, int):
                continue
            if s_char >= e_char:
                continue
            # Clip to context text length
            if s_char < 0 or e_char > len(context_text):
                continue
            text = context_text[s_char:e_char]
            score = float(start_logits[si] + end_logits[ei])
            candidates.append({
                "text": text,
                "start_token": int(si),
                "end_token": int(ei),
                "start_char": int(s_char),
                "end_char": int(e_char),
                "score": score,
                "feature_index": int(feature_index),
            })
    return candidates
def postprocess(question: str, context: str, features, start_logits_all: np.ndarray, end_logits_all: np.ndarray, top_k: int = 3):
    """
    将所有 feature 的 logits 聚合，返回 top_k 答案（按概率排序）。
    features: tokenizer encoding (a BatchEncoding with tensors)
    start_logits_all / end_logits_all: numpy arrays shape (n_feats, seq_len)
    返回 list of candidate dicts (text, start_token, end_token, start_char, end_char, score, probability).
    """
    all_candidates: List[Dict[str, Any]] = []

    # Extract arrays
    offset_mappings = features["offset_mapping"].tolist()  # list of list of (start,end) or None
    token_type_ids_batch = features["token_type_ids"].tolist()
    input_ids = features["input_ids"].tolist()

    n_features = len(offset_mappings)
    # Add CLS/no-answer candidate once (aggregate best CLS across features)
    if ALLOW_NO_ANSWER:
        cls_scores = []
        for idx in range(n_features):
            s_logits = start_logits_all[idx]
            e_logits = end_logits_all[idx]
            # CLS is token 0
            if s_logits.shape[0] > 0 and e_logits.shape[0] > 0:
                cls_scores.append(float(s_logits[0] + e_logits[0]))
        if cls_scores:
            best_cls = max(cls_scores) + NO_ANSWER_BIAS
            all_candidates.append({
                "text": "",
                "start_token": -1,
                "end_token": -1,
                "start_char": -1,
                "end_char": -1,
                "score": float(best_cls),
                "feature_index": -1,
            })
    for idx in range(n_features):
        offsets = offset_mappings[idx]
        token_type_ids = token_type_ids_batch[idx]
        s_logits = start_logits_all[idx]
        e_logits = end_logits_all[idx]

        # Normalize offsets to list of tuples (None or ints)
        norm_offsets: List[Tuple[Any, Any]] = []
        for o in offsets:
            if o is None:
                norm_offsets.append((None, None))
            else:
                # o may be a list/tuple of ints
                start_o, end_o = o
                if isinstance(start_o, int) and isinstance(end_o, int):
                    norm_offsets.append((int(start_o), int(end_o)))
                else:
                    norm_offsets.append((None, None))
        feats_candidates = get_candidates_from_feature(
            start_logits=s_logits,
            end_logits=e_logits,
            offsets=norm_offsets,
            token_type_ids=token_type_ids,
            context_text=context,
            feature_index=idx
        )
        all_candidates.extend(feats_candidates)
    if not all_candidates:
        return []
    # deduplicate by (start_char,end_char,text) keep highest score
    unique_map: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
    for c in all_candidates:
        key = (c["start_char"], c["end_char"], c["text"])
        if key not in unique_map or c["score"] > unique_map[key]["score"]:
            unique_map[key] = c
    unique_candidates = list(unique_map.values())
    # sort by score desc
    unique_candidates.sort(key=lambda x: x["score"], reverse=True)
    # compute probabilities via softmax over top M candidates (to keep numeric stable)
    # choose M = min(len(unique_candidates), 50)
    M = min(len(unique_candidates), 50)
    top_for_softmax = unique_candidates[:M]
    scores = [c["score"] for c in top_for_softmax]
    probs = compute_softmax(scores)
    for c, p in zip(top_for_softmax, probs):
        c["probability"] = float(p)
    # for the rest, set probability 0
    for c in unique_candidates[M:]:
        c["probability"] = 0.0
    # return top_k (ensure not to exceed available)
    return unique_candidates[:min(top_k, len(unique_candidates))]
# -------------------- 主调用函数 --------------------
def answer_question(question: str, context: str, top_k: int = 3):
    """
    返回 top_k 候选答案列表（每项包含 text, start_token, end_token, start_char, end_char, score, probability）
    """
    features = prepare_features(question, context)
    input_ids = features["input_ids"].to(DEVICE)
    attention_mask = features["attention_mask"].to(DEVICE)
    token_type_ids = features["token_type_ids"].to(DEVICE)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        start_logits = outputs.start_logits.cpu().numpy()  # shape (n_feats, seq_len)
        end_logits = outputs.end_logits.cpu().numpy()
    results = postprocess(question, context, features, start_logits, end_logits, top_k=top_k)
    return results
# -------------------- 示例运行 --------------------
if __name__ == "__main__":
    context = (
        "The Apollo program was the third United States human spaceflight program carried out by NASA, "
        "which accomplished landing the first humans on the Moon from 1969 to 1972. Five other Apollo "
        "missions also landed astronauts on the Moon, the last in December 1972."
    )
    question = "When did Apollo missions land the first humans on the Moon?"
    top_k = 3
    answers = answer_question(question, context, top_k=top_k)
    for i, a in enumerate(answers, 1):
        print(f"Answer {i}:")
        print(f"  text: {a['text']!r}")
        print(f"  start_token: {a['start_token']}, end_token: {a['end_token']}")
        print(f"  start_char: {a['start_char']}, end_char: {a['end_char']}")
        print(f"  score: {a['score']:.4f}, probability: {a.get('probability', 0.0):.4f}")
