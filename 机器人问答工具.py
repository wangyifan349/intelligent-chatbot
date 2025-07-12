# -*- coding: utf-8 -*-
import json
import os

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
    "怎么写一个 for 循环": """下面给你一个简单的 Python for 循环示例：

for i in range(5):
    # 打印当前 i 的值
    print(f"当前 i = {i}")

# 循环结束后的空行也会保留

print("循环结束")
""",
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
# --------------------------------------------------
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
# --------------------------------------------------
if __name__ == "__main__":
    # 1. 合并本地外部问答（外部覆盖同键内置）
    external = load_external_qa()
    for key in external:
        QA_DICT[key] = external[key]

    if len(external) > 0:
        print("[系统] 已加载并合并 {} 条本地外部问答。".format(len(external)))

    print("输入“退出”结束对话。")
    while True:
        user_input = input("用户: ").strip()
        if user_input in ("退出", "再见", "bye"):
            print("助手: 再见！祝您生活愉快！")
            break

        reply = compose_reply(user_input, 2)
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




