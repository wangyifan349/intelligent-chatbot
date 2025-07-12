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




