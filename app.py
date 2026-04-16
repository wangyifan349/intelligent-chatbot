#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, render_template_string      # Flask 基础组件
from flask_cors import CORS                                            # 允许跨域
from sentence_transformers import SentenceTransformer                   # 文本向量模型
import faiss                                                            # 向量检索库
import numpy as np                                                      # 数值计算
app = Flask(__name__)                                                   # 创建 Flask 应用
CORS(app)                                                               # 开启跨域支持
# ============================================================
# 1. 系统配置（参数集中，便于修改）
# ============================================================
MODEL_NAME = "BAAI/bge-m3"                                              # 更强的多语言向量模型
HOST = "0.0.0.0"                                                        # 监听地址
PORT = 8080                                                             # 监听端口
TOP_K = 3                                                               # 先召回前 K 条候选
SCORE_THRESHOLD = 0.45                                                  # 最低相似度阈值
NORMALIZE_EMBEDDINGS = True                                             # 是否归一化向量
SHOW_DEBUG_SCORE = True                                                 # 是否返回分数供调试查看
# ============================================================
# 2. 问答数据（问题、答案交替存放）
#    你后续也可以改成从数据库 / JSON / Excel 读取
# ============================================================
QA_LIST = [
    "什么是Python", "Python是一种解释型、面向对象的高级编程语言。",         # 基础问答
    "心脏病的症状有哪些", "心脏病的症状包括胸痛、气短、心悸等。",             # 医疗问答
    "什么是合同法", "合同法是调整平等主体之间权利义务关系的法律规范。",         # 法律问答

    "AES加密算法的示例代码",
    """<pre><code>from Crypto.Cipher import AES
cipher = AES.new(key, AES.MODE_CBC, iv)
ciphertext = cipher.encrypt(plaintext)</code></pre>""",                 # AES 示例代码

    "如何使用RSA算法进行加密",
    """<pre><code>from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
key = RSA.generate(2048)
cipher = PKCS1_OAEP.new(key)
ciphertext = cipher.encrypt(message)</code></pre>""",                   # RSA 示例代码

    "MD5算法的作用是什么", "MD5是一种常见的密码散列函数，可生成128位摘要值。",  # MD5 问答

    "Python中如何进行文件的SHA-256哈希计算",
    """<pre><code>import hashlib

def sha256_file(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()</code></pre>"""                           # SHA-256 示例代码
]
QUESTIONS = QA_LIST[::2]                                                # 取偶数位：问题列表
ANSWERS = QA_LIST[1::2]                                                 # 取奇数位：答案列表
# ============================================================
# 3. 加载模型并构建向量索引
# ============================================================
model = SentenceTransformer(MODEL_NAME)                                 # 加载语义向量模型
question_embeddings = model.encode(                                     # 将所有问题编码为向量
    QUESTIONS,
    normalize_embeddings=NORMALIZE_EMBEDDINGS
)
question_embeddings = np.array(question_embeddings, dtype="float32")    # 转成 float32 供 faiss 使用
embedding_dim = question_embeddings.shape[1]                            # 向量维度
faiss_index = faiss.IndexFlatIP(embedding_dim)                          # 归一化后用内积近似余弦相似度
faiss_index.add(question_embeddings)                                    # 把问题向量加入索引
# ============================================================
# 4. 检索函数
#    流程：
#    1）对用户输入编码
#    2）检索 top_k
#    3）选择最高分答案
# ============================================================
def search_answers(user_input, top_k=TOP_K):
    user_embedding = model.encode(                                      # 用户问题转向量
        [user_input],
        normalize_embeddings=NORMALIZE_EMBEDDINGS
    )
    user_embedding = np.array(user_embedding, dtype="float32")          # 转成 float32

    scores, indices = faiss_index.search(user_embedding, top_k)         # 检索 top_k 个最相似问题

    candidates = []                                                     # 存储候选结果
    for score, idx in zip(scores[0], indices[0]):                       # 遍历 top_k 检索结果
        if idx == -1:                                                   # faiss 无效索引保护
            continue
        candidates.append({
            "question": QUESTIONS[idx],                                 # 命中的问题
            "answer": ANSWERS[idx],                                     # 对应答案
            "score": float(score),                                      # 相似度分数
            "index": int(idx)                                           # 数据索引
        })

    return candidates                                                   # 返回候选结果列表


def find_best_answer(user_input, top_k=TOP_K, threshold=SCORE_THRESHOLD):
    candidates = search_answers(user_input, top_k=top_k)                # 先召回候选

    if not candidates:                                                  # 没有候选结果
        return {
            "answer": "抱歉，我暂时没有找到相关答案。",
            "score": 0.0,
            "matched_question": "",
            "candidates": []
        }

    best = candidates[0]                                                # 取第一条作为最佳结果

    if best["score"] < threshold:                                       # 分数低于阈值则拒答
        return {
            "answer": "抱歉，我没有理解您的意思。请换一种问法试试。",
            "score": best["score"],
            "matched_question": best["question"],
            "candidates": candidates
        }

    return {
        "answer": best["answer"],                                       # 返回最佳答案
        "score": best["score"],                                         # 返回最佳分数
        "matched_question": best["question"],                           # 返回匹配到的问题
        "candidates": candidates                                        # 返回候选列表，便于调试
    }


# ============================================================
# 5. 页面模板（单文件内联 HTML）
# ============================================================
INDEX_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">                                              <!-- 字符编码 -->
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  <!-- 移动端适配 -->
    <title>智能问答机器人</title>                                       <!-- 页面标题 -->
    <style>
        * {
            box-sizing: border-box;                                     /* 统一盒模型 */
        }

        body {
            margin: 0;                                                  /* 去掉默认外边距 */
            min-height: 100vh;                                          /* 高度铺满屏幕 */
            font-family: "Microsoft YaHei", "PingFang SC", Arial, sans-serif;  /* 中文字体 */
            background: linear-gradient(135deg, #f8fbff 0%, #eef5ff 100%);      /* 柔和背景 */
            display: flex;                                              /* 居中布局 */
            align-items: center;                                        /* 垂直居中 */
            justify-content: center;                                    /* 水平居中 */
            padding: 24px;                                              /* 页面留白 */
            color: #1f2937;                                             /* 主文字颜色 */
        }

        .app-container {
            width: 100%;                                                /* 宽度自适应 */
            max-width: 860px;                                           /* 最大宽度 */
            background: #ffffff;                                        /* 白底卡片 */
            border-radius: 20px;                                        /* 圆角 */
            box-shadow: 0 18px 60px rgba(31, 41, 55, 0.12);             /* 阴影 */
            overflow: hidden;                                           /* 裁剪内部溢出 */
            border: 1px solid rgba(148, 163, 184, 0.16);                /* 边框 */
        }

        .header {
            padding: 24px 28px;                                         /* 内边距 */
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); /* 头部渐变 */
            color: #ffffff;                                             /* 白字 */
        }

        .header h1 {
            margin: 0 0 8px 0;                                          /* 标题间距 */
            font-size: 26px;                                            /* 标题字号 */
            font-weight: 700;                                           /* 标题加粗 */
        }

        .header p {
            margin: 0;                                                  /* 去掉默认间距 */
            font-size: 14px;                                            /* 描述字号 */
            opacity: 0.92;                                              /* 略透明 */
            line-height: 1.7;                                           /* 行高 */
        }

        .content {
            display: flex;                                              /* 两栏布局 */
            min-height: 620px;                                          /* 最小高度 */
        }

        .left-panel {
            flex: 1;                                                    /* 主聊天区 */
            display: flex;                                              /* 纵向布局 */
            flex-direction: column;                                     /* 垂直排列 */
            background: #f8fbff;                                        /* 浅色背景 */
        }

        .right-panel {
            width: 280px;                                               /* 右侧信息栏宽度 */
            border-left: 1px solid #e5e7eb;                             /* 分隔线 */
            background: #ffffff;                                        /* 白底 */
            padding: 20px;                                              /* 内边距 */
        }

        .panel-title {
            font-size: 16px;                                            /* 标题字号 */
            font-weight: 700;                                           /* 标题加粗 */
            margin-bottom: 14px;                                        /* 下间距 */
            color: #111827;                                             /* 深色文字 */
        }

        .config-item {
            margin-bottom: 12px;                                        /* 配置项间距 */
            padding: 10px 12px;                                         /* 内边距 */
            background: #f9fafb;                                        /* 配置块背景 */
            border-radius: 10px;                                        /* 圆角 */
            border: 1px solid #eef2f7;                                  /* 细边框 */
        }

        .config-item .label {
            display: block;                                             /* 独占一行 */
            font-size: 12px;                                            /* 标签字号 */
            color: #6b7280;                                             /* 次级颜色 */
            margin-bottom: 4px;                                         /* 下间距 */
        }

        .config-item .value {
            font-size: 14px;                                            /* 值字号 */
            font-weight: 600;                                           /* 值加粗 */
            color: #111827;                                             /* 深色 */
            word-break: break-all;                                      /* 长内容换行 */
        }

        #messages {
            flex: 1;                                                    /* 占满剩余高度 */
            padding: 22px;                                              /* 内边距 */
            overflow-y: auto;                                           /* 纵向滚动 */
            background:
                radial-gradient(circle at top left, rgba(99, 102, 241, 0.08), transparent 32%),
                radial-gradient(circle at bottom right, rgba(124, 58, 237, 0.08), transparent 28%),
                #f8fbff;                                                /* 聊天背景 */
        }

        .message-row {
            display: flex;                                              /* 消息行布局 */
            margin-bottom: 16px;                                        /* 消息间距 */
        }

        .message-row.user {
            justify-content: flex-end;                                  /* 用户消息靠右 */
        }

        .message-row.bot {
            justify-content: flex-start;                                /* 机器人消息靠左 */
        }

        .message-bubble {
            max-width: 78%;                                             /* 气泡最大宽度 */
            padding: 12px 15px;                                         /* 内边距 */
            border-radius: 16px;                                        /* 圆角 */
            line-height: 1.75;                                          /* 行高 */
            font-size: 15px;                                            /* 字号 */
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);              /* 阴影 */
            word-break: break-word;                                     /* 自动换行 */
        }

        .message-row.user .message-bubble {
            background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%); /* 用户气泡渐变 */
            color: #ffffff;                                             /* 用户消息白字 */
            border-bottom-right-radius: 6px;                            /* 右下角变化 */
        }

        .message-row.bot .message-bubble {
            background: #ffffff;                                        /* 机器人气泡白底 */
            color: #1f2937;                                             /* 深色文字 */
            border: 1px solid #e5e7eb;                                  /* 边框 */
            border-bottom-left-radius: 6px;                             /* 左下角变化 */
        }

        .meta {
            margin-top: 8px;                                            /* 与正文间距 */
            font-size: 12px;                                            /* 元信息字号 */
            color: #6b7280;                                             /* 灰色 */
        }

        .input-area {
            padding: 18px 20px 20px 20px;                               /* 输入区域内边距 */
            background: #ffffff;                                        /* 输入区白底 */
            border-top: 1px solid #e5e7eb;                              /* 顶部分隔线 */
        }

        .input-box {
            display: flex;                                              /* 输入框与按钮横排 */
            gap: 12px;                                                  /* 间距 */
            align-items: center;                                        /* 垂直居中 */
        }

        #msg-input {
            flex: 1;                                                    /* 输入框自适应 */
            height: 52px;                                               /* 固定高度 */
            border: 1px solid #d1d5db;                                  /* 边框 */
            border-radius: 14px;                                        /* 圆角 */
            padding: 0 16px;                                            /* 左右内边距 */
            font-size: 15px;                                            /* 字号 */
            outline: none;                                              /* 去掉默认高亮 */
            transition: all 0.2s ease;                                  /* 动画过渡 */
            background: #f9fafb;                                        /* 输入框背景 */
        }

        #msg-input:focus {
            border-color: #6366f1;                                      /* 聚焦边框颜色 */
            background: #ffffff;                                        /* 聚焦背景 */
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.10);             /* 聚焦外阴影 */
        }

        #send-btn {
            min-width: 110px;                                           /* 按钮最小宽度 */
            height: 52px;                                               /* 按钮高度 */
            border: none;                                               /* 去边框 */
            border-radius: 14px;                                        /* 圆角 */
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); /* 按钮渐变 */
            color: #ffffff;                                             /* 白字 */
            font-size: 15px;                                            /* 字号 */
            font-weight: 700;                                           /* 加粗 */
            cursor: pointer;                                            /* 手型 */
            transition: transform 0.18s ease, box-shadow 0.18s ease;    /* 动画 */
            box-shadow: 0 12px 28px rgba(99, 102, 241, 0.24);           /* 阴影 */
        }

        #send-btn:hover {
            transform: translateY(-1px);                                /* 悬停上移 */
            box-shadow: 0 14px 30px rgba(99, 102, 241, 0.28);           /* 阴影增强 */
        }

        #send-btn:disabled {
            cursor: not-allowed;                                        /* 禁用态鼠标 */
            opacity: 0.7;                                               /* 禁用透明度 */
            transform: none;                                            /* 不上移 */
        }

        .status-bar {
            display: flex;                                              /* 横向布局 */
            justify-content: space-between;                             /* 左右分散 */
            align-items: center;                                        /* 垂直居中 */
            margin-top: 10px;                                           /* 与输入框间距 */
            font-size: 13px;                                            /* 字号 */
            color: #6b7280;                                             /* 灰色 */
        }

        .loading-dot {
            display: inline-block;                                      /* 行内块 */
            width: 6px;                                                 /* 小圆点大小 */
            height: 6px;                                                /* 小圆点大小 */
            margin-left: 4px;                                           /* 间距 */
            border-radius: 50%;                                         /* 圆形 */
            background: #6366f1;                                        /* 点颜色 */
            animation: blink 1.2s infinite ease-in-out;                 /* 呼吸动画 */
        }

        .loading-dot:nth-child(2) {
            animation-delay: 0.2s;                                      /* 第二个延迟 */
        }

        .loading-dot:nth-child(3) {
            animation-delay: 0.4s;                                      /* 第三个延迟 */
        }

        @keyframes blink {
            0%, 80%, 100% {
                transform: scale(0.6);                                  /* 初始缩小 */
                opacity: 0.45;                                          /* 初始透明 */
            }
            40% {
                transform: scale(1);                                    /* 中间放大 */
                opacity: 1;                                             /* 中间完全显示 */
            }
        }

        pre {
            margin: 10px 0 0 0;                                         /* 上边距 */
            padding: 12px 14px;                                         /* 内边距 */
            background: #0f172a;                                        /* 深色代码背景 */
            color: #e5e7eb;                                             /* 代码文字颜色 */
            border-radius: 10px;                                        /* 圆角 */
            overflow-x: auto;                                           /* 横向滚动 */
            font-size: 13px;                                            /* 字号 */
            line-height: 1.65;                                          /* 行高 */
        }

        code {
            font-family: Consolas, "Courier New", monospace;            /* 等宽字体 */
        }

        .hint-list {
            padding-left: 18px;                                         /* 列表缩进 */
            margin: 0;                                                  /* 去掉默认 margin */
            color: #4b5563;                                             /* 说明文字颜色 */
            line-height: 1.8;                                           /* 行高 */
            font-size: 13px;                                            /* 字号 */
        }

        @media (max-width: 900px) {
            .content {
                flex-direction: column;                                 /* 小屏改成上下布局 */
            }

            .right-panel {
                width: 100%;                                            /* 小屏右侧变全宽 */
                border-left: none;                                      /* 去左边框 */
                border-top: 1px solid #e5e7eb;                          /* 改成上边框 */
            }

            .message-bubble {
                max-width: 88%;                                         /* 小屏消息更宽 */
            }
        }
    </style>
</head>
<body>
    <div class="app-container">                                         <!-- 应用整体容器 -->
        <div class="header">                                            <!-- 顶部头部区域 -->
            <h1>智能问答机器人</h1>                                     <!-- 页面主标题 -->
            <p>支持中文问答、代码问答、多语言语义匹配，已加入 top_k 检索与 AJAX 交互。</p>  <!-- 页面描述 -->
        </div>

        <div class="content">                                           <!-- 主体内容区域 -->
            <div class="left-panel">                                    <!-- 左侧聊天区域 -->
                <div id="messages">                                     <!-- 消息容器 -->
                    <div class="message-row bot">                       <!-- 初始欢迎消息 -->
                        <div class="message-bubble">
                            你好，我是智能问答机器人。你可以直接输入问题，例如：<br>
                            1. 什么是 Python<br>
                            2. AES 加密算法的示例代码<br>
                            3. 如何计算文件 SHA-256 哈希
                            <div class="meta">系统消息</div>
                        </div>
                    </div>
                </div>

                <div class="input-area">                                <!-- 输入区域 -->
                    <div class="input-box">                             <!-- 输入框与按钮 -->
                        <input
                            type="text"
                            id="msg-input"
                            placeholder="请输入你的问题..."
                            autocomplete="off"
                        >
                        <button id="send-btn">发送</button>
                    </div>

                    <div class="status-bar">                            <!-- 状态栏 -->
                        <span id="status-text">准备就绪</span>
                        <span>AJAX / Top-K / 语义检索</span>
                    </div>
                </div>
            </div>

            <div class="right-panel">                                   <!-- 右侧参数说明区域 -->
                <div class="panel-title">当前配置</div>

                <div class="config-item">
                    <span class="label">模型</span>
                    <span class="value">{{ model_name }}</span>
                </div>

                <div class="config-item">
                    <span class="label">Top K</span>
                    <span class="value">{{ top_k }}</span>
                </div>

                <div class="config-item">
                    <span class="label">阈值</span>
                    <span class="value">{{ score_threshold }}</span>
                </div>

                <div class="config-item">
                    <span class="label">向量归一化</span>
                    <span class="value">{{ normalize_embeddings }}</span>
                </div>

                <div class="panel-title" style="margin-top: 22px;">使用说明</div>
                <ul class="hint-list">
                    <li>前端通过 AJAX 向 <code>/chat</code> 发送 POST 请求。</li>
                    <li>后端先召回 Top-K，再取最高分答案。</li>
                    <li>当最高分低于阈值时，返回兜底回复。</li>
                    <li>右侧参数区方便你后续调试模型效果。</li>
                </ul>
            </div>
        </div>
    </div>

    <script>
        const messagesDiv = document.getElementById("messages");        // 消息容器
        const inputEl = document.getElementById("msg-input");           // 输入框
        const sendBtn = document.getElementById("send-btn");            // 发送按钮
        const statusText = document.getElementById("status-text");      // 状态文本

        function escapeHtml(text) {                                     // 文本转义，防止注入
            const div = document.createElement("div");
            div.innerText = text;
            return div.innerHTML;
        }

        function appendUserMessage(text) {                              // 追加用户消息
            const row = document.createElement("div");
            row.className = "message-row user";

            row.innerHTML = `
                <div class="message-bubble">
                    ${escapeHtml(text)}
                    <div class="meta">你</div>
                </div>
            `;

            messagesDiv.appendChild(row);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function appendBotMessage(html, metaText = "机器人") {          // 追加机器人消息
            const row = document.createElement("div");
            row.className = "message-row bot";

            row.innerHTML = `
                <div class="message-bubble">
                    ${html}
                    <div class="meta">${escapeHtml(metaText)}</div>
                </div>
            `;

            messagesDiv.appendChild(row);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function appendLoadingMessage() {                               // 追加“思考中”消息
            const row = document.createElement("div");
            row.className = "message-row bot";
            row.id = "loading-row";

            row.innerHTML = `
                <div class="message-bubble">
                    正在思考中
                    <span class="loading-dot"></span>
                    <span class="loading-dot"></span>
                    <span class="loading-dot"></span>
                    <div class="meta">系统处理中</div>
                </div>
            `;

            messagesDiv.appendChild(row);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function removeLoadingMessage() {                               // 删除“思考中”消息
            const loadingRow = document.getElementById("loading-row");
            if (loadingRow) {
                loadingRow.remove();
            }
        }

        function setSendingState(isSending) {                           // 设置发送状态
            inputEl.disabled = isSending;
            sendBtn.disabled = isSending;
            statusText.textContent = isSending ? "正在请求后端..." : "准备就绪";
        }

        async function sendMessage() {                                  // 发送消息主函数
            const msg = inputEl.value.trim();

            if (!msg) {                                                 // 空输入直接返回
                inputEl.focus();
                return;
            }

            appendUserMessage(msg);                                     // 先显示用户消息
            inputEl.value = "";                                         // 清空输入框
            setSendingState(true);                                      // 切换发送状态
            appendLoadingMessage();                                     // 显示加载消息

            try {
                const response = await fetch("/chat", {                 // AJAX 请求后端
                    method: "POST",
                    headers: {
                        "Content-Type": "application/x-www-form-urlencoded"
                    },
                    body: "message=" + encodeURIComponent(msg)
                });

                const data = await response.json();                     // 解析 JSON

                removeLoadingMessage();                                 // 删除加载消息

                if (!response.ok) {                                     // HTTP 非 200 处理
                    appendBotMessage(
                        "请求失败：" + escapeHtml(data.error || "未知错误"),
                        "错误"
                    );
                    return;
                }

                let metaText = "机器人";                                // 默认元信息

                if (data.score !== undefined) {                         // 拼接分数信息
                    metaText = "匹配分数: " + Number(data.score).toFixed(4);
                }

                appendBotMessage(data.bot || "暂无回复", metaText);     // 显示机器人回复
            } catch (error) {
                removeLoadingMessage();                                 // 删除加载消息
                appendBotMessage(
                    "网络请求失败，请检查服务是否启动。",
                    "网络错误"
                );
            } finally {
                setSendingState(false);                                 // 恢复输入状态
                inputEl.focus();                                        // 光标回到输入框
            }
        }

        sendBtn.addEventListener("click", sendMessage);                 // 点击发送按钮
        inputEl.addEventListener("keydown", function (e) {              // 回车发送
            if (e.key === "Enter") {
                sendMessage();
            }
        });
    </script>
</body>
</html>
"""
# ============================================================
# 6. 路由
# ============================================================
@app.route("/", methods=["GET"])
def index():
    return render_template_string(                                      # 渲染内联页面模板
        INDEX_HTML,
        model_name=MODEL_NAME,
        top_k=TOP_K,
        score_threshold=SCORE_THRESHOLD,
        normalize_embeddings=NORMALIZE_EMBEDDINGS
    )
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form.get("message", "").strip()                # 获取前端输入
    if not user_input:                                                  # 空消息校验
        return jsonify({"error": "消息不能为空"}), 400
    result = find_best_answer(                                          # 进行语义检索
        user_input=user_input,
        top_k=TOP_K,
        threshold=SCORE_THRESHOLD
    )
    response_data = {
        "user": user_input,                                             # 原始用户输入
        "bot": result["answer"],                                        # 最终答案
        "matched_question": result["matched_question"],                 # 命中的问题
        "top_k": TOP_K                                                  # 当前 top_k 配置
    }
    if SHOW_DEBUG_SCORE:                                                # 是否返回调试分数
        response_data["score"] = result["score"]                        # 最佳匹配得分
        response_data["candidates"] = result["candidates"]              # 候选结果列表
    return jsonify(response_data)                                       # 返回 JSON 响应
# ============================================================
# 7. 启动入口
# ============================================================
if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=True)                           # 启动 Flask 服务
