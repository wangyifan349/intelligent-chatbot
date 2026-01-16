from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = Flask(__name__)
CORS(app)

# 直接一维list："问题","答案","问题","答案", ...
qa_list = [
    "什么是Python", "Python是一种解释型、面向对象的高级编程语言。",
    "心脏病的症状有哪些", "心脏病的症状包括胸痛、气短、心悸等。",
    "什么是合同法", "合同法是调整平等主体的自然人、法人、其他组织之间权利义务关系的法律规范。",
    "AES加密算法的示例代码",
    '''
<pre><code>from Crypto.Cipher import AES
cipher = AES.new(key, AES.MODE_CBC, iv)
ciphertext = cipher.encrypt(plaintext)
</code></pre>''',
    "如何使用RSA算法进行加密",
    '''
<pre><code>from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
key = RSA.generate(2048)
cipher = PKCS1_OAEP.new(key)
ciphertext = cipher.encrypt(message)
</code></pre>''',
    "MD5算法的作用是什么", "MD5是一种广泛使用的密码散列函数，可以产生一个128位的散列值，用于保证信息传输完整性。",
    "Python中如何进行文件的SHA-256哈希计算",
    '''
<pre><code>import hashlib
def sha256_file(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()
</code></pre>'''
]

questions = qa_list[::2]
answers = qa_list[1::2]

# 加载模型、构建索引
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
model = SentenceTransformer(MODEL_NAME)
question_emb = model.encode(questions, normalize_embeddings=True)
dim = question_emb.shape[1]
faiss_index = faiss.IndexFlatIP(dim)
faiss_index.add(np.array(question_emb))

def find_best_answer(user_input):
    user_emb = model.encode([user_input], normalize_embeddings=True)
    score, idx = faiss_index.search(user_emb, k=1)
    if score[0][0] < 0.4:
        return "抱歉，我没有理解您的意思。"
    else:
        return answers[int(idx[0][0])]

@app.route('/')
def index():
    return render_template("chat.html")

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['message']
    res = find_best_answer(user_input)
    return jsonify({'user': user_input, 'bot': res})

if __name__ == '__main__':
    app.run('0.0.0.0', 8080)


---

## 2. templates/chat.html


<!DOCTYPE html>
<html lang="zh-cn">
<head>
    <meta charset="UTF-8">
    <title>Q&A 智能问答机器人</title>
    <style>
        body {
            background: #fff4f7;
            font-family: "微软雅黑", "Helvetica Neue", Helvetica, Arial, sans-serif;
        }
        #chat-box {
            width: 500px;
            margin: 36px auto;
            background: #fff0f3;
            border-radius: 18px;
            box-shadow: 0 6px 20px #fab3be60, 0 1.5px 7px #fab3be10;
            padding: 22px 28px 16px 28px;
        }
        h2 {
            color: #d04861;
            text-align: center;
            font-weight: bold;
            letter-spacing: 2px;
            margin-top: 0;
        }
        #messages {
            height: 340px;
            overflow-y: auto;
            border: 1.8px solid #ffdbe0;
            border-radius: 6px;
            background: #fff;
            padding: 11px 9px 8px 16px;
            margin-bottom: 13px;
            font-size: 16px;
            box-shadow: 0 0 7px #ffe0e380 inset;
        }
        .msg { margin: 10px 0 10px 0; line-height: 1.8;}
        .user {
            color: #c63767;
            font-weight: bold;
        }
        .bot {
            color: #d5536a;
            background: #fbe2e8;
            border-radius: 8px;
            display: inline-block;
            padding: 8px 15px 8px 13px;
            margin-top: 2px;
            margin-bottom: 4px;
            box-shadow: 0 2px 4px #ffcccc30;
        }
        #input-area {
            display: flex;
            gap: 9px;
        }
        #msg-input {
            flex: 1;
            padding: 8px 15px;
            border-radius: 20px;
            border: 1.2px solid #fab3be80;
            font-size: 16px;
            background: #fff5f8;
            outline: none;
            transition: border 0.2s;
        }
        #msg-input:focus { border-color: #ea7b95; }
        #send-btn {
            padding: 8px 22px;
            border-radius: 18px;
            border: none;
            background: linear-gradient(90deg, #fe7fa5 10%, #fab3be 100%);
            color: #fff;
            font-weight: bold;
            font-size: 16px;
            box-shadow: 0 2px 6px #ff668020;
            cursor: pointer;
            transition: background 0.2s, box-shadow 0.2s;
        }
        #send-btn:hover {
            background: linear-gradient(90deg, #fd5d9c 0%, #fa6699 100%);
            box-shadow: 0 1px 16px #fe93b830;
        }
        .user-tip {
            color: #ad7f8a;
            font-size: 13px;
            margin-bottom: 6px;
            text-align: right;
            margin-right: 3px;
        }
        pre, code {
            background: #FFF4F7;
            border-radius: 5px;
            color: #384350;
            font-size: 15px;
            font-family: Consolas, "Fira Mono", monospace, "Courier New";
            margin: 1px 0 3px 0;
        }
        pre {
            display: block;
            padding: 10px 15px;
            white-space: pre;
            border-left: 4px solid #fd9aaf70;
            overflow-x: auto;
        }
        code {
            padding: 2px 6px;
        }
        @media (max-width: 650px) {
            #chat-box { width: 98vw; padding: 2vw 1vw;}
            #messages { font-size:15px;}
        }
    </style>
</head>
<body>
    <div id="chat-box">
        <h2>Q&A 智能问答机器人</h2>
        <div class="user-tip">支持中英文及部分代码问答，输入后回车或点击发送即可</div>
        <div id="messages"></div>
        <div id="input-area">
            <input type="text" id="msg-input" placeholder="请输入您的问题..." autocomplete="off" autofocus />
            <button id="send-btn">发送</button>
        </div>
    </div>
    <script>
        let messagesDiv = document.getElementById('messages');
        function escapeHtml(text) {
            var div = document.createElement("div");
            div.innerText = text;
            return div.innerHTML;
        }
        function sendMsg() {
            const input = document.getElementById('msg-input');
            let msg = input.value.trim();
            if (!msg) return;
            let userMsg = document.createElement("div");
            userMsg.className = "msg user";
            userMsg.innerHTML = "用户: " + escapeHtml(msg);
            messagesDiv.appendChild(userMsg);
            input.value = '';
            fetch('/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                body: 'message=' + encodeURIComponent(msg)
            }).then(resp => resp.json()).then(data => {
                let botMsg = document.createElement("div");
                botMsg.className = "msg bot";
                let botHtml = data.bot || '';
                botMsg.innerHTML = botHtml;
                messagesDiv.appendChild(botMsg);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            });
        }
        document.getElementById('send-btn').onclick = sendMsg;
        document.getElementById('msg-input').onkeyup = function(e) {
            if (e.key === 'Enter') sendMsg();
        };
    </script>
</body>
</html>

