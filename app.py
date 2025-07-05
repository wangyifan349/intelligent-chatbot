# app.py
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba

app = Flask(__name__)

# 预设的问答对
qa_pairs = [
    {'question': '什么是Python', 'answer': 'Python是一种解释型、面向对象的高级编程语言。'},
    {'question': '心脏病的症状有哪些', 'answer': '心脏病的症状包括胸痛、气短、心悸等。'},
    {'question': '什么是合同法', 'answer': '合同法是调整平等主体的自然人、法人、其他组织之间权利义务关系的法律规范。'},
    {'question': 'AES加密算法的示例代码', 'answer': '''以下是使用Python实现AES加密的示例代码：
```python
from Crypto.Cipher import AES
cipher = AES.new(key, AES.MODE_CBC, iv)
ciphertext = cipher.encrypt(plaintext)
```'''},

    {'question': '如何使用RSA算法进行加密', 'answer': '''使用RSA算法加密的Python示例：
```python
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
key = RSA.generate(2048)
cipher = PKCS1_OAEP.new(key)
ciphertext = cipher.encrypt(message)
```'''},

    {'question': 'MD5算法的作用是什么', 'answer': 'MD5是一种广泛使用的密码散列函数，可以产生一个128位的散列值，用于保证信息传输完整性。'},

    {'question': 'Python中如何进行文件的SHA-256哈希计算', 'answer': '''使用Python计算文件的SHA-256哈希值：
```python
import hashlib
def sha256_file(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()
```'''},
    # 可以继续添加更多的问答对...
]

# 构建用于TF-IDF的语料库（预设问题的集合）
corpus = []
for pair in qa_pairs:
    corpus.append(pair['question'])

# 定义自定义的分词器，使用jieba分词
def jieba_tokenizer(text):
    return jieba.lcut(text)

# 初始化TfidfVectorizer
vectorizer = TfidfVectorizer(tokenizer=jieba_tokenizer)
# 拟合语料库
tfidf_matrix = vectorizer.fit_transform(corpus)

def find_best_answer(user_input):
    # 将用户输入转换为TF-IDF向量
    user_tfidf = vectorizer.transform([user_input])
    # 计算余弦相似度
    similarities = cosine_similarity(user_tfidf, tfidf_matrix)
    # 获取最相似的问题的索引
    max_sim_index = 0
    max_sim_value = 0
    for idx, similarity in enumerate(similarities[0]):
        if similarity > max_sim_value:
            max_sim_value = similarity
            max_sim_index = idx
    # 如果相似度过低，表示无法理解用户输入
    if max_sim_value < 0.1:
        return "抱歉，我没有理解您的意思。"
    else:
        return qa_pairs[max_sim_index]['answer']

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['message']
    bot_response = find_best_answer(user_input)
    return jsonify({'user': user_input, 'bot': bot_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
