from sentence_transformers import SentenceTransformer, util
import os

local_model_dir = 'local_sbert_model'

if os.path.exists(local_model_dir):
    print("从本地模型目录加载模型...")
    model = SentenceTransformer(local_model_dir)
else:
    print("本地没有模型，正在从huggingface下载...")
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    print("正在保存模型到本地...")
    model.save(local_model_dir)
    print("模型已保存。")

knowledge_base = {
    "怎样定义和调用带有默认参数和返回值的Python函数？": 
"""
# 定义一个有默认参数和返回值的函数
def greet(name, message="你好！"):
    '''
    向指定姓名打印问候语，如果未指定问候内容则使用默认值。
    '''
    return f"{name}：{message}"
# 示例使用
result1 = greet("小明")
result2 = greet("小红", "早上好！")
print(result1)    # 输出：小明：你好！
print(result2)    # 输出：小红：早上好！
""",

    "Python如何进行文件的读写和内容输出？":
"""
# 写入文件
with open('example.txt', 'w', encoding='utf-8') as file:
    lines = ["第一行内容", "第二行内容", "第三行内容"]
    for line in lines:
        file.write(line + '\\n')
# 读取文件并输出到终端
with open('example.txt', 'r', encoding='utf-8') as file:
    for line in file:
        print(line.strip())
# 说明：使用with能自动管理文件的关闭，读写时要指定正确的编码格式。
""",
    "Python如何遍历一个列表并判断元素类型？":
"""
my_list = [1, "hello", 3.5, [1, 2]]
for item in my_list:
    print(f"元素：{item}, 类型：{type(item)}")
# 结果会输出元素内容和对应的类型，适合初步理解Python类型动态特性。
""",
    "怎么用列表推导式和条件语句生成偶数平方列表？": """
# 生成1到20范围内所有偶数的平方，存在一个新列表里
even_squares = [x ** 2 for x in range(1, 21) if x % 2 == 0]
print(even_squares)
# 输出: [4, 16, 36, 64, 100, 144, 196, 256, 324, 400]
""",

    "如何用try...except结构安全处理用户输入并计算除法？":
"""
while True:
    try:
        a = float(input("请输入被除数a："))
        b = float(input("请输入除数b："))
        result = a / b
    except ValueError:
        print("请输入有效数字！")
        continue
    except ZeroDivisionError:
        print("除数不能为零！")
        continue
    else:
        print(f"结果为：{result}")
        break
# 结构包含多种异常判断，能友好应对非数字输入和除零错误。
""",

    "如何用Python统计字符串中每个字符出现的次数？":
"""
text = "hello world, this is python!"
frequency = {}
for char in text:
    if char in frequency:
        frequency[char] += 1
    else:
        frequency[char] = 1

for k, v in frequency.items():
    print(f"字符'{k}': 出现了{v}次")
# 此方式适用任意字符串统计，也可以换为collections.Counter简化。
""",

    "如何用Python实现一个简单的猜数字游戏？":
"""
import random

number = random.randint(1, 100)
print("欢迎玩猜数字游戏，请输入1到100之间的整数：")

for count in range(1, 11):  # 最多10次机会
    try:
        guess = int(input(f"第{count}次猜测："))
    except ValueError:
        print("请输入有效整数！")
        continue
    if guess == number:
        print(f"恭喜你，{count}次猜对了！答案就是{number}。")
        break
    elif guess < number:
        print("太小了，请再试试。")
    else:
        print("太大了，还可以继续猜。")
else:
    print(f"很遗憾，10次机会用完。正确答案是{number}。")
""",

    "怎么用字典统计一组学生成绩的平均值、最大值和最小值？":
"""
scores = {'张三': 85, '李四': 92, '王五': 78, '小明': 88}
score_values = scores.values()

average_score = sum(score_values) / len(score_values)
max_score = max(score_values)
min_score = min(score_values)

print(f"平均分：{average_score}")
print(f"最高分：{max_score}")
print(f"最低分：{min_score}")
# 字典结构非常适合用来管理和处理学生成绩等映射数据。
""",

    "如何用类(class)表示学生及其成绩，如何实例化和调用方法？": """
class Student:
    def __init__(self, name, score):
        self.name = name
        self.score = score

    def is_pass(self):
        if self.score >= 60:
            return True
        else:
            return False

    def show(self):
        status = "及格" if self.is_pass() else "不及格"
        print(f"学生: {self.name}, 分数: {self.score}, {status}")

# 实例化对象
stu = Student("小王", 76)
stu.show()
""",

    "怎么用datetime获取当前时间、日期、格式化以及计算时间差？": """
import datetime
now = datetime.datetime.now()
date_str = now.strftime("%Y-%m-%d")
time_str = now.strftime("%H:%M:%S")
print(f"当前日期：{date_str}")
print(f"当前时间：{time_str}")
# 计算两个时间之间的间隔(天数)
d1 = datetime.datetime(2023, 6, 1)
d2 = datetime.datetime(2024, 6, 1)
delta = d2 - d1
print(f"两个日期相差{delta.days}天")
""",
}
kb_questions = []
for q in knowledge_base:
    kb_questions.append(q)
kb_embeddings = model.encode(kb_questions, convert_to_tensor=True)
while True:
    user_question = input("请输入您的问题（输入exit退出）：")
    if user_question.strip().lower() == "exit":
        break
    question_embedding = model.encode(user_question, convert_to_tensor=True)
    cos_scores = util.cos_sim(question_embedding, kb_embeddings)[0]
    max_index = 0
    max_score = cos_scores[0].item()
    for i in range(1, len(cos_scores)):
        score = cos_scores[i].item()
        if score > max_score:
            max_score = score
            max_index = i
    best_question = kb_questions[max_index]
    answer = knowledge_base[best_question]
    print("最相关的答案：")
    print(answer)
