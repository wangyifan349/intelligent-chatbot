# 智能聊天机器人 🤖

这是一个基于 **Flask** 框架的智能聊天机器人，支持 **代码**、**医学**、**法律**、**自定义的任何领域** 等领域的知识查询。通过 **TF-IDF** 词袋模型和 **余弦相似度** 实现对用户问题的匹配，提供相应的答案。

## 仓库地址 📌

[https://github.com/wangyifan349/intelligent-chatbot](https://github.com/wangyifan349/intelligent-chatbot)

## 功能特点 ✨

- **多领域支持**：内置编程、医学、法律、密码学等多领域的知识问答，满足多种咨询需求。
- **代码展示**：支持代码片段的展示，方便技术类问题的学习和交流。
- **中文分词**：使用 **jieba** 进行中文分词，提升问题匹配的准确性。
- **美观界面**：前端采用 **Bootstrap** 进行美化，提供简洁清新的聊天界面。
- **一键运行**：前后端集成在一起，运行简单方便，可跨设备使用。

## 安装与运行 🚀

### 环境依赖 📦

- **Python 3.x**
- 必要的 Python 库：
  - Flask
  - scikit-learn
  - jieba

### 安装步骤 🛠️

1. **克隆项目代码**

   ```bash
   git clone https://github.com/wangyifan349/intelligent-chatbot.git
   cd intelligent-chatbot
   ```

2. **安装依赖库**

   ```bash
   pip install -r requirements.txt
   ```

3. **运行应用程序**

   ```bash
   python app.py
   ```

4. **访问聊天页面**

   在浏览器中打开 **[http://localhost:8080](http://localhost:8080)**，即可使用聊天机器人。

## 项目结构 📁

```
- app.py
- requirements.txt
- templates/
  - chat.html
```

- **app.py**：主应用程序，包含后端逻辑和预设的问答对。
- **templates/chat.html**：前端页面模板，构建聊天界面。
- **requirements.txt**：项目所需的 Python 库。

## 使用说明 💡

- 启动程序后，在浏览器中打开指定地址。
- 在输入框中输入您的问题，例如：“如何使用 RSA 算法进行加密”。
- 点击 **发送** 按钮，机器人会根据您的问题提供答案。
- 支持包含代码的回答，代码片段会以格式化的方式展示。

## 自定义与扩展 🔧

- **添加问答对**

  在 `app.py` 文件中的 `qa_pairs` 列表中，添加新的问答：

  ```python
  qa_pairs = [
      {'question': '你的问题', 'answer': '对应的回答'},
      # 其他问答对
  ]
  ```

- **调整相似度阈值**

  修改 `find_best_answer` 函数中的阈值：

  ```python
  if max_sim_value < 0.1:
      return "抱歉，我没有理解您的意思。"
  ```

  可根据需要提升或降低机器人回复的敏感度。

- **界面美化**

  前端页面位于 `templates/chat.html`，使用了 Bootstrap，可以根据需要自行修改样式和布局。

- **功能扩展**

  - **接入数据库**：保存和管理问答对，实现更复杂的数据管理。
  - **高级模型**：使用更先进的自然语言处理模型，提升匹配准确性。
  - **多轮对话**：添加上下文理解，实现更自然的多轮交互。

## 许可证 📄

本项目采用 **GNU 通用公共许可证第三版（GPL-3.0）**。您可以自由复制、分发和修改本软件，但必须保留相同的许可证。详情请参阅 [LICENSE](https://www.gnu.org/licenses/gpl-3.0.zh-cn.html) 文件。

## 联系方式 ✉️

如有任何问题或建议，欢迎联系：

- **GitHub**: [wangyifan349](https://github.com/wangyifan349)
- **邮箱**: *wangyifan349@gmail.com*

---

感谢您的使用！😊
