# Deeplot

**Deeplot 能够快速的渲染出来 AI 生成的画图代码！**

Deeplot 是一个基于 Gradio 和 Matplotlib 的 Python 代码可视化工具！ 
允许用户通过简单的 Web 界面编写和执行 Python 绘图代码，并立即查看可视化结果。
该工具特别支持中文字体显示，适合中文环境下的数据可视化工作。

[🔗在线试用](https://tool.ailln.com/)

## 功能特点

- 🖥️ 简洁直观的 Web 界面
- 📊 支持 Matplotlib、Seaborn 等主流绘图库
- 🀄 完善的中文字体支持，自动适配不同操作系统
- 📥 支持图像下载功能
- 📝 内置多个示例代码，方便快速上手

## 安装说明

1. 克隆仓库到本地：

```bash
git clone https://github.com/open-v2ai/deeplot.git
cd deeplot
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

1. 运行应用：

```bash
python run.py
```

2. 在浏览器中访问：`http://localhost:8080`

3. 在左侧代码编辑区输入 Python 绘图代码

4. 点击"执行代码"按钮查看结果

5. 可以下载生成的图像用于其他用途

## 示例

Deeplot 内置了多个示例代码，包括：
- 基础折线图（正弦波）
- 柱状图
- 饼图（支持中文标签）

## 项目结构

```
deeplot/
├── run.py           # 主程序文件
├── requirements.txt # 项目依赖
└── README.md        # 项目说明文档
```

## 许可证

本项目采用 MIT 开源许可证。
