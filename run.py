import gradio as gr
import matplotlib.pyplot as plt
import io
from PIL import Image
import matplotlib as mpl
import platform
import os

# 配置中文字体支持
def configure_chinese_font():
    # 保存原始字体配置
    original_font = mpl.rcParams['font.sans-serif'].copy()
    
    # 根据系统添加合适的中文字体
    system = platform.system()
    if system == 'Windows':
        chinese_fonts = ['Microsoft YaHei', 'SimHei', 'Arial']
    elif system == 'Darwin':  # macOS
        chinese_fonts = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'STHeiti', 'Arial']
    else:  # Linux
        # 在Linux上尝试更多通用字体
        chinese_fonts = ['DejaVu Sans', 'Liberation Sans', 'WenQuanYi Micro Hei', 'Droid Sans Fallback', 'Noto Sans CJK SC', 'Noto Sans CJK TC', 'Arial']
    
    # 将中文字体添加到字体列表的前面，而不是完全替换
    mpl.rcParams['font.sans-serif'] = chinese_fonts + original_font
    mpl.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    

# 应用中文字体配置
configure_chinese_font()

# 确保 data 目录存在
def ensure_data_dir():
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir

def execute_python_code(code):
    # 创建一个内存中的图像缓冲区
    buf = io.BytesIO()
    
    try:
        # 清除之前的图形
        plt.clf()
        plt.close('all')
        
        # 使用更简单的方法处理中文字体
        font_setup_code = """
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 尝试找到可用的中文字体
def find_chinese_font():
    fonts_to_try = [
        'Noto Sans CJK SC', 'Noto Sans CJK TC', 'Noto Sans CJK JP',
        'WenQuanYi Micro Hei', 'Droid Sans Fallback', 'SimHei', 
        'Microsoft YaHei', 'PingFang SC', 'STHeiti', 'Arial Unicode MS'
    ]
    
    for font in fonts_to_try:
        if any(font.lower() in f.name.lower() for f in fm.fontManager.ttflist):
            return font
    return None

# 找到中文字体
chinese_font = find_chinese_font()

# 设置全局字体
if chinese_font:
    plt.rcParams['font.sans-serif'] = [chinese_font] + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
"""
        # 先执行字体设置代码
        exec(font_setup_code, globals())
        
        # 执行用户代码
        exec(code, globals())
        
        # 检查是否有图形对象
        if plt.get_fignums():

            # 保存图形到缓冲区
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            
            # 转换为PIL图像并复制到内存中，这样可以安全关闭原始缓冲区
            img = Image.open(buf).copy()

            return img, None
        else:
            # 创建一个带有文本的图像，提示用户没有生成图像
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "代码执行成功，但没有生成图像。\n请确保代码中包含绘图命令。", 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf).copy()
            return img, None
    
    except Exception as e:
        # 创建一个带有错误信息的图像
        fig, ax = plt.subplots(figsize=(10, 6))
        error_msg = f"代码执行错误:\n{str(e)}"
        ax.text(0.5, 0.5, error_msg, ha='center', va='center', fontsize=12, color='red')
        ax.axis('off')
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf).copy()
        return img, None
    
    finally:
        # 关闭缓冲区
        buf.close()

# 修改默认代码
default_code = '''import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# 生成数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 创建图表
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2)

# 设置标题和标签
plt.title('正弦波')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)'''

# 创建Gradio界面
with gr.Blocks(title="Deepplot: Python 代码可视化工具", theme=gr.themes.Base()) as demo:
    gr.Markdown("# Deepplot: Python 代码可视化工具")
    gr.Markdown("在左侧输入 Python 代码，点击「执行代码」按钮查看结果。支持 matplotlib、seaborn 等绘图库。")
    
    with gr.Row():
        with gr.Column(scale=1):
            code_input = gr.Code(
                label="Python代码", 
                language="python",
                value=default_code,
                lines=20
            )
            run_button = gr.Button("执行代码", variant="primary")
    
        with gr.Column(scale=1):
            image_output = gr.Image(label="生成的图像", type="pil")
            file_output = gr.File(label="下载图片", visible=False)
    
    # 执行代码按钮事件
    def process_code(code):
        img, _ = execute_python_code(code)
        # 始终隐藏下载链接
        return img, gr.update(visible=False)
    
    run_button.click(
        fn=process_code,
        inputs=[code_input],
        outputs=[image_output, file_output]
    )
    
    gr.Examples(
        [
            ['''import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2)
plt.title('正弦波')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)'''],
            ['''import matplotlib.pyplot as plt
import numpy as np

# 创建数据
categories = ['A', 'B', 'C', 'D', 'E']
values = [22, 35, 14, 28, 19]

# 创建柱状图
plt.figure(figsize=(10, 6))
plt.bar(categories, values, color='skyblue')
plt.title('简单柱状图')
plt.xlabel('类别')
plt.ylabel('数值')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)'''],
            
            ['''import matplotlib.pyplot as plt
import numpy as np

# 创建数据
labels = ['苹果', '香蕉', '橙子', '葡萄', '西瓜']
sizes = [25, 20, 15, 30, 10]
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
explode = (0.1, 0, 0, 0, 0)  # 突出第一个切片

# 创建饼图
plt.figure(figsize=(10, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')  # 确保饼图是圆的
plt.title('水果比例')''']
        ],
        inputs=[code_input],
        label="示例代码"
    )

# 启动应用
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)
