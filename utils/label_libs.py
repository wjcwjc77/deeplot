from openai import OpenAI
import os
from dotenv import load_dotenv
from tqdm import tqdm 
load_dotenv()
client = OpenAI(
    # 替换为您需要调用的模型服务Base Url
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    # 环境变量中配置您的API Key
    api_key=os.environ.get("ARK_API_KEY")
)

# 新增：处理lib文件夹下的txt文件
lib_dir = "lib"
if os.path.exists(lib_dir):
    # 获取所有txt文件
    txt_files = [f for f in os.listdir(lib_dir) if f.endswith(".txt")]
    
    # 使用tqdm添加进度条
    for filename in tqdm(txt_files, desc="处理文件中", unit="文件"):
        filepath = os.path.join(lib_dir, filename)
        
        # 读取文件内容
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # 检查是否已经处理过（开头是否有"图片描述："）
        if content and not content.startswith("图片描述："):
            # 调用模型获取描述
            completion = client.chat.completions.create(
                model="ep-20250205113409-kwbtt",
                messages=[
                    {"role": "system", "content": "你是一个图片分析的专家，请根据用户的输入图片（xml），\
                     给出详细的描述。\
                     示例：这是一个战略投资组合分析图：横轴为投资金额（20k至120k），纵轴为机会规模（1M至5M）。使用不同颜色和大小的圆圈表示项目属性——橙色（销售）、紫色（研发）、红色（营销）、绿色（合资），圆圈大小对应成功率（大圆75-100%，中圆50-75%，小圆<50%）。高投资高回报项目（如红色大圆）置于右上角，中小型研发和合资项目分散在左下方。右侧添加图例，包含颜色分类和三个虚线同心圆标注成功率分级。坐标轴需清晰标注刻度及标题。"},
                    {"role": "user", "content": f"请描述以下内容：{content}"},
                ],
            )
            description = completion.choices[0].message.content
            
            # 将描述写入文件开头
            with open(filepath, 'r+', encoding='utf-8') as f:
                existing_content = f.read()
                f.seek(0, 0)
                f.write(f"图片描述：{description}\n\n{existing_content}")
        else:
            print(f"跳过已处理文件: {filename}")
else:
    print(f"警告：{lib_dir} 文件夹不存在")