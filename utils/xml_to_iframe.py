import os
import base64
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

def decode_base64_diagrams(directory):
    """
    解码目录中所有XML文件中的Base64图表数据
    """
    for filename in os.listdir(directory):
        if filename.endswith('.xml'):
            filepath = os.path.join(directory, filename)
            print(f"Processing file: {filename}")
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 使用BeautifulSoup解析XML
                soup = BeautifulSoup(content, 'xml')
                diagram_tag = soup.find('diagram')
                
                if diagram_tag and diagram_tag.string:
                    # 解码Base64内容
                    decoded_content = base64.b64decode(diagram_tag.string).decode('utf-8')
                    
                    # 创建解码后的文件名
                    decoded_filename = f"decoded_{filename}"
                    decoded_filepath = os.path.join(directory, decoded_filename)
                    
                    # 保存解码后的内容
                    with open(decoded_filepath, 'w', encoding='utf-8') as f:
                        f.write(decoded_content)
                    
                    print(f"Successfully decoded and saved as: {decoded_filename}")
                else:
                    print("No Base64 diagram content found in this file")
            
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    # 指定包含XML文件的目录
    directory_path = "/Users/bytedance/PycharmProjects/deeplot/utils/mxfile_report.txt"  # 替换为你的实际目录路径
    decode_base64_diagrams(directory_path)