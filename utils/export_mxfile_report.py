import os
from pathlib import Path

def is_valid_mxfile(content):
    """验证内容是否是有效的mxfile格式"""
    content = content.strip()
    return content.startswith('<mxfile') and content.endswith('</mxfile>')

def read_mxfile(file_path):
    """读取符合mxfile格式的文件原始内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if is_valid_mxfile(content):
                # 检查是否在一行内
                if '\n' not in content.strip():
                    return None  # 单行内容不返回
                return content.strip()  # 返回去首尾空白的原始内容
            return None  # 不符合格式返回None
    except Exception as e:
        return f"[读取错误] {str(e)}: {file_path}"

def export_mxfile_to_lib(root_dir):
    """将mxfile文件导出到lib目录，每个文件单独保存"""
    valid_extensions = ('.drawio', '.xml')
    lib_dir = os.path.join(root_dir, 'lib')
    
    # 创建lib目录
    Path(lib_dir).mkdir(parents=True, exist_ok=True)
    
    exported_files = []
    
    for dirpath, _, filenames in os.walk(root_dir):
        # 跳过lib目录本身
        if os.path.normpath(dirpath) == os.path.normpath(lib_dir):
            continue
            
        for filename in sorted(f for f in filenames if f.lower().endswith(valid_extensions)):
            file_path = os.path.join(dirpath, filename)
            content = read_mxfile(file_path)
            
            if content is None:  # 跳过不符合格式的文件
                continue
                
            # 生成新的txt文件名
            base_name = os.path.splitext(filename)[0]
            txt_filename = f"{base_name}.txt"
            txt_path = os.path.join(lib_dir, txt_filename)
            
            try:
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                exported_files.append(txt_path)
            except IOError as e:
                print(f"保存文件 {txt_path} 时出错: {str(e)}")
    
    return exported_files

if __name__ == "__main__":
    target_dir = "/Users/bytedance/PycharmProjects/deeplot/drawio-diagrams"
    
    print("正在导出mxfile格式文件（.drawio和.xml）到lib目录...")
    exported_files = export_mxfile_to_lib(target_dir)
    
    if exported_files:
        print(f"成功导出 {len(exported_files)} 个文件到 lib 目录:")
        for file in exported_files[:5]:  # 只显示前5个文件路径
            print(f"- {file}")
        if len(exported_files) > 5:
            print(f"- ...(共 {len(exported_files)} 个文件)")
    else:
        print("未找到有效的mxfile格式文件")