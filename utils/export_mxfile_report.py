import os

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

def generate_mxfile_report(root_dir):
    """生成mxfile格式文件报告（包含.drawio和.xml）"""
    report = []
    valid_extensions = ('.drawio', '.xml')
    
    for dirpath, _, filenames in os.walk(root_dir):
        rel_dir = os.path.relpath(dirpath, start=root_dir)
        if rel_dir == ".":
            continue
            
        dir_files = []
        
        for filename in sorted(f for f in filenames if f.lower().endswith(valid_extensions)):
            file_path = os.path.join(dirpath, filename)
            content = read_mxfile(file_path)
            
            if content is None:  # 跳过不符合格式的文件
                continue
                
            dir_files.append((filename, content))
        
        if dir_files:  # 只显示有有效文件的目录
            report.append(f"\n目录名称: {rel_dir}")
            for filename, content in dir_files:
                report.append(f"\n文件名: {filename}")
                report.append("文件内容:")
                report.append(content)
    
    return "\n".join(report) if report else "未找到有效的mxfile格式文件"

if __name__ == "__main__":
    target_dir = "/Users/bytedance/PycharmProjects/deeplot/drawio-diagrams"
    
    print("正在扫描mxfile格式文件（.drawio和.xml）...")
    report = generate_mxfile_report(target_dir)
    
    # 保存报告
    output_file = "mxfile_report.txt"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"报告已生成，保存到: {os.path.abspath(output_file)}")
        print("="*50)
        print("目录结构示例:")
        if report:  # 检查report是否为空
            print("\n".join([line for line in report.split('\n') if line.startswith("目录名称:")][:3]))
        else:
            print("无有效目录结构")
    except IOError as e:
        print(f"保存报告时出错: {str(e)}")