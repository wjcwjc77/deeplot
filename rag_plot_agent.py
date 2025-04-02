import io
import re
from PIL import Image
import logging
import asyncio
from typing import AsyncGenerator,List,Iterable
from openai.types.responses import ResponseTextDeltaEvent

from agents import Agent, Runner
import gradio as gr
import matplotlib
import tiktoken
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader
import xml.etree.ElementTree as ET

class RAGAgent(Agent):
    def __init__(self, name, instructions, model, data_path="drawio-diagrams"):
        super().__init__(name=name, instructions=instructions, model=model)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            chunk_size=256,  # 限制每次处理的文本块大小
            max_retries=3  
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=self.tiktoken_len
        )
        self.vectorstore = self.init_vectorstore(data_path)

    def tiktoken_len(self, text):
        """计算文本token长度"""
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

    def init_vectorstore(self, data_path):
        """初始化向量数据库"""
        
        class DrawIOLoader:
            def __init__(self, file_path: str):
                self.file_path = file_path
            
            def load(self) -> List[Document]:
                """实现 load 方法返回 Document 列表"""
                return list(self.lazy_load())
            
            def lazy_load(self) -> Iterable[Document]:
                """实现 lazy_load 方法"""
                content = self.parse_drawio(self.file_path)
                yield Document(page_content=content)
            
            @staticmethod
            def parse_drawio(file_path: str) -> str:
                """解析DrawIO文件内容"""
                try:
                    tree = ET.parse(file_path)
                    root = tree.getroot()
                    content = []
                    for elem in root.iter():
                        if elem.text and elem.text.strip():
                            content.append(elem.text.strip())
                        if 'value' in elem.attrib:
                            content.append(elem.attrib['value'])
                    return "\n".join(content)
                except ET.ParseError as e:
                    print(f"解析错误 {file_path}: {str(e)}")
                    return ""
        try:
            loader = DirectoryLoader(
                data_path,
                glob="**/*.drawio",
                loader_cls=DrawIOLoader,
                show_progress=True
            )
            documents = loader.load()
            logger.info(f"已加载 {len(documents)} 个文档")
            docs = self.text_splitter.split_documents(documents)
            logger.info(f"分割为 {len(docs)} 个文本块")
            return Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory="./chroma_db"
            )
        except Exception as e:
            logger.error(f"初始化向量存储失败: {str(e)}")
            raise


    async def search(self, query, k=3):
        """执行相似性检索"""
        try:
            # 添加查询预处理
            query = query.strip()
            if not query:
                raise ValueError("查询内容不能为空")
                
            logger.info(f"执行搜索: {query}")
            results = self.vectorstore.similarity_search(query, k=k)
            
            # 添加结果日志
            logger.info(f"找到 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            return []


# 创建流程图检索专家代理
rag_agent = RAGAgent(
    name="流程图绘制专家",
    instructions="""你是一个专业的流程图绘制专家，专门负责处理和创建 DrawIO 格式的流程图。

    1. 首先理解用户的检索需求，使用语义相似度搜索找到相关的流程图内容。
    2. 分析检索到的流程图内容，提取关键元素和结构关系。
    3. 根据分析结果，生成新的DrawIO格式流程图XML代码。
    4. 总是使用英文标点符号，不使用中文标点符号。
    5. 对于检索结果，提供简洁的解释说明其如何影响新流程图的生成。
    6. 生成的DrawIO代码必须符合标准格式，包含必要的形状、连接线和文本。
    7. 如果检索结果不够理想，提供改进搜索关键词的建议或询问更多细节。
    8. 保持专业和客观的语气。
    9. 在回复中，先展示检索到的相关内容，然后提供生成的DrawIO代码。
    10. 对生成的流程图进行简要说明，解释关键设计决策。

    回复格式：
    - 首先说明搜索策略和检索结果
    - 然后展示生成的DrawIO代码
    - 最后解释流程图设计思路
    """,
    model="ep-20250205113409-kwbtt",
)

