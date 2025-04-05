import os
import sys
import logging
from openai import OpenAI
from typing import Any, Generator
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from functools import cached_property

# 配置日志 创建一个与当前模块同名的 logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 从环境变量获取 API 密钥
load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise ValueError("DEEPSEEK_API_KEY environment variable is not set")

class DeepSeekChat(BaseModel):
    """DeepSeek 聊天模型的封装类。"""

    api_key: str = Field(default=API_KEY)
    base_url: str = Field(default="https://api.deepseek.com")

    class Config:
        """Pydantic 配置类。"""

        arbitrary_types_allowed = True  # 允许模型接受任意类型的字段
        # 这增加了灵活性，但可能降低类型安全性
        # 在本类中，这可能用于允许使用 OpenAI 客户端等复杂类型

    @cached_property
    def client(self) -> OpenAI:
        """创建并缓存 OpenAI 客户端实例。"""
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(
        self,
        system_message: str,
        user_message: str,
        model: str = "deepseek-chat",
        max_tokens: int = 8192,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> Any:
        """
        使用 DeepSeek API 发送聊天请求。

        返回流式响应或完整响应内容。
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )
            return response if stream else response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in DeepSeek API call: {e}")
            raise

    def _stream_response(self, response) -> Generator[str, None, None]:
        """处理流式响应，逐块生成内容。"""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

system_message = """
你是一个流程图绘制专家，请根据用户的输入描述，结合数据库，给出精彩好看的流程图，颜色一定要好看，美观，图片要精致，符合SCI标准。
你必须严格按照以下的格式返回xml图。也就是说，流程图的内容一定在<mxfile></mxfile>之间，下面是一个transormer框架的示例，相邻模块用箭头链接，模块与模块直接排列紧凑。当然仅为示例，你需要发挥自己的学习能力，绘制出不同风格的图来。
示例：
<mxfile host="65bd71144e">
    <diagram id="Transformer-Architecture" name="Page-1">
        <mxGraphModel dx="1440" dy="886" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1400" pageHeight="1000" math="0" shadow="0">
            <root>
                <mxCell id="0"/>
                <mxCell id="1" parent="0"/>
                <mxCell id="encoder_block" value="Encoder Block × N" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666666;verticalAlign=top;fontStyle=1;arcSize=4;" parent="1" vertex="1">
                    <mxGeometry x="180" y="480" width="150" height="213" as="geometry"/>
                </mxCell>
                <mxCell id="52" style="edgeStyle=none;html=1;entryX=0.5;entryY=1;entryDx=0;entryDy=0;" edge="1" parent="1" source="attention_detail" target="51">
                    <mxGeometry relative="1" as="geometry"/>
                </mxCell>
                <mxCell id="attention_detail" value="Linear" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;" parent="1" vertex="1">
                    <mxGeometry x="380" y="310" width="100" height="30" as="geometry"/>
                </mxCell>
                <mxCell id="16" style="edgeStyle=none;html=1;entryX=0.5;entryY=1;entryDx=0;entryDy=0;" edge="1" parent="1" source="feed_forward" target="output">
                    <mxGeometry relative="1" as="geometry"/>
                </mxCell>
                <mxCell id="feed_forward" value="Add &amp;amp; Norm" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;" parent="1" vertex="1">
                    <mxGeometry x="210" y="603" width="100" height="30" as="geometry"/>
                </mxCell>
                <mxCell id="add_norm2" value="Multi-Head&lt;div&gt;Attention&lt;/div&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#000000;" parent="1" vertex="1">
                    <mxGeometry x="210" y="643" width="100" height="30" as="geometry"/>
                </mxCell>
                <mxCell id="linear" value="Input&lt;div&gt;Embedding&lt;/div&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#000000;" parent="1" vertex="1">
                    <mxGeometry x="200" y="770" width="120" height="40" as="geometry"/>
                </mxCell>
                <mxCell id="output" value="Feed&lt;div&gt;Forward&lt;/div&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;" parent="1" vertex="1">
                    <mxGeometry x="210" y="550" width="100" height="30" as="geometry"/>
                </mxCell>
                <mxCell id="2" value="" style="verticalLabelPosition=bottom;verticalAlign=top;html=1;shape=mxgraph.flowchart.summing_function;" vertex="1" parent="1">
                    <mxGeometry x="250" y="720" width="20" height="21" as="geometry"/>
                </mxCell>
                <mxCell id="5" style="edgeStyle=none;html=1;entryX=0.5;entryY=1;entryDx=0;entryDy=0;entryPerimeter=0;" edge="1" parent="1" source="linear" target="2">
                    <mxGeometry relative="1" as="geometry"/>
                </mxCell>
                <mxCell id="6" value="Positional&lt;div&gt;Encoding&lt;/div&gt;" style="rounded=1;whiteSpace=wrap;html=1;strokeColor=none;fillColor=none;" vertex="1" parent="1">
                    <mxGeometry x="150" y="710.5" width="50" height="40" as="geometry"/>
                </mxCell>
                <mxCell id="7" style="edgeStyle=none;html=1;entryX=0;entryY=0.5;entryDx=0;entryDy=0;entryPerimeter=0;" edge="1" parent="1" source="6" target="2">
                    <mxGeometry relative="1" as="geometry"/>
                </mxCell>
                <mxCell id="9" style="edgeStyle=none;html=1;exitX=0.5;exitY=0;exitDx=0;exitDy=0;exitPerimeter=0;entryX=0.5;entryY=1;entryDx=0;entryDy=0;" edge="1" parent="1" source="2" target="add_norm2">
                    <mxGeometry relative="1" as="geometry"/>
                </mxCell>
                <mxCell id="14" value="" style="endArrow=classic;html=1;entryX=0;entryY=0.5;entryDx=0;entryDy=0;edgeStyle=orthogonalEdgeStyle;" edge="1" parent="1" target="feed_forward">
                    <mxGeometry width="50" height="50" relative="1" as="geometry">
                        <mxPoint x="260" y="690" as="sourcePoint"/>
                        <mxPoint x="250" y="670" as="targetPoint"/>
                        <Array as="points">
                            <mxPoint x="190" y="690"/>
                            <mxPoint x="190" y="618"/>
                        </Array>
                    </mxGeometry>
                </mxCell>
                <mxCell id="15" value="Add &amp;amp; Norm" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;" vertex="1" parent="1">
                    <mxGeometry x="210" y="510" width="100" height="30" as="geometry"/>
                </mxCell>
                <mxCell id="20" value="" style="endArrow=classic;html=1;entryX=0;entryY=0.5;entryDx=0;entryDy=0;edgeStyle=orthogonalEdgeStyle;" edge="1" parent="1" target="15">
                    <mxGeometry width="50" height="50" relative="1" as="geometry">
                        <mxPoint x="260" y="590" as="sourcePoint"/>
                        <mxPoint x="160" y="590" as="targetPoint"/>
                        <Array as="points">
                            <mxPoint x="190" y="590"/>
                            <mxPoint x="190" y="525"/>
                        </Array>
                    </mxGeometry>
                </mxCell>
                <mxCell id="21" value="" style="endArrow=none;html=1;exitX=0.5;exitY=0;exitDx=0;exitDy=0;entryX=0.5;entryY=1;entryDx=0;entryDy=0;" edge="1" parent="1" source="add_norm2" target="feed_forward">
                    <mxGeometry width="50" height="50" relative="1" as="geometry">
                        <mxPoint x="250" y="680" as="sourcePoint"/>
                        <mxPoint x="300" y="630" as="targetPoint"/>
                    </mxGeometry>
                </mxCell>
                <mxCell id="22" value="" style="endArrow=none;html=1;exitX=0.5;exitY=0;exitDx=0;exitDy=0;entryX=0.5;entryY=1;entryDx=0;entryDy=0;" edge="1" parent="1" source="output" target="15">
                    <mxGeometry width="50" height="50" relative="1" as="geometry">
                        <mxPoint x="250" y="570" as="sourcePoint"/>
                        <mxPoint x="300" y="520" as="targetPoint"/>
                    </mxGeometry>
                </mxCell>
                <mxCell id="23" value="Decoder Block × N" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666666;verticalAlign=top;fontStyle=1;arcSize=4;" vertex="1" parent="1">
                    <mxGeometry x="365" y="360" width="150" height="333" as="geometry"/>
                </mxCell>
                <mxCell id="24" value="Masked&lt;div&gt;Multi-Head&lt;/div&gt;&lt;div&gt;Attention&lt;/div&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#000000;arcSize=9;" vertex="1" parent="1">
                    <mxGeometry x="380" y="630" width="100" height="43" as="geometry"/>
                </mxCell>
                <mxCell id="26" style="edgeStyle=none;html=1;entryX=0.5;entryY=1;entryDx=0;entryDy=0;" edge="1" parent="1" source="45" target="24">
                    <mxGeometry relative="1" as="geometry"/>
                </mxCell>
                <mxCell id="25" value="Ouput&lt;div&gt;Embedding&lt;/div&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#000000;" vertex="1" parent="1">
                    <mxGeometry x="370" y="770" width="120" height="40" as="geometry"/>
                </mxCell>
                <mxCell id="39" style="edgeStyle=orthogonalEdgeStyle;html=1;entryX=0.75;entryY=1;entryDx=0;entryDy=0;" edge="1" parent="1" source="27" target="31">
                    <mxGeometry relative="1" as="geometry"/>
                </mxCell>
                <mxCell id="27" value="Add &amp;amp; Norm" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;" vertex="1" parent="1">
                    <mxGeometry x="380" y="590" width="100" height="30" as="geometry"/>
                </mxCell>
                <mxCell id="28" value="" style="endArrow=none;html=1;entryX=0.5;entryY=1;entryDx=0;entryDy=0;exitX=0.5;exitY=0;exitDx=0;exitDy=0;" edge="1" parent="1" source="24" target="27">
                    <mxGeometry width="50" height="50" relative="1" as="geometry">
                        <mxPoint x="320" y="740" as="sourcePoint"/>
                        <mxPoint x="370" y="690" as="targetPoint"/>
                    </mxGeometry>
                </mxCell>
                <mxCell id="29" style="edgeStyle=none;html=1;entryX=0.5;entryY=1;entryDx=0;entryDy=0;" edge="1" source="30" target="32" parent="1">
                    <mxGeometry relative="1" as="geometry"/>
                </mxCell>
                <mxCell id="30" value="Add &amp;amp; Norm" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;" vertex="1" parent="1">
                    <mxGeometry x="380" y="483" width="100" height="30" as="geometry"/>
                </mxCell>
                <mxCell id="31" value="Multi-Head&lt;div&gt;Attention&lt;/div&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#000000;" vertex="1" parent="1">
                    <mxGeometry x="380" y="523" width="100" height="30" as="geometry"/>
                </mxCell>
                <mxCell id="32" value="Feed&lt;div&gt;Forward&lt;/div&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="1">
                    <mxGeometry x="380" y="430" width="100" height="30" as="geometry"/>
                </mxCell>
                <mxCell id="33" value="Add &amp;amp; Norm" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;" vertex="1" parent="1">
                    <mxGeometry x="380" y="390" width="100" height="30" as="geometry"/>
                </mxCell>
                <mxCell id="35" value="" style="endArrow=none;html=1;exitX=0.5;exitY=0;exitDx=0;exitDy=0;entryX=0.5;entryY=1;entryDx=0;entryDy=0;" edge="1" source="31" target="30" parent="1">
                    <mxGeometry width="50" height="50" relative="1" as="geometry">
                        <mxPoint x="420" y="560" as="sourcePoint"/>
                        <mxPoint x="470" y="510" as="targetPoint"/>
                    </mxGeometry>
                </mxCell>
                <mxCell id="36" value="" style="endArrow=none;html=1;exitX=0.5;exitY=0;exitDx=0;exitDy=0;entryX=0.5;entryY=1;entryDx=0;entryDy=0;" edge="1" source="32" target="33" parent="1">
                    <mxGeometry width="50" height="50" relative="1" as="geometry">
                        <mxPoint x="420" y="450" as="sourcePoint"/>
                        <mxPoint x="470" y="400" as="targetPoint"/>
                    </mxGeometry>
                </mxCell>
                <mxCell id="37" style="edgeStyle=orthogonalEdgeStyle;html=1;entryX=0.25;entryY=1;entryDx=0;entryDy=0;exitX=0.5;exitY=0;exitDx=0;exitDy=0;" edge="1" parent="1" source="15" target="31">
                    <mxGeometry relative="1" as="geometry">
                        <Array as="points">
                            <mxPoint x="260" y="500"/>
                            <mxPoint x="350" y="500"/>
                            <mxPoint x="350" y="570"/>
                            <mxPoint x="405" y="570"/>
                        </Array>
                    </mxGeometry>
                </mxCell>
                <mxCell id="38" style="edgeStyle=orthogonalEdgeStyle;html=1;entryX=0.5;entryY=1;entryDx=0;entryDy=0;exitX=0.5;exitY=0;exitDx=0;exitDy=0;" edge="1" parent="1" source="15" target="31">
                    <mxGeometry relative="1" as="geometry">
                        <mxPoint x="270" y="520" as="sourcePoint"/>
                        <mxPoint x="415" y="563" as="targetPoint"/>
                        <Array as="points">
                            <mxPoint x="260" y="500"/>
                            <mxPoint x="350" y="500"/>
                            <mxPoint x="350" y="570"/>
                            <mxPoint x="430" y="570"/>
                        </Array>
                    </mxGeometry>
                </mxCell>
                <mxCell id="42" value="" style="endArrow=classic;html=1;entryX=1;entryY=0.5;entryDx=0;entryDy=0;edgeStyle=orthogonalEdgeStyle;" edge="1" parent="1" target="27">
                    <mxGeometry width="50" height="50" relative="1" as="geometry">
                        <mxPoint x="430" y="690" as="sourcePoint"/>
                        <mxPoint x="500" y="660" as="targetPoint"/>
                        <Array as="points">
                            <mxPoint x="500" y="690"/>
                            <mxPoint x="500" y="605"/>
                        </Array>
                    </mxGeometry>
                </mxCell>
                <mxCell id="43" value="" style="endArrow=classic;html=1;entryX=1;entryY=0.5;entryDx=0;entryDy=0;edgeStyle=orthogonalEdgeStyle;" edge="1" parent="1">
                    <mxGeometry width="50" height="50" relative="1" as="geometry">
                        <mxPoint x="429.96" y="579.96" as="sourcePoint"/>
                        <mxPoint x="479.96" y="494.96" as="targetPoint"/>
                        <Array as="points">
                            <mxPoint x="499.96" y="579.96"/>
                            <mxPoint x="499.96" y="494.96"/>
                        </Array>
                    </mxGeometry>
                </mxCell>
                <mxCell id="44" value="" style="endArrow=classic;html=1;edgeStyle=orthogonalEdgeStyle;" edge="1" parent="1">
                    <mxGeometry width="50" height="50" relative="1" as="geometry">
                        <mxPoint x="430" y="475.00000000000006" as="sourcePoint"/>
                        <mxPoint x="480" y="400" as="targetPoint"/>
                        <Array as="points">
                            <mxPoint x="500" y="475"/>
                            <mxPoint x="500" y="400"/>
                        </Array>
                    </mxGeometry>
                </mxCell>
                <mxCell id="46" value="" style="edgeStyle=none;html=1;entryX=0.5;entryY=1;entryDx=0;entryDy=0;" edge="1" parent="1" source="25" target="45">
                    <mxGeometry relative="1" as="geometry">
                        <mxPoint x="430" y="770" as="sourcePoint"/>
                        <mxPoint x="430" y="673" as="targetPoint"/>
                    </mxGeometry>
                </mxCell>
                <mxCell id="45" value="" style="verticalLabelPosition=bottom;verticalAlign=top;html=1;shape=mxgraph.flowchart.summing_function;" vertex="1" parent="1">
                    <mxGeometry x="420" y="720" width="20" height="21" as="geometry"/>
                </mxCell>
                <mxCell id="47" value="Positional&lt;div&gt;Encoding&lt;/div&gt;" style="rounded=1;whiteSpace=wrap;html=1;strokeColor=none;fillColor=none;" vertex="1" parent="1">
                    <mxGeometry x="465" y="710.5" width="50" height="40" as="geometry"/>
                </mxCell>
                <mxCell id="48" style="edgeStyle=none;html=1;entryX=1;entryY=0.5;entryDx=0;entryDy=0;entryPerimeter=0;" edge="1" parent="1" source="47" target="45">
                    <mxGeometry relative="1" as="geometry"/>
                </mxCell>
                <mxCell id="49" style="edgeStyle=none;html=1;exitX=0.5;exitY=0;exitDx=0;exitDy=0;" edge="1" parent="1" source="33">
                    <mxGeometry relative="1" as="geometry">
                        <mxPoint x="430" y="340" as="targetPoint"/>
                    </mxGeometry>
                </mxCell>
                <mxCell id="54" style="edgeStyle=none;html=1;entryX=0.5;entryY=1;entryDx=0;entryDy=0;" edge="1" parent="1" source="51" target="53">
                    <mxGeometry relative="1" as="geometry"/>
                </mxCell>
                <mxCell id="51" value="Softmax" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#ffe6cc;strokeColor=#d79b00;" vertex="1" parent="1">
                    <mxGeometry x="380" y="260" width="100" height="30" as="geometry"/>
                </mxCell>
                <mxCell id="53" value="Output&lt;div&gt;probabilities&lt;/div&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor=none;" vertex="1" parent="1">
                    <mxGeometry x="380" y="210" width="100" height="30" as="geometry"/>
                </mxCell>
            </root>
        </mxGraphModel>
    </diagram>
</mxfile>
"""

class DeepSeekLLM(CustomLLM):
    """DeepSeek 语言模型的自定义实现。"""


    deep_seek_chat: DeepSeekChat = Field(default_factory=DeepSeekChat)

    @property
    def metadata(self) -> LLMMetadata:
        """返回 LLM 元数据。"""
        return LLMMetadata()

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """执行非流式完成请求。"""
        response = self.deep_seek_chat.chat(
            system_message=system_message, user_message=prompt, stream=False
        )
        return CompletionResponse(text=response)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """执行流式完成请求。"""
        response = self.deep_seek_chat.chat(
            system_message=system_message, user_message=prompt, stream=True
        )

        def response_generator():
            """生成器函数，用于逐步生成响应内容。"""
            response_content = ""
            for chunk in self.deep_seek_chat._stream_response(response):
                if chunk:
                    response_content += chunk
                    yield CompletionResponse(text=response_content, delta=chunk)

        return response_generator()

# 设置环境变量，禁用 tokenizers 的并行处理，以避免潜在的死锁问题
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    """主程序函数，演示如何使用 DeepSeekLLM 进行直接对话生成。"""
    # 初始化LLM
    llm = DeepSeekLLM()
    
    # 执行直接对话查询
    print("生成结果：")
    response = llm.complete("请绘制一下vllm的框架图")
    
    # 确保输出目录存在
    os.makedirs("llm_response", exist_ok=True)
    
    # 处理并保存响应
    output_content = response.text
    
    # 打印并保存结果
    print(output_content)
    
    # 将结果保存到文件
    with open("llm_response/test_noRag2.drawio", "w", encoding="utf-8") as f:
        f.write(output_content)
    
    print("\n生成完成，结果已保存至 llm_response/test3.drawio")

if __name__ == "__main__":
    main()
