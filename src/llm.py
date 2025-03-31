import os
from dotenv import load_dotenv
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
load_dotenv()

class LLM:
    def __init__(self):
        self.llm = BaseChatOpenAI(
            model='deepseek-chat',
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base=os.getenv("DEEPSEEK_API_BASE"),
            temperature=0.5,
            max_tokens=1024
        )

    def generate_response(self, prompt, context):
        messages = [
            SystemMessage(content="""Bạn là một trợ lý AI chuyên phân tích báo cáo tài chính, hoạt động như một chuyên gia. Nhiệm vụ của bạn là trả lời câu hỏi dựa vào nội dung trong phần 'Context' được cung cấp.
            Hãy tuân thủ các quy tắc sau:
            -  **Chỉ sử dụng thông tin từ 'Context'.** Không được thêm thông tin bên ngoài hoặc kiến thức trước đó của bạn.
            -  Có thể thêm một chút kiến thức liên quan đến tài chính, báo cáo, nếu được hỏi.
            -  Trình bày câu trả lời một cách rõ ràng, mạch lạc và chuyên nghiệp bằng tiếng Việt chuyên ngành tài chính.
            -  Luôn trả lời bằng tiếng Việt.
            """),
            HumanMessage(content=f"Context: {context}\n\nQuestion: {prompt}:")
        ]
        return self.llm.invoke(messages).content
