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
            max_tokens=1024
        )

    def generate_response(self, prompt, context):
        messages = [
            SystemMessage(content="Bạn là trợ lý thông minh hỗ trợ phân tích báo cáo tài chính. Hãy trả lời dựa vào thông tin được cung cấp."),
            HumanMessage(content=f"Context: {context}\n\nQuestion: {prompt}\n\nTrả lời bằng tiếng Việt:")
        ]
        return self.llm.invoke(messages).content
