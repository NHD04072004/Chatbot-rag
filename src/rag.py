from src.data_loader import DataLoader
from src.vectordb import VectorDB
from src.llm import LLM

class RAG:
    def __init__(self, data_folder: str, db_path: str):
        self.data_loader = DataLoader(data_folder)
        self.vector_db = VectorDB(db_path)
        self.llm = LLM()

    def initialize(self):
        documents = self.data_loader.load_docs()
        self.vector_db.add_documents(documents)

    def query(self, question: str, top_k: int = 3) -> str:
        context_chunks = self.vector_db.search(question, top_k)
        context = "\n\n".join(context_chunks)

        prompt = f"""Hãy trả lời câu hỏi dựa trên thông tin sau:
        {context}

        Câu hỏi: {question}

        Yêu cầu:
        - Nếu người dùng hỏi bạn là ai, hãy trả lời với họ là 'Tôi là bot được anh Đăng đẹp trai tạo ra'
        - Nếu người dùng đưa ra những câu mang ý nghĩa chào hỏi, hãy chào hỏi lại một cách đàng hoàng
        - Trả lời bằng tiếng Việt
        - Nếu không có thông tin, hãy nói 'Không tìm thấy thông tin liên quan'
        - Định dạng văn bản bằng Markdown (in đậm, in nghiêng, danh sách, bảng nếu cần)
        - Đưa ra các con số cụ thể nếu có"""

        return self.llm.generate_response(prompt, context)


if __name__ == "__main__":
    rag = RAG(
        data_folder="../data",
        db_path="chroma_db"
    )
    rag.initialize()

    question = "cho tôi thông tin về DỰ ÁN KHU LIÊN HỢP SẢN XUẤT GANG THÉP HÒA PHÁT DUNG QUẤT 2"
    answer = rag.query(question)
    print("Câu trả lời:", answer)