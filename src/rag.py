import os
import re
from src.vectordb import VectorDB
from src.llm import LLM

class RAG:
    def __init__(self, vectordb: VectorDB, llm: LLM, top_k=5):
        self.vectordb = vectordb
        self.llm = llm
        self.top_k = top_k

    def _preprocess_question(self, question: str) -> str:
        question = re.sub(r'[^\w\s]', '', question)  # Loại bỏ dấu câu
        return question.strip().lower()

    def generate(self, questions: str):
        question_processed = self._preprocess_question(questions)
        if question_processed in ["xin chào", "chào", "hello", "hi"]:
            return "**Xin chào!** Tôi là trợ lý AI chuyên về phân tích tài chính. Bạn cần hỗ trợ gì hôm nay?"
        if "bạn là ai" in question_processed or "who are you" in question_processed:
            return "Tôi là trợ lý AI chuyên về phân tích tài chính, được huấn luyện để trả lời câu hỏi dựa trên tài liệu được cung cấp."
        context_chunks_docs = self.vectordb.search(question_processed)
        if not context_chunks_docs:
            context = "Không tìm thấy thông tin liên quan trong tài liệu."
        else:
            context = "\n".join([doc.page_content for doc in context_chunks_docs])

        return self.llm.generate_response(prompt=question_processed, context=context)

if __name__ == "__main__":
    from data_loader import DataLoader
    from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
    dir_path = '../data'
    # loader = DataLoader(dir_path)
    # docs = loader.load_docs()
    embed_func = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_db = VectorDB(db_path="chroma_db", embedding_function=embed_func)
    # vector_db.add_documents(docs)
    llm = LLM()
    rag = RAG(vectordb=vector_db, llm=llm)
    query = "chủ tịch tập đoàn Hòa Phát là ai?"
    response = rag.generate(query)
    print(response)