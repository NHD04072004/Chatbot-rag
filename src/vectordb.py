import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

load_dotenv()

class VectorDB:
    def __init__(self, db_path):
        # self.embedding_function = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert", model_kwargs={'device': 'cuda'})
        # self.embedding_function = HuggingFaceEmbeddings(
        #     model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        #     model_kwargs={'device': 'cuda'},
        #     encode_kwargs={'normalize_embeddings': True}
        # )
        self.embedding_function = OpenAIEmbeddings(
            model='text-embedding-3-small',
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.vectorstore = Chroma(
            collection_name='documents',
            embedding_function=self.embedding_function,
            persist_directory=db_path
        )

    def add_documents(self, docs):
        self.vectorstore.add_documents(docs)

    def search(self, query, top_k=3):
        results = self.vectorstore.similarity_search(query, k=top_k)
        return [doc.page_content for doc in results]

if __name__ == "__main__":
    from data_loader import DataLoader
    dir_path = '../data'
    loader = DataLoader(dir_path)
    docs = loader.load_docs()
    vector_db = VectorDB(db_path="chroma_db")
    vector_db.add_documents(docs)
    query = "tổng quan về hòa phát"
    results = vector_db.search(query)
    print(results)