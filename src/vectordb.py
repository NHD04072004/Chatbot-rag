import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
load_dotenv()

class VectorDB:
    def __init__(self, db_path, embedding_function):
        self.vectorstore = Chroma(
            collection_name='documents',
            embedding_function=embedding_function,
            persist_directory=db_path
        )

    def add_documents(self, docs):
        self.vectorstore.add_documents(docs)

    def search(self, query, top_k=5):
        return self.vectorstore.similarity_search(query, k=top_k)

if __name__ == "__main__":
    from data_loader import DataLoader
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
    query = "chủ tịch tập đoàn Hòa Phát"
    results = vector_db.search(query)
    print(results)