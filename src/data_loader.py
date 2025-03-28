import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

class DataLoader:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    @staticmethod
    def read_pdf(file_path):
        loader = PDFPlumberLoader(file_path)
        return loader.load()

    def load_docs(self):
        documents = []
        for file_name in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, file_name)
            docs = self.read_pdf(file_path)
            documents.extend(docs)

        # text_splitter = SemanticChunker(HuggingFaceEmbeddings(
        #     model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        #     model_kwargs={'device': 'cuda'},
        #     encode_kwargs={'normalize_embeddings': True}
        # ))
        text_splitter = SemanticChunker(OpenAIEmbeddings(
            model='text-embedding-3-small',
            openai_api_key=os.getenv("OPENAI_API_KEY")
        ))
        return text_splitter.split_documents(documents)

if __name__ == "__main__":
    dir_path = '../data'
    loader = DataLoader(dir_path)
    docs = loader.load_docs()
    print(docs)
    print(len(docs))
