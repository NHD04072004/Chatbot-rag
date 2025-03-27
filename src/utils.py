from langchain_community.document_loaders import PDFPlumberLoader

def read_pdf(pdf_path):
    loader = PDFPlumberLoader(pdf_path, extract_images=True)
    return loader.load()

def load_documents():
    documents = []
    