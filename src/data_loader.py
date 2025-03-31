import os
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker
from langchain.schema import Document
from dotenv import load_dotenv
load_dotenv()

class DataLoader:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend
                )
            }
        )
        self.chunker = HybridChunker(tokenizer='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', merge_peers=True)

    def process_file(self, file_path):
        documents = []
        full_doc = self.converter.convert(file_path).document
        chunks = self.chunker.chunk(dl_doc=full_doc)
        metadata = {'source': file_path}
        for chunk in chunks:
            documents.append(Document(
                page_content=self.chunker.serialize(chunk=chunk),
                metadata=metadata
            ))
        return documents

    def load_docs(self):
        pdf_files = [
            os.path.join(self.folder_path, f)
            for f in os.listdir(self.folder_path)
            if f.lower().endswith('.pdf')
        ]
        results = map(self.process_file, pdf_files)

        all_docs = []
        for docs in results:
            all_docs.extend(docs)
        return all_docs

if __name__ == "__main__":
    dir_path = '../data'
    loader = DataLoader(dir_path)
    docs = loader.load_docs()
    print(docs)
    print(len(docs))
