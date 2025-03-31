from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from langchain_docling import DoclingLoader

source = 'data/HPG_Baocaothuongnien_2023.pdf'

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False
pipeline_options.do_table_structure = True
pipeline_options.table_structure_options.do_cell_matching = True
# pipeline_options.ocr_options.lang = ["vi"]
# pipeline_options.accelerator_options = AcceleratorOptions(
#     num_threads=4, device=AcceleratorDevice.AUTO
# )

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend)
    }
)
# print(doc.export_to_text())
chunker = HybridChunker(tokenizer='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', merge_peers=True)

loader = DoclingLoader(file_path=source, converter=converter, chunker=chunker)
docs = loader.load()
print(docs)
print(len(docs))
# chunk_iter = chunker.chunk(dl_doc=doc)
# for i, chunk in enumerate(chunk_iter):
#     print(f"=== {i} ===")
#     # print(f"chunk.text:\n{repr(f'{chunk.text[:300]}…')}")
#     print(chunk.text)
#     # enriched_text = chunker.serialize(chunk=chunk)
#     # print(f"chunker.serialize(chunk):\n{f'{enriched_text[:300]}…'}")
#
#     print()