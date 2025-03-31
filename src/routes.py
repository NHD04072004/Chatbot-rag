from flask import Flask, render_template, request
from src.rag import RAG
from src.llm import LLM
from src.vectordb import VectorDB
from langchain_community.embeddings import HuggingFaceEmbeddings

app = Flask(__name__)
embed_func = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
vector_db = VectorDB(db_path="chroma_db", embedding_function=embed_func)
rag = RAG(vectordb=vector_db, llm=LLM())

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        question = request.form['question']
        response = rag.generate(question)
        return render_template('index.html', question=question, response=response)
    else:
        return render_template('index.html')
