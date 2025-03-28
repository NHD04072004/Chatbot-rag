from flask import Flask, render_template, request, jsonify, url_for
from src.rag import RAG

app = Flask(__name__)

rag = RAG(data_folder="data", db_path="chroma_db")
rag.initialize()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")

    if not user_input:
        return jsonify({"response": "Vui lòng nhập câu hỏi!"})

    response = rag.query(user_input)

    return jsonify({"response": response})