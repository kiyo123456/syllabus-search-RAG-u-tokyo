from flask import Flask, request, render_template, jsonify
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

class SyllabusVectorSearch:
    def __init__(self, json_path, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.json_path = json_path
        self.data = self._load_json()
        self.index, self.id_to_data = self._build_index()

    def _load_json(self):
        with open(self.json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_index(self):
        texts = [entry["description"] for entry in self.data]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        dim = embeddings.shape[1]

        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        id_to_data = {i: self.data[i] for i in range(len(self.data))}
        return index, id_to_data

    def search_vector(self, query, top_k=5):
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            result = self.id_to_data[idx]
            result["score"] = float(distances[0][i])
            results.append(result)

        return sorted(results, key=lambda x: x["score"], reverse=False)

    def search_exact(self, key, value):
        """完全一致検索（大文字小文字無視）"""
        value = value.lower()
        return [entry for entry in self.data if key in entry and str(entry[key]).lower() == value]

    def search_partial(self, key, value):
        """部分一致検索（大文字小文字無視）"""
        value = value.lower()
        return [entry for entry in self.data if key in entry and value in str(entry[key]).lower()]

    def search_list(self, key, values):
        """リスト検索（指定リストに1つでも該当するもの）"""
        values_set = set(v.strip() for v in values)
        return [entry for entry in self.data if key in entry and isinstance(entry[key], list) and values_set & set(entry[key])]

    def search_word(self, query):
        """単語検索（大文字小文字無視、キーワードやタイトル、説明を対象）"""
        words = [w.lower() for w in query.split()]
        results = []
        for entry in self.data:
            combined_text = " ".join([
                entry.get("title", ""),
                entry.get("keywordtexts", ""),
                entry.get("description", "")
            ]).lower()
            if any(word in combined_text for word in words):
                results.append(entry)
        return results

    def search_hyoka(self, eval_category, min_ratio):
        """成績評価検索"""
        return [entry for entry in self.data if "hyoka" in entry and entry["hyoka"].get(eval_category, 0) >= min_ratio]


# シラバス検索システムのセットアップ
searcher = SyllabusVectorSearch("syllabus_data.json")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query")
    search_type = request.form.get("searchType")
    list_query = request.form.get("listQuery")
    eval_category = request.form.get("evalCategory")
    eval_percentage = request.form.get("evalPercentage")

    if not query and not list_query and not eval_category:
        return jsonify({"error": "検索ワードまたは条件を入力してください。"})

    if search_type == "vector":
        results = searcher.search_vector(query, top_k=5)
    elif search_type == "exact":
        results = searcher.search_exact("title", query)
    elif search_type == "partial":
        results = searcher.search_partial("description", query)
    elif search_type == "list":
        results = searcher.search_list("yojigen", list_query.split(","))
    elif search_type == "word":
        results = searcher.search_word(query)
    elif search_type == "hyoka":
        if not eval_category or not eval_percentage:
            return jsonify({"error": "成績評価検索には評価基準とパーセンテージを入力してください。"})
        results = searcher.search_hyoka(eval_category, int(eval_percentage))
    else:
        results = {"error": "無効な検索タイプ"}

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)




