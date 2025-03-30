import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# JSONファイルの読み込み
with open("../data/json/ocr_result.json", "r") as f:
    syllabus_data = json.load(f)

# 埋め込みモデルの読み込み
model = SentenceTransformer("all-MiniLM-L6-v2")

# テキストデータの抽出と埋め込み
texts = [entry["content"] for entry in syllabus_data]
embeddings = model.encode(texts)

# FAISS のインデックス作成
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# FAISS インデックス保存
faiss.write_index(index, "../data/faiss/syllabus_index.faiss")

print("✅ FAISS インデックス作成完了！")
