# scripts/extract_and_summarize.py
import fitz  # PyMuPDF
import re
import json
from tqdm import tqdm
import google.generativeai as genai

genai.configure(api_key="AIzaSyCCJ7i7peCIUK3ff4g0YETeEy0wBcSKPwI")  # ← APIキー設定しておくこと
model = genai.GenerativeModel("models/gemini-1.5-pro")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    texts = [page.get_text() for page in doc]
    return "\n".join(texts)

def split_lectures(text):
    pattern = r"(?:授業科目名)?\s*([^\n]{5,40})\n+([^\n]*教員)?"
    matches = list(re.finditer(pattern, text))
    lectures = []
    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        lectures.append(section_text)
    return lectures

def summarize_lecture(text):
    prompt = (
        "以下は大学の講義情報が含まれるテキストです。この内容から、授業名・教員名・授業概要・評価方法・教科書・曜日時限を抽出し、以下の形式でJSONにしてください。\n"
        "出力は必ず **JSON形式のみ** にしてください。説明やマークダウンなどは不要です。\n"
        "出力形式：\n"
        "{\n"
        "  \"title\": \"...\",\n"
        "  \"professors\": \"...\",\n"
        "  \"description\": \"...\",\n"
        "  \"evaluation\": { \"平常点\": 50, \"期末試験\": 50 },\n"
        "  \"textbook\": \"...\",\n"
        "  \"yojigen\": \"...\"\n"
        "}\n"
        "----\n"
        f"{text}\n"
    )

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()

        # JSON以外のマークダウンとかが混ざる可能性を排除
        if "```json" in raw:
            raw = raw.split("```json")[-1].split("```")[0].strip()

        if not raw or raw[0] not in ['{', '[']:
            raise ValueError("返ってきたデータがJSONっぽくない！\n" + raw)

        return json.loads(raw)

    except Exception as e:
        print("❌ 要約失敗:", e)
        return {
            "title": "要約エラー",
            "professors": "",
            "description": "この講義は要約に失敗しました。",
            "evaluation": {},
            "textbook": "",
            "yojigen": ""
        }



def process_pdf_to_json(pdf_path, output_path="data/json/2025_summary.json"):
    raw_text = extract_text_from_pdf(pdf_path)
    sections = split_lectures(raw_text)

    summaries = []
    for section in tqdm(sections, desc="要約中"):
        result = summarize_lecture(section)
        if result:
            summaries.append(result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print(f"✅ 要約完了！→ {output_path}")
    return output_path
