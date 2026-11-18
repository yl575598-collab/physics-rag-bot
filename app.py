import streamlit as st
import os
from PyPDF2 import PdfReader
import chromadb
from google import genai

st.set_page_config(page_title="高一物理 AI 答題助手", page_icon="⚡", layout="centered")
st.title("⚡ 高一物理 AI 答題小幫手")
st.caption("基於 RAG 技術與 Google Gemini API，精準依據講義回答問題。")

# 側邊欄設定
with st.sidebar:
    st.header("⚙️ 系統設定")
    api_key = st.text_input("請輸入 Google Gemini API Key", type="password")
    uploaded_file = st.file_uploader("可選：自訂上傳其他 PDF 講義", type=["pdf"])

def process_pdf(file_source):
    """可同時處理檔案物件或本機/GitHub 上的檔案路徑"""
    reader = PdfReader(file_source)
    text = ""
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += f"\n[頁碼 {i+1}] " + page_text.replace("\n", " ").strip()
    return text

def chunk_text(text, chunk_size=300, chunk_overlap=40):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += (chunk_size - chunk_overlap)
    return chunks

@st.cache_resource
def build_vector_db(_chunks):
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="physics_notes_app")
    ids = [f"chunk_{i}" for i in range(len(_chunks))]
    collection.add(documents=_chunks, ids=ids)
    return collection

# 初始化歷史對話
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！請在左側輸入 API Key，即可開始詢問《高一物理重點筆記》相關問題囉！"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input("請輸入物理問題（例如：電梯加速上升時視重如何變化？）"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    if not api_key:
        st.error("請先在左側邊欄輸入 Gemini API Key！")
    else:
        # 決定 PDF 來源：若有手動上傳則用上傳檔，否則自動讀取預設的 physics.pdf
        pdf_source = None
        if uploaded_file:
            pdf_source = uploaded_file
        elif os.path.exists("physics.pdf"):
            pdf_source = "physics.pdf"

        if not pdf_source:
            st.error("❌ 找不到物理講義！請在左側上傳 PDF 檔，或確認 GitHub 上已有 physics.pdf。")
        else:
            with st.spinner("🔍 正在檢索講義並生成精準回答..."):
                try:
                    raw_text = process_pdf(pdf_source)
                    chunks = chunk_text(raw_text)
                    collection = build_vector_db(chunks)

                    results = collection.query(query_texts=[user_input], n_results=1)
                    retrieved_context = results['documents'][0][0]

                    client = genai.Client(api_key=api_key)
                    prompt = f"""
你是一個嚴謹的高中物理答題助手。請「僅依據」下方提供的物理講義內容回答問題。
若講義無記載，請明確回答：「根據提供的物理講義，無法回答此問題。」

【物理講義內容】
{retrieved_context}

【學生提問】
{user_input}
"""
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt
                    )

                    final_answer = response.text + f"\n\n---\n📌 **參考講義引據片段：**\n`{retrieved_context[:120]}...`"
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})
                    st.chat_message("assistant").write(final_answer)

                except Exception as e:
                    st.error(f"執行時發生錯誤：{e}")
