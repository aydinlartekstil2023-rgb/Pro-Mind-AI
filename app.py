import streamlit as st
from transformers import pipeline
import pandas as pd
import altair as alt
import os
import google.generativeai as genai

# Sistem Ayarları
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Pro-Mind AI: Akıllı Mentor", page_icon="🧠", layout="wide")

# --- GEMINI KONFİGÜRASYONU (DİNAMİK MODEL SEÇİCİ) ---
def get_gemini_response(user_text, sentiment):
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        
        # MEVCUT MODELLERİ LİSTELE VE UYGUN OLANI SEÇ
        # Bu kısım 404 hatasını tamamen ortadan kaldırır
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # Tercih sırasına göre modeller
        preferred_models = ['models/gemini-1.5-flash', 'models/gemini-pro', 'models/gemini-1.5-flash-latest']
        
        selected_model = None
        for pm in preferred_models:
            if pm in available_models:
                selected_model = pm
                break
        
        if not selected_model:
            selected_model = available_models[0] # Hiçbiri yoksa ilk bulduğunu al

        model = genai.GenerativeModel(model_name=selected_model)
        
        prompt = f"""
        Rolün: Profesyonel bir psikolojik danışman ve mentor.
        Kullanıcı Girişi: "{user_text}"
        Tespit Edilen Duygu: {sentiment}.
        Görev: Kullanıcıya bu moduna uygun, empati kuran, samimi ve yapıcı bir tavsiye ver. 
        Yanıtın 3 cümleyi geçmesin.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Mentor şu an derin düşüncelerde... (Teknik detay: {str(e)})"

# --- DUYGU ANALİZİ MODELİ ---
@st.cache_resource
def load_models():
    return pipeline("text-classification", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", return_all_scores=True)

classifier = load_models()

# --- ARAYÜZ ---
st.title("🧠 Pro-Mind AI: Akıllı Mentor")

with st.sidebar:
    st.header("👤 Kullanıcı Paneli")
    user_id = st.text_input("E-posta", value="danisan@akademi.com")
    st.divider()
    st.info("💡 AI Mentor, duygularınızı analiz eder ve size rehberlik eder.")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Günlük Girişi")
    entry = st.text_area("Bugün neler hissettiğini anlatmak ister misin?", height=200)
    generate_btn = st.button("✨ Analiz Et ve Akıl Danış")

if generate_btn and entry:
    with st.spinner("Mentorun mesajını hazırlıyor..."):
        prediction = classifier(entry)
        raw_results = prediction[0] if isinstance(prediction[0], list) else prediction
        
        data_list = [{"Duygu": res.get('label'), "Score": res.get('score')} for res in raw_results]
        df = pd.DataFrame(data_list)
        
        label_map = {"positive": "Mutlu", "neutral": "Nötr", "negative": "Stresli"}
        df['Duygu'] = df['Duygu'].map(label_map).fillna(df['Duygu'])
        top_sentiment = df.sort_values(by="Score", ascending=False).iloc[0]['Duygu']

        ai_advice = get_gemini_response(entry, top_sentiment)

        with col2:
            st.markdown("### 📊 Duygu Dağılımı")
            chart = alt.Chart(df).mark_arc(innerRadius=50).encode(
                theta=alt.Theta(field="Score", type="quantitative"),
                color=alt.Color(field="Duygu", type="nominal", scale=alt.Scale(
                    domain=['Mutlu', 'Nötr', 'Stresli'], range=['#2ecc71', '#3498db', '#e74c3c'])),
            ).properties(height=250)
            st.altair_chart(chart, use_container_width=True)

        st.divider()
        
        m_col1, m_col2 = st.columns([1, 2])
        with m_col1:
            st.info(f"🔍 Baskın Mod: **{top_sentiment}**")
        with m_col2:
            st.markdown("#### 💬 Mentorun Mesajı")
            st.success(ai_advice)
