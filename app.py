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

# --- GEMINI KONFİGÜRASYONU ---
def get_gemini_response(user_text, sentiment):
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        
        # 404 HATASINI ÇÖZEN KRİTİK DEĞİŞİKLİK: 
        # En güncel model isimlendirmesini kullanıyoruz
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')
        
        prompt = f"""
        Rolün: Profesyonel bir psikolojik danışman ve mentor.
        Kullanıcı Girişi: "{user_text}"
        Tespit Edilen Duygu: {sentiment}.
        Görev: Kullanıcıya bu moduna uygun, empati kuran, 
        samimi ve yapıcı bir tavsiye ver. Yanıtın 3 cümleyi geçmesin.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # Hata devam ederse alternatif model ismini dene
        try:
            model = genai.GenerativeModel(model_name='models/gemini-1.5-flash')
            response = model.generate_content(prompt)
            return response.text
        except:
            return f"Mentor şu an derin düşüncelerde... (Teknik detay: {str(e)})"

# --- DUYGU ANALİZİ MODELİ ---
@st.cache_resource
def load_models():
    # Çok dilli (Türkçe destekli) duygu analiz modeli
    return pipeline("text-classification", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", return_all_scores=True)

classifier = load_models()

# --- ARAYÜZ ---
st.title("🧠 Pro-Mind AI: Akıllı Mentor")

with st.sidebar:
    st.header("👤 Kullanıcı Paneli")
    user_id = st.text_input("E-posta", value="danisan@akademi.com")
    st.divider()
    st.markdown("💡 *AI Mentor, yazdıklarınızı analiz eder ve size özel bir yol haritası sunar.*")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Günlük Girişi")
    entry = st.text_area("Bugün neler hissettiğini anlatmak ister misin?", height=200)
    generate_btn = st.button("✨ Analiz Et ve Akıl Danış")

if generate_btn and entry:
    with st.spinner("Mentorun mesajını hazırlıyor..."):
        # 1. Duygu Analizi Süreci
        prediction = classifier(entry)
        raw_results = prediction[0] if isinstance(prediction[0], list) else prediction
        
        data_list = [{"Duygu": res.get('label'), "Score": res.get('score')} for res in raw_results]
        df = pd.DataFrame(data_list)
        
        label_map = {"positive": "Mutlu", "neutral": "Nötr", "negative": "Stresli"}
        df['Duygu'] = df['Duygu'].map(label_map).fillna(df['Duygu'])
        top_sentiment = df.sort_values(by="Score", ascending=False).iloc[0]['Duygu']

        # 2. Gemini Mentor Tavsiyesi
        ai_advice = get_gemini_response(entry, top_sentiment)

        # 3. Grafik ve Görselleştirme
        with col2:
            st.markdown("### 📊 Duygu Dağılımı")
            chart = alt.Chart(df).mark_arc(innerRadius=50).encode(
                theta=alt.Theta(field="Score", type="quantitative"),
                color=alt.Color(field="Duygu", type="nominal", scale=alt.Scale(
                    domain=['Mutlu', 'Nötr', 'Stresli'], range=['#2ecc71', '#3498db', '#e74c3c'])),
            ).properties(height=250)
            st.altair_chart(chart, use_container_width=True)

        st.divider()
        
        # 4. Mentor Paneli
        m_col1, m_col2 = st.columns([1, 2])
        with m_col1:
            st.info(f"🔍 Baskın Mod: **{top_sentiment}**")
        with m_col2:
            st.markdown("#### 💬 Mentorun Mesajı")
            st.success(ai_advice)
