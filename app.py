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
        
        # Hata ihtimalini sıfıra indirmek için 'latest' takısını ekledik
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        prompt = f"""
        Bir psikolojik danışman ve mentor rolündesin. 
        Kullanıcı günlük girişinde şunları yazdı: "{user_text}"
        Duygu analizi sonucu: {sentiment}.
        Bu kullanıcıya moduna uygun, samimi, destekleyici ve motive edici bir tavsiye ver. 
        Yanıtın en fazla 3 cümle olsun ve doğrudan kullanıcıya hitap et.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # Eğer flash-latest da hata verirse, klasik gemini-pro'ya dön
        try:
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return response.text
        except Exception as inner_e:
            return f"Mentor şu an meşgul, ama duyguların değerli. (Teknik Detay: {str(inner_e)})"

# --- MODELLER ---
@st.cache_resource
def load_models():
    return pipeline("text-classification", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", return_all_scores=True)

classifier = load_models()

# --- ARAYÜZ ---
st.title("🧠 Pro-Mind AI: Akıllı Mentor")

with st.sidebar:
    st.header("👤 Kullanıcı Girişi")
    user_type = st.radio("Hesap Tipi", ["Bireysel", "Kurumsal", "Terapi"])
    user_id = st.text_input("E-posta", value="user@domain.com")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Bugün Neler Oldu?")
    entry = st.text_area("Duygularınızı paylaşın...", height=200, placeholder="Bugün nasıl hissediyorsun?")
    generate_btn = st.button("🚀 Analiz Et ve Mentoruma Sor")

if generate_btn and entry:
    with st.spinner("AI Mentorunuz analiz ediyor ve yazıyor..."):
        # 1. Duygu Analizi
        prediction = classifier(entry)
        raw_results = prediction[0] if isinstance(prediction[0], list) else prediction
        
        data_list = [{"Duygu": res.get('label'), "Score": res.get('score')} for res in raw_results]
        df = pd.DataFrame(data_list)
        label_map = {"positive": "Mutlu", "neutral": "Nötr", "negative": "Stresli"}
        df['Duygu'] = df['Duygu'].map(label_map).fillna(df['Duygu'])
        top_sentiment = df.sort_values(by="Score", ascending=False).iloc[0]['Duygu']

        # 2. Gemini'den Tavsiye Al
        ai_advice = get_gemini_response(entry, top_sentiment)

        # 3. Görselleştirme
        with col2:
            st.markdown("### 📊 Ruh Hali Analizi")
            chart = alt.Chart(df).mark_arc(innerRadius=50).encode(
                theta=alt.Theta(field="Score", type="quantitative"),
                color=alt.Color(field="Duygu", type="nominal", scale=alt.Scale(
                    domain=['Mutlu', 'Nötr', 'Stresli'], range=['#2ecc71', '#3498db', '#e74c3c'])),
            ).properties(height=250)
            st.altair_chart(chart, use_container_width=True)

        st.divider()
        
        # 4. Sonuç Ekranı
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.markdown(f"#### Baskın Duygu: **{top_sentiment}**")
            if top_sentiment == "Stresli": st.error("💡 Biraz mola vermeye ne dersin?")
            elif top_sentiment == "Mutlu": st.success("🌟 Harika enerjini koru!")
            else: st.info("🧘‍♂️ Dengeli bir gün.")
        
        with res_col2:
            st.markdown("#### ✨ AI Mentorunuzun Tavsiyesi")
            st.write(ai_advice)
