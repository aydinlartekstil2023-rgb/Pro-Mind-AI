import streamlit as st
from transformers import pipeline
import pandas as pd
import altair as alt

# --- AYARLAR VE MODEL ---
st.set_page_config(page_title="Pro-Mind AI: Kurumsal & Terapi Destek", page_icon="🧠", layout="wide")

@st.cache_resource
def load_analysis_model():
    # Çok dilli duygu analiz modeli
    return pipeline("text-classification", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", return_all_scores=True)

classifier = load_analysis_model()

# --- ARAYÜZ ---
st.title("🧠 Pro-Mind AI")
st.subheader("Kurumsal Psikoloji ve Terapi Takip Portalı")

# Giriş Paneli (Kurumsal/Terapi ayrımı için)
with st.sidebar:
    st.header("👤 Kullanıcı Girişi")
    user_type = st.radio("Hesap Tipi", ["Bireysel", "Kurumsal Çalışan", "Terapi Danışanı"])
    user_id = st.text_input("Kullanıcı ID / E-posta")
    st.divider()
    st.info("Bu veriler şifrelenerek ilgili danışman/yönetici paneline aktarılır.")

# Günlük ve Analiz Bölümü
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Günlük Girişi")
    entry = st.text_area("Bugünkü duygu durumunuzu detaylandırın...", height=250)
    generate_btn = st.button("Analiz Et ve Raporla")

if generate_btn and entry:
    results = classifier(entry)[0]
    df = pd.DataFrame(results)
    df.columns = ['Duygu', 'Skor']
    label_map = {"positive": "Mutlu", "neutral": "Nötr", "negative": "Stresli"}
    df['Duygu'] = df['Duygu'].map(label_map)
    
    top_sentiment = df.sort_values(by="Skor", ascending=False).iloc[0]['Duygu']

    with col2:
        st.markdown("### 📊 Ruh Hali Analizi")
        chart = alt.Chart(df).mark_arc().encode(
            theta=alt.Theta(field="Skor", type="quantitative"),
            color=alt.Color(field="Duygu", type="nominal"),
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

    # --- ÖNERİ SİSTEMİ (PARA KAZANDIRAN KISIM) ---
    st.divider()
    st.markdown("### 🌿 Size Özel Destek Önerileri")
    
    rec_col1, rec_col2 = st.columns(2)
    
    if top_sentiment == "Stresli":
        with rec_col1:
            st.warning("Yüksek stres tespit edildi. İşte sizin için seçtiğimiz rahatlatıcı müzik:")
            st.video("https://www.youtube.com/watch?v=lFcSrYw-ARY") # Örnek Lofi videosu
        with rec_col2:
            st.info("💡 **Terapi Notu:** Derin nefes egzersizi yapmayı deneyin. Bu rapor terapistinize iletildi.")
            
    elif top_sentiment == "Mutlu":
        with rec_col1:
            st.success("Pozitif enerji! Bu modu kutlamak için enerjik bir şarkı:")
            st.video("https://www.youtube.com/watch?v=ZbZSe6N_BXs") # Happy - Pharrell
        with rec_col2:
            st.info("💡 **Kurumsal Not:** Harika bir verimlilik günündesiniz! Ekibinizle bu enerjiyi paylaşın.")
    
    else:
        st.info("Dengeli bir gün. Odaklanmanıza yardımcı olacak bir çalışma müziği ister misiniz?")
        st.video("https://www.youtube.com/watch?v=jfKfPfyJRdk")

# --- VERİ TABANI SİMÜLASYONU (KURUMSAL İÇİN) ---
if st.checkbox("Yönetici/Terapist Paneline Gönder"):
    st.toast(f"Veriler {user_id} kimliğiyle güvenli sunucuya iletildi.")