import streamlit as st
from transformers import pipeline
import pandas as pd
import altair as alt
import os

# Sistem ayarları
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Pro-Mind AI: Kurumsal & Terapi Destek", page_icon="🧠", layout="wide")

# --- MODEL YÜKLEME ---
@st.cache_resource
def load_analysis_model():
    # Çok dilli duygu analiz modeli
    return pipeline("text-classification", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", return_all_scores=True)

classifier = load_analysis_model()

# --- ARAYÜZ ---
st.title("🧠 Pro-Mind AI")
st.subheader("Kurumsal Psikoloji ve Terapi Takip Portalı")

with st.sidebar:
    st.header("👤 Kullanıcı Girişi")
    user_type = st.radio("Hesap Tipi", ["Bireysel", "Kurumsal Çalışan", "Terapi Danışanı"])
    user_id = st.text_input("Kullanıcı ID / E-posta", placeholder="user@domain.com")
    st.divider()
    st.info("Verileriniz anonimleştirilerek profesyonel panellere aktarılır.")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Günlük Girişi")
    entry = st.text_area("Bugünkü duygu durumunuzu yazın...", height=250)
    generate_btn = st.button("🚀 Analiz Et ve Raporla")

if generate_btn:
    if entry:
        with st.spinner("Analiz ediliyor..."):
            # Modelden gelen ham veriyi al
            raw_results = classifier(entry)[0]
            
            # --- HATAYI KESİN ÇÖZEN YENİ MANTIK ---
            # Veriyi manuel olarak listelere ayırıp DataFrame oluşturuyoruz
            labels = [res['label'] for res in raw_results]
            scores = [res['score'] for res in raw_results]
            
            df = pd.DataFrame({
                'Duygu': labels,
                'Skor': scores
            })
            
            # Etiketleri Türkçeleştir
            label_map = {"positive": "Mutlu", "neutral": "Nötr", "negative": "Stresli"}
            df['Duygu'] = df['Duygu'].map(label_map)
            
            # En yüksek skorlu duyguyu bul
            top_sentiment = df.sort_values(by="Skor", ascending=False).iloc[0]['Duygu']

            with col2:
                st.markdown("### 📊 Ruh Hali Analizi")
                # Grafik oluşturma
                chart = alt.Chart(df).mark_arc(innerRadius=50).encode(
                    theta=alt.Theta(field="Skor", type="quantitative"),
                    color=alt.Color(field="Duygu", type="nominal", scale=alt.Scale(
                        domain=['Mutlu', 'Nötr', 'Stresli'], 
                        range=['#2ecc71', '#3498db', '#e74c3c']
                    )),
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)

            st.divider()
            st.markdown(f"### 🌿 Bugünün Özeti: **{top_sentiment}**")
            
            rec_col1, rec_col2 = st.columns(2)
            
            if top_sentiment == "Stresli":
                with rec_col1:
                    st.warning("Stresli görünüyorsunuz. Bu müzik size iyi gelebilir:")
                    st.video("https://www.youtube.com/watch?v=lFcSrYw-ARY")
                with rec_col2:
                    st.info("💡 **Terapist Notu:** Kısa bir yürüyüş zihninizi boşaltmanıza yardımcı olabilir.")
            
            elif top_sentiment == "Mutlu":
                with rec_col1:
                    st.success("Harika bir gün! Bu enerjiyi kutlayalım:")
                    st.video("https://www.youtube.com/watch?v=ZbZSe6N_BXs")
                with rec_col2:
                    st.info("💡 **İK Notu:** Bu pozitif enerjiyi çalışma arkadaşlarınızla paylaşın!")
            
            else:
                with rec_col1:
                    st.info("Sakin ve dengeli bir gün. Odaklanmak için:")
                    st.video("https://www.youtube.com/watch?v=jfKfPfyJRdk")
                with rec_col2:
                    st.info("💡 **Gelişim Notu:** Rutininize sadık kalmak başarıyı getirir.")
    else:
        st.error("Lütfen bir metin girin!")
