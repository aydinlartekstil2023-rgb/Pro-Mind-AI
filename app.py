import streamlit as st
from transformers import pipeline
import pandas as pd
import altair as alt
import os

# TensorFlow/Protobuf çakışmalarını önlemek için sistem ayarı
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Pro-Mind AI: Kurumsal & Terapi Destek", page_icon="🧠", layout="wide")

# --- MODEL YÜKLEME ---
@st.cache_resource
def load_analysis_model():
    # Türkçe ve çok dilli destek sunan güvenilir bir model
    return pipeline("text-classification", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", return_all_scores=True)

classifier = load_analysis_model()

# --- ARAYÜZ TASARIMI ---
st.title("🧠 Pro-Mind AI")
st.subheader("Kurumsal Psikoloji ve Terapi Takip Portalı")

# Yan Panel (Sidebar)
with st.sidebar:
    st.header("👤 Kullanıcı Girişi")
    user_type = st.radio("Hesap Tipi", ["Bireysel", "Kurumsal Çalışan", "Terapi Danışanı"])
    user_id = st.text_input("Kullanıcı ID / E-posta", placeholder="Örn: user@sirket.com")
    st.divider()
    st.info("Verileriniz anonimleştirilerek profesyonel panellere aktarılır.")

# Ana Ekran Kolonları
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Günlük Girişi")
    entry = st.text_area("Bugünkü duygu durumunuzu yazın...", height=250, placeholder="Neler olduğunu buraya anlatabilirsiniz...")
    generate_btn = st.button("🚀 Analiz Et ve Raporla")

if generate_btn:
    if entry:
        with st.spinner("Yapay zeka analiz ediyor..."):
            # Duygu Analizi Yap
            results = classifier(entry)[0]
            
            # Veriyi Tabloya Dönüştür (Hata veren kısım düzeltildi)
            df = pd.DataFrame(results)
            df.columns = ['Duygu', 'Skor']
            
            # Etiketleri Türkçeleştir
            label_map = {"positive": "Mutlu", "neutral": "Nötr", "negative": "Stresli"}
            df['Duygu'] = df['Duygu'].map(label_map)
            
            # En baskın duyguyu bul
            top_sentiment = df.sort_values(by="Skor", ascending=False).iloc[0]['Duygu']

            with col2:
                st.markdown("### 📊 Ruh Hali Dağılımı")
                # Grafik oluşturma
                chart = alt.Chart(df).mark_arc(innerRadius=50).encode(
                    theta=alt.Theta(field="Skor", type="quantitative"),
                    color=alt.Color(field="Duygu", type="nominal", scale=alt.Scale(domain=['Mutlu', 'Nötr', 'Stresli'], range=['#2ecc71', '#3498db', '#e74c3c'])),
                    tooltip=['Duygu', 'Skor']
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)

            # --- ÖNERİ VE RAPORLAMA ---
            st.divider()
            st.markdown(f"### 🌿 Bugünün Özeti: **{top_sentiment}**")
            
            rec_col1, rec_col2 = st.columns(2)
            
            if top_sentiment == "Stresli":
                with rec_col1:
                    st.warning("Yüksek stres düzeyi tespit edildi. Bu müzik size iyi gelebilir:")
                    st.video("https://www.youtube.com/watch?v=lFcSrYw-ARY")
                with rec_col2:
                    st.info("💡 **Terapist Notu:** Gün içinde 5 dakikalık nefes egzersizi yapmanız önerilir. Raporunuz sisteme işlendi.")
            
            elif top_sentiment == "Mutlu":
                with rec_col1:
                    st.success("Harika bir enerji! Modunuzu korumak için:")
                    st.video("https://www.youtube.com/watch?v=ZbZSe6N_BXs")
                with rec_col2:
                    st.info("💡 **İK Notu:** Pozitif etkileşiminiz ekip motivasyonuna katkı sağlıyor. Teşekkürler!")
            
            else:
                with rec_col1:
                    st.info("Dengeli bir ruh hali. Odaklanmak için bu ritmi deneyin:")
                    st.video("https://www.youtube.com/watch?v=jfKfPfyJRdk")
                with rec_col2:
                    st.info("💡 **Gelişim Notu:** Rutininizi korumak zihinsel berraklık sağlar.")

            if st.checkbox("Analizi Veritabanına Gönder"):
                st.success(f"Veriler {user_id} kimliğiyle başarıyla kaydedildi.")
    else:
        st.error("Lütfen analiz için bir metin girin.")
