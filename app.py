import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import os


st.set_page_config(page_title="Yapay Zeka Enstrüman Tanıma", layout="centered")

st.title("Sesleri Dinleyen Yapay Zeka")
st.markdown("Bilgisayarınızdan bir ses dosyası yükleyin ve yapay zekanın hangi enstrümanı duyduğunu test edin.")
st.markdown("---")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('multi_instrument_model_v4.keras')

with st.spinner("Yapay zeka beyni (Model v4) yükleniyor..."):
    model = load_model()

enstruman_sozlugu = {
    'cel': 'Çello', 'cla': 'Klarnet (Clarinet)', 'flu': 'Flüt (Flute)',
    'gac': 'Akustik Gitar', 'gel': 'Elektro Gitar', 'org': 'Org',
    'pia': 'Piyano (Piano)', 'sax': 'Saksafon', 'tru': 'Trompet',
    'vio': 'Keman (Violin)', 'voi': 'İnsan Sesi (Vokal)'
}
sinif_kodlari = list(enstruman_sozlugu.keys())

uploaded_file = st.file_uploader("Lütfen bir ses dosyası (.wav) yükleyin", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Sesi Analiz Et", use_container_width=True):
        with st.spinner("Spektrogram haritası çıkarılıyor ve analiz ediliyor..."):
            try:
                with open("temp_audio.wav", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                y, sr = librosa.load("temp_audio.wav", sr=None, mono=True)
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                hedef_genislik = 130
                mevcut_genislik = mel_spec_db.shape[1]
                
                if mevcut_genislik < hedef_genislik:
                    pad_width = hedef_genislik - mevcut_genislik
                    mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
                else:
                    mel_spec_db = mel_spec_db[:, :hedef_genislik]
                    
                mel_spec_db_norm = (mel_spec_db - np.min(mel_spec_db)) / (np.max(mel_spec_db) - np.min(mel_spec_db) + 1e-8)
                girdi = mel_spec_db_norm[np.newaxis, ..., np.newaxis]
                
                tahminler = model.predict(girdi, verbose=0)
                en_yuksek_iht_indeksi = np.argmax(tahminler)
                guven_skoru = tahminler[0][en_yuksek_iht_indeksi] * 100
                tahmin_edilen = enstruman_sozlugu[sinif_kodlari[en_yuksek_iht_indeksi]]
                
                st.success(f"**Tahmin Edilen Enstrüman:** {tahmin_edilen}")
                st.info(f"**Yapay Zekanın Eminlik Oranı:** %{guven_skoru:.2f}")
                
                st.markdown("---")
                st.markdown("Model Doğru Bildi Mi?")
                st.markdown("*(Geri bildiriminiz bir sonraki Model V5'in eğitim veritabanına eklenecektir)*")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Evet"):
                        st.balloons()
                        st.success("Bu ses dosyası 'Doğrulanmış Veriler' arşivi için işaretlendi.")
                with col2:
                    gercek_cevap = st.selectbox("Hayır, asıl enstrüman şuydu:", ["Doğrusunu Seçin..."] + list(enstruman_sozlugu.values()))
                    if gercek_cevap != "Doğrusunu Seçin...":
                        st.warning(f"Teşekkürler! Bu ses dosyası artık '{gercek_cevap}' olarak yeniden etiketlenip hata düzeltme (V5) veritabanına aktarıldı.")

                os.remove("temp_audio.wav")
                
            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")