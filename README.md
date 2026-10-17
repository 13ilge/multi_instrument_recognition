# Çoklu Enstrüman Tanıma (Multi-Instrument Recognition)

IRMAS veri seti ile 11 enstrüman sınıfını mel-spektrogram + CNN kullanarak tanıyan derin öğrenme projesi.

## Veri Seti

- **Kaynak:** IRMAS (Instrument Recognition in Musical Audio Signals) — Music Technology Group (MTG), Universitat Pompeu Fabra (UPF)
- **Bağlantı:** https://www.upf.edu/web/mtg/irmas
- **Zenodo:** https://doi.org/10.5281/zenodo.1290750
- **Lisans:** CC BY-NC-SA 3.0 — yalnızca ticari olmayan kullanım için ücretsizdir; yeniden dağıtılamaz ve değiştirilemez.
- **Referans:** Bosch, J. J., Janer, J., Fuhrmann, F., & Herrera, P. (2012). "A Comparison of Sound Segregation Techniques for Predominant Instrument Recognition in Musical Audio Signals", ISMIR, 559-564.
- **İçerik:** 6705 adet 3 saniyelik, 16-bit, 44.1 kHz WAV kaydı; 11 sınıf (çello, klarnet, flüt, akustik gitar, elektro gitar, org, piyano, saksafon, trompet, keman, insan sesi).

## Dosyalar

| Dosya | Açıklama |
|---|---|
| `01_veri_kesfi.ipynb` | Problem tanımı, veri keşfi (dalga formu, mel-spektrogram, sınıf dağılımı) ve ön işleme |
| `02_model_egitimi.ipynb` | CNN model varyantları (v1–v4), eğitim, değerlendirme, hata analizi |
| `03_test_ve_tahmin.ipynb` | Tek ses dosyası üzerinde çıkarım testi |
| `app.py` | Streamlit ile canlı demo |
| `multi_instrument_model_v4.keras` | Uygulamada kullanılan model |

## Kurulum ve Çalıştırma

```bash
python -m venv venv
venv\Scripts\activate
pip install tensorflow librosa streamlit matplotlib seaborn scikit-learn
streamlit run app.py
```
