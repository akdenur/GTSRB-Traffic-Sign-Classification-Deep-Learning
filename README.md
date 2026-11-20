# GTSRB Trafik Levhası Sınıflandırma — Derin Öğrenme

**Alman Trafik Levhası Tanıma Veri Kümesi (GTSRB)** üzerinde sıfırdan tasarlanan **Özel CNN** (**%99,92 test doğruluğu**) ile transfer öğrenme tabanlı **MobileNetV2** (**%97,60 test doğruluğu**) mimarilerinin **Grad-CAM** açıklanabilirlik analizleriyle birlikte karşılaştırmalı incelenmesi.

## Proje Özeti

[Geleneksel Makine Öğrenmesi](https://github.com/akdenur/GTSRB-Traffic-Sign-Classification-Traditional-ML) çalışmasının (%98,72 doğruluk) devamı niteliğindeki bu projede, manuel öznitelik çıkarımının ulaştığı performans sınırını aşmak amacıyla uçtan uca derin öğrenme mimarileri geliştirilmiş ve karşılaştırılmıştır.

### Veri Kümesi ve Ön İşleme
- **Kaynak:** German Traffic Sign Recognition Benchmark (GTSRB)
- **Sınıf Sayısı:** 43 trafik levhası kategorisi
- **Toplam Görüntü:** `64x64x3` (RGB) boyutuna ölçeklendirilmiş 39.209 etiketli görüntü
- **Katmanlı Bölme (Stratified Split — %80 / %10 / %10):**
  - **Eğitim Kümesi:** 31.367 görüntü
  - **Doğrulama Kümesi:** 3.921 görüntü (`EarlyStopping` ve `ReduceLROnPlateau` kontrolü için)
  - **Bağımsız Test Kümesi:** 3.921 görüntü

## Model Mimarileri

1. **Özel CNN (~2,3M Parametre):**
   - `64x64` boyutundaki trafik levhası görüntüleri için özel olarak tasarlanmış 3 bloklu evrişimsel sinir ağı.
   - Batch Normalization, MaxPooling, Dropout ve tam bağlantılı katmanlarla ince rakam ve sembol detaylarını kaybetmeden hiyerarşik öznitelikler öğrenir.
2. **MobileNetV2 (Transfer Öğrenme):**
   - ImageNet üzerinde ön eğitimli (pre-trained) omurga yapısının 43 sınıflı trafik levhası sınıflandırma problemine uyarlanması.

## Karşılaştırmalı Sonuçlar

| Model | Yaklaşım | Test Doğruluğu | Eğitim Süresi | Çıkarım Süresi (Test Kümesi) |
| :--- | :--- | :---: | :---: | :---: |
| **Özel CNN** | Derin Öğrenme (Sıfırdan) | **%99,92** | 550,80 sn | **0,78 sn** |
| **SVM (Önceki Çalışma)** | Geleneksel ML (HOG + LBP + PCA) | %98,72 | — | — |
| **MobileNetV2** | Transfer Öğrenme (ImageNet) | %97,60 | 370,46 sn | 4,16 sn |

### Temel Bulgular ve Açıklanabilirlik
- **Özel CNN Neden MobileNetV2'den Daha Başarılı?** MobileNetV2'nin derin omurgası `224x224` çözünürlüklü ImageNet görüntüleri için optimize edilmiştir. `64x64` boyutundaki küçük trafik levhalarına uygulandığında, ardışık uzamsal indirgemeler (downsampling) sırasında ince rakam ve sembol detayları kaybolmaktadır. Özel CNN mimarisi ise bu detayları koruyarak **%99,92** doğruluğa ve çok daha hızlı çıkarım süresine (**0,78 sn**) ulaşmıştır.
- **Grad-CAM Görselleştirmeleri:** Sınıf Aktivasyon Haritaları, Özel CNN modelinin arka plan gürültüsü yerine doğrudan levha üzerindeki ayırt edici bölgelere (rakamlar, oklar ve iç semboller) odaklandığını doğrulamaktadır.

## Kullanılan Teknolojiler

- **Dil:** Python
- **Ortam:** Google Colab (GPU)
- **Kütüphaneler:** TensorFlow 2.21.0 (Keras API), NumPy, OpenCV, Scikit-learn, Matplotlib, Seaborn

## Depo Yapısı

```text
├── GTSRB_Deep_Learning_CNN_MobileNetV2.ipynb   # Model eğitimi, değerlendirme ve Grad-CAM notebook'u
├── GTSRB_Deep_Learning_Report.pdf              # IEEE formatında detaylı akademik proje raporu
└── README.md                                   # Proje dokümantasyonu
