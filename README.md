# 🍎 Meyve Tanıma Uygulaması

Bu uygulama, yapay zeka kullanarak meyve görüntülerini tanımlayan bir web uygulamasıdır. EfficientNetV2B3 modeli kullanılarak geliştirilmiştir.

## 🌟 Özellikler

- Yüksek doğruluk oranı ile meyve tanıma
- Gelişmiş görüntü işleme
- Gerçek zamanlı sınıflandırma
- Detaylı performans metrikleri
- Kullanıcı dostu arayüz

## 📋 Gereksinimler

- Python 3.8+
- Streamlit
- TensorFlow
- OpenCV
- Pillow
- NumPy
- Plotly

## 🚀 Kurulum

1. Projeyi klonlayın:
```bash
git clone https://github.com/ueculfa/meyve-tanima.git
cd meyve-tanima
```

2. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

3. Uygulamayı çalıştırın:
```bash
streamlit run app.py
```

## 💻 Kullanım

1. Uygulamayı başlattıktan sonra tarayıcınızda `http://localhost:8501` adresine gidin
2. "Bir meyve görüntüsü seçin" butonuna tıklayın
3. Bir meyve fotoğrafı yükleyin
4. "Görüntüyü Tanı" butonuna tıklayın
5. Sonuçları görüntüleyin

## 📊 Performans Metrikleri

- Top-1 Doğruluk: %85
- Top-3 Doğruluk: %97
- Meyve Tanıma: %96
- Genel Nesne Tanıma: %90

## 🔧 Teknik Detaylar

- Model: EfficientNetV2B3
- Görüntü Boyutu: 300x300
- Minimum Güven Skoru: %10
- Maksimum Görüntü Boyutu: 1024px

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 👥 Katkıda Bulunma

1. Bu depoyu fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/yeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik: Açıklama'`)
4. Branch'inizi push edin (`git push origin feature/yeniOzellik`)
5. Bir Pull Request oluşturun 

git remote set-url origin https://github.com/ueculfa/meyve-tanima.git
git push -u origin main 