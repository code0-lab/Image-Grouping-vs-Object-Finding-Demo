# Image Clustering + Car Check (Streamlit)

Bu uygulama, birden çok resmi yükleyip HOG + renk histogramı özellikleriyle KMeans kümeleri oluşturan ve isteğe bağlı olarak YOLOv8 ile “car” sınıfını tespit eden bir Streamlit arayüzüdür.

## Özellikler
- Çoklu görüntü yükleme ve otomatik özellik çıkarımı
- KMeans ile kümeleme; küme sayısı `K` ayarlanabilir
- HOG parametreleri (`pixels_per_cell`, `cells_per_block`) ve renk histogram `bins` ayarlanabilir
- Opsiyonel YOLOv8 “car” tespiti ve kümeler için “araba oranı” özeti
- Tek görüntü üzerinde hızlı “Araba mı?” kontrolü

## Kurulum
Sanal ortam önerilir:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
```

Bağımlılıklar:
```bash
pip install streamlit pillow numpy scikit-image scikit-learn ultralytics
```
YOLOv8 opsiyoneldir; yalnızca araba tespiti için gereklidir.

## Çalıştırma
```bash
streamlit run app.py
```

YOLO ağırlığı (opsiyonel):
- `yolov8n.pt` dosyasını `app.py` ile aynı dizine koyarsanız yerel dosya kullanılır.
- Dosya yoksa `ultralytics` paketinin varsayılan modeli indirilmeye çalışılır.

## Kullanım
1. Kenar çubuğundan ayarları yapın:
   - `K` (2–10): Küme sayısı
   - `HOG pixels_per_cell`: 8/12/16/24
   - `HOG cells_per_block`: 1x1 / 2x2 / 3x3
   - `Renk histogram bins`: 32 / 64 / 128
   - `Araba tespiti (YOLOv8)` ve `YOLO confidence`
2. “Birden çok resim yükleyin” alanından `jpg/jpeg/png` dosyaları seçin.
3. Kümeleri ve (aktifse) araba oranı özetlerini inceleyin.
4. Alt bölümde “Tek Resim: Araba mı?” ile tek resim kontrolü yapın.

## Mimarî
- `load_image(...)`: Dosyadan RGB görüntü yükleme (Pillow)
- `extract_features(...)`: HOG (grayscale) + R/G/B histogram birleştirme
- `cluster_images(...)`: StandardScaler + KMeans
- `detect_car(...)`: Ultralytics YOLOv8 ile nesne sınıfı (car) tespiti
- `show_clusters(...)`: Küme görselleştirme ve araba oranı özeti
- `main(...)`: Streamlit arayüz akışı

## İpuçları
- En az 2 görüntü ile kümeleme daha anlamlı sonuç verir.
- `K > N` durumunda `K`, otomatik olarak `N`’e düşürülür.
- YOLO kurulu değilse arayüz uyarı verir ve tespit devre dışıdır.

## Repo ve Gitignore
- Takip edilen: `app.py`, `REEDME`, `.gitignore`, `yolov8n.pt`
- Hariç tutulan: `.venv/`, `.venv311/`, `.env*`, `.streamlit/`, `__pycache__/`, `*.log`, `*.tmp`, `.vscode/`, `.idea/`, `*.pt` (yalnızca `yolov8n.pt` dahil), `*.onnx`, `*.pth`, `*.weights`, `Deneme resimleri/`, `uploads/`, `.DS_Store`

## Ignore edilen öğeleri nasıl temin ederim?
- Sanal ortam: `python -m venv .venv && source .venv/bin/activate` (Windows: ` .\.venv\Scripts\activate`)
- Bağımlılıklar: `pip install streamlit pillow numpy scikit-image scikit-learn ultralytics`
- YOLO ağırlıkları:
  - `yolov8n.pt` zaten repoda; başka ağırlık kullanacaksanız dosyayı `app.py` ile aynı dizine koyun.
  - Alternatif: `ultralytics` ilk kullanımda `yolov8n.pt`’yi otomatik indirir.
- Örnek veri: `Deneme resimleri/` klasörünü yerelde oluşturup içine `jpg/jpeg/png` görseller ekleyin (uygulama için zorunlu değil; yükleyiciyle kendi görsellerinizi seçebilirsiniz).
- Streamlit yapılandırması (opsiyonel): `mkdir .streamlit` oluşturup gerekirse `config.toml` ekleyin; uygulama varsayılanlarla da çalışır.
- `uploads/` (opsiyonel): Sunucu tarafı dosya saklama ihtiyacınız olursa klasörü yerelde oluşturun.
- `.env` (opsiyonel): Şimdiki kod `.env` okumuyor; ileride gizli anahtarlar için `python-dotenv` veya ortam değişkenleri kullanabilirsiniz.
