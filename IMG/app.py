import io
from typing import Optional, List, Tuple

import numpy as np
from PIL import Image
import streamlit as st

# Clustering
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# YOLO (opsiyonel)
def _lazy_load_yolo():
    import os
    local_weights = os.path.join(os.path.dirname(__file__), "yolov8n.pt")
    try:
        from ultralytics import YOLO
    except Exception:
        return None
    if os.path.exists(local_weights):
        return YOLO(local_weights)
    try:
        return YOLO("yolov8n.pt")
    except Exception:
        return None


def load_image(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def extract_features(img: np.ndarray, hog_pixels_per_cell=(16, 16), hog_cells_per_block=(2, 2), hist_bins=64, resize_to: int = 256) -> np.ndarray:
    """
    Gömleme: HOG (grayscale) + renk histogramı (R,G,B)
    img: RGB uint8 [H,W,3]
    Sabit boyuta (resize_to x resize_to) getirerek vektör uzunluklarını eşitliyoruz.
    """
    if resize_to is not None:
        pil = Image.fromarray(img)
        pil = pil.resize((resize_to, resize_to))
        img = np.array(pil)

    gray = rgb2gray(img)
    hog_vec = hog(
        gray,
        pixels_per_cell=hog_pixels_per_cell,
        cells_per_block=hog_cells_per_block,
        feature_vector=True,
    )

    hist_r, _ = np.histogram(img[:, :, 0], bins=hist_bins, range=(0, 255), density=True)
    hist_g, _ = np.histogram(img[:, :, 1], bins=hist_bins, range=(0, 255), density=True)
    hist_b, _ = np.histogram(img[:, :, 2], bins=hist_bins, range=(0, 255), density=True)

    feat = np.concatenate([hog_vec, hist_r, hist_g, hist_b]).astype(np.float32)
    return feat


def cluster_images(features: np.ndarray, n_clusters: int, random_state: int = 42) -> Tuple[np.ndarray, KMeans]:
    """
    features: [N, D]
    return: labels [N], fitted KMeans
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    return labels, km


def detect_car(pil_img: Image.Image, conf_threshold: float = 0.25) -> Tuple[Optional[bool], Optional[List[str]]]:
    """
    YOLOv8 ile araba var mı? (opsiyonel)
    ultralytics yoksa (None, None) döner.
    """
    if "yolo_model" not in st.session_state:
        st.session_state["yolo_model"] = _lazy_load_yolo()
    model = st.session_state["yolo_model"]
    if model is None:
        return None, None

    img_np = np.array(pil_img)
    results = model.predict(img_np, conf=conf_threshold, verbose=False)
    result = results[0]
    boxes = result.boxes
    names = model.names
    if boxes is None or boxes.cls is None or len(boxes) == 0:
        return False, []

    classes = boxes.cls.cpu().numpy().astype(int)
    detected_names = [names[c] for c in classes]
    is_car = any(n == "car" for n in detected_names)
    return is_car, detected_names


def show_clusters(images: List[Image.Image], labels: np.ndarray, car_flags: Optional[List[Optional[bool]]] = None):
    """
    Kümeleri görsel olarak gösterir; car_flags varsa küme özetlerini ekler.
    """
    unique_labels = sorted(set(labels.tolist()))
    for cluster_id in unique_labels:
        st.subheader(f"Küme {cluster_id}")
        idxs = [i for i, l in enumerate(labels) if l == cluster_id]

        if car_flags is not None:
            flags = [car_flags[i] for i in idxs if car_flags[i] is not None]
            if len(flags) > 0:
                car_ratio = sum(1 for f in flags if f) / len(flags)
                st.markdown(f"- Araba oranı: `{car_ratio:.2f}` ({sum(1 for f in flags if f)}/{len(flags)})")
            else:
                st.markdown("- Araba oranı: `(veri yok)`")

        cols = st.columns(4)
        for j, i in enumerate(idxs):
            with cols[j % 4]:
                st.image(images[i], width='stretch')


def main():
    st.set_page_config(page_title="Image Clustering + Car Check", page_icon="🧩", layout="wide")
    st.title("Gerçek Image Clustering ve Araba Tespiti 🧩🚗")

    st.markdown("1) Birden çok resmi yükleyin ve KMeans ile kümelenmiş halini görün.")
    st.markdown("2) İsteğe bağlı olarak YOLOv8 ile her resimde 'car' olup olmadığını tespit edin.")

    with st.sidebar:
        st.header("Ayarlar")
        k = st.slider("Küme sayısı (K)", min_value=2, max_value=10, value=3, step=1)
        hog_ppc = st.select_slider("HOG pixels_per_cell", options=[8, 12, 16, 24], value=16)
        # HOG cells_per_block: select_slider yerine selectbox kullanıyoruz
        hog_cpb_str = st.selectbox("HOG cells_per_block", options=["1x1", "2x2", "3x3"], index=1)
        hog_cpb_map = {"1x1": (1, 1), "2x2": (2, 2), "3x3": (3, 3)}
        hog_cpb = hog_cpb_map[hog_cpb_str]
        hist_bins = st.select_slider("Renk histogram bins", options=[32, 64, 128], value=64)
        do_car = st.checkbox("Araba tespitini (YOLOv8) ekle", value=False)
        conf = st.slider("YOLO confidence", 0.05, 0.90, 0.25, 0.05)

    uploaded = st.file_uploader("Birden çok resim yükleyin", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
    if uploaded:
        images: List[Image.Image] = [load_image(f.read()) for f in uploaded]
        N = len(images)

        if N < 2:
            st.warning("Kümelenme için en az 2 resim yükleyin.")
            for img in images:
                st.image(img, width='stretch')
        else:
            st.info("Özellikler hesaplanıyor (HOG + renk histogramı)...")
            feats = []
            for img in images:
                np_img = np.array(img)
                feats.append(
                    extract_features(
                        np_img,
                        hog_pixels_per_cell=(hog_ppc, hog_ppc),
                        hog_cells_per_block=hog_cpb,
                        hist_bins=hist_bins,
                        resize_to=256,  # sabit boyut
                    )
                )
            feats = np.stack(feats, axis=0)

            if k > N:
                st.warning(f"Küme sayısı K={k}, örnek sayısı N={N}. K, N'e düşürüldü.")
                k = N

            st.info(f"KMeans ile {k} kümeye ayırılıyor...")
            labels, _km = cluster_images(feats, n_clusters=k)

            car_flags: Optional[List[Optional[bool]]] = None
            if do_car:
                st.info("Araba tespiti çalıştırılıyor (YOLOv8)...")
                car_flags = []
                for img in images:
                    is_car, detected = detect_car(img, conf_threshold=conf)
                    if is_car is None:
                        st.warning("YOLOv8 (ultralytics) kurulu değil. Araba tespiti devre dışı.")
                        car_flags = None
                        break
                    car_flags.append(is_car)

            # labels burada tanımlı, güvenle çağrılabilir
            show_clusters(images, labels, car_flags)

    st.divider()
    st.header("Tek Resim: Araba mı?")
    one = st.file_uploader("Tek resim yükleyin", type=["jpg", "jpeg", "png"], key="single")
    if one:
        img = load_image(one.read())
        st.image(img, caption="Yüklenen resim", width='stretch')
        is_car, detected = detect_car(img, conf_threshold=conf)
        if is_car is None:
            st.warning("YOLOv8 (ultralytics) kurulu değil. Kurulum için aşağıya bakın.")
        else:
            st.markdown(f"- Araba: **{'Evet' if is_car else 'Hayır'}**")
            st.markdown(f"- Tespit edilen sınıflar: `{detected if detected else ['(none)']}`")


if __name__ == "__main__":
    main()