# app.py  (MobileNetV2)
import os
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

IMG_SIZE = 224  # Debe coincidir con train_cnn.py
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

st.set_page_config(page_title="UAV Maritime Perception - Live Test",
                   page_icon="🛰️", layout="wide")

ss = st.session_state
ss.setdefault("manual_labels", {})   # solo contiene imágenes REALMENTE etiquetadas
ss.setdefault("res", None)

LABEL_TXT = {-1: "❔ Sin etiquetar", 0: "🌊 No Barco (0)", 1: "🚢 Barco (1)"}

st.title("🛰️ Sistema Embarcado UAV: Detección y Monitoreo de Barcos")
st.markdown("**Inspección Portuaria en Tiempo Real - Red Neuronal MobileNetV2 (Puerto de Rotterdam)**")
st.caption("Criterios ABET: RAE-140 (Arquitectura y Optimización Deep Learning) | "
           "RAE-144 (Inferencia y Validación en Vivo)")

# ----------------- SIDEBAR -----------------
st.sidebar.header("⚙️ Configuración")
model_path = st.sidebar.text_input("Pesos MobileNetV2 (.pth):", "models/mobilenet_uav.pth")
test_folder = st.sidebar.text_input("Carpeta de Prueba (Test Ciego):", "test_ciego")

st.sidebar.markdown("---")
threshold = st.sidebar.slider("Umbral de Sensibilidad UAV:", 0.10, 0.90, 0.50, 0.05)
use_tta = st.sidebar.checkbox("Usar TTA (promedia 4 vistas)", value=True)
square_crop = st.sidebar.checkbox("Recorte cuadrado central (imágenes no cuadradas)", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📋 Modo de Etiquetas")
LBL_UI = "🏷️ Etiquetar interactivamente en la UI (Requisito E1)"
LBL_CSV = "📄 Archivo CSV en la carpeta"
LBL_UP = "📤 Subir archivo CSV manualmente"
LBL_BLIND = "👁️ Inferir sin etiquetas (Modo Ciego Puro)"
gt_source = st.sidebar.radio("Selecciona cómo ingresar las etiquetas:",
                             [LBL_UI, LBL_CSV, LBL_UP, LBL_BLIND])
uploaded_csv = (st.sidebar.file_uploader("Subir archivo .csv", type=["csv"])
                if gt_source == LBL_UP else None)


# ----------------- MODELO -----------------
@st.cache_resource
def load_mobilenet(path, mtime):
    if not os.path.exists(path):
        return None
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    )
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


mtime = os.path.getmtime(model_path) if os.path.exists(model_path) else 0
model = load_mobilenet(model_path, mtime)
if model is None:
    st.sidebar.error("❌ Modelo MobileNetV2 no encontrado. Ejecuta 'train_cnn.py'.")

valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
image_files = (sorted([f for f in os.listdir(test_folder)
                       if f.lower().endswith(valid_extensions)])
               if os.path.isdir(test_folder) else [])

infer_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


# ----------------- UTILIDADES -----------------
def read_rgb(path):
    bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def thumb(arr, max_side=320):
    h, w = arr.shape[:2]
    s = max_side / max(h, w)
    return arr if s >= 1 else cv2.resize(arr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def center_square(img_rgb):
    """Recorta el cuadrado central para no deformar el aspecto al redimensionar."""
    h, w = img_rgb.shape[:2]
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    return np.ascontiguousarray(img_rgb[top:top + s, left:left + s])


def predict_prob_ship(model, tensor, tta=True):
    """Probabilidad de 'barco' (promedio de vistas si TTA)."""
    if tta:
        views = [tensor,
                 torch.flip(tensor, [3]),
                 torch.flip(tensor, [2]),
                 torch.rot90(tensor, 1, [2, 3])]
    else:
        views = [tensor]
    with torch.no_grad():
        return float(torch.stack([torch.softmax(model(v), 1) for v in views]).mean(0)[0, 1])


def read_gt(df):
    df.columns = [c.strip().lower() for c in df.columns]
    return {str(f): int(l) for f, l in zip(df["filename"], df["label"])}


def get_ground_truth():
    if gt_source == LBL_UI:
        return dict(ss["manual_labels"])
    if gt_source == LBL_CSV:
        p = os.path.join(test_folder, "ground_truth.csv")
        if os.path.exists(p):
            return read_gt(pd.read_csv(p))
        st.warning("No se encontró ground_truth.csv en la carpeta.")
    if gt_source == LBL_UP and uploaded_csv is not None:
        return read_gt(pd.read_csv(uploaded_csv))
    return {}


# ----------------- PANEL 1: ETIQUETADO EN VIVO -----------------
if gt_source == LBL_UI:
    st.subheader("🏷️ 1. Módulo de Etiquetado en Vivo (Ground Truth)")
    if image_files:
        per_page = 24
        n_pages = (len(image_files) - 1) // per_page + 1
        page = int(st.number_input(f"Página (1-{n_pages})", 1, n_pages, 1)) if n_pages > 1 else 1
        subset = image_files[(page - 1) * per_page: page * per_page]

        cols = st.columns(6)
        for i, fname in enumerate(subset):
            arr = read_rgb(os.path.join(test_folder, fname))
            if arr is None:
                continue
            with cols[i % 6]:
                st.image(thumb(arr), caption=fname, use_container_width=True)
                current_val = ss["manual_labels"].get(fname, 0)
                choice = st.selectbox(
                    "Clase:", [0, 1],
                    format_func=lambda x: "🚢 Barco (1)" if x == 1 else "🌊 No Barco (0)",
                    index=current_val, key=f"label_{fname}")
                ss["manual_labels"][fname] = choice

        ss.setdefault("visited_pages", set()).add(page)
        if len(ss["visited_pages"]) < n_pages:
            st.warning(f"⚠️ Has visto {len(ss['visited_pages'])}/{n_pages} páginas. "
                       "Abre las demás para etiquetarlas; las no visitadas NO cuentan en las métricas.")
        else:
            st.success(f"✅ Visitaste las {n_pages} páginas ({len(ss['manual_labels'])} imágenes etiquetadas).")

        if st.button("💾 Guardar Etiquetas a CSV"):
            recs = [{"filename": f, "label": ss["manual_labels"][f]}
                    for f in image_files if f in ss["manual_labels"]]
            pd.DataFrame(recs).to_csv(os.path.join(test_folder, "ground_truth.csv"), index=False)
            st.success(f"Guardadas {len(recs)} etiquetas en ground_truth.csv")
    else:
        st.info("No hay imágenes en la carpeta de prueba.")

st.markdown("---")

# ----------------- PANEL 2: INFERENCIA -----------------
st.subheader("⚡ 2. Percepción e Inferencia en Tiempo Real")
st.info(f"📁 Carpeta: `{test_folder}` | 🖼️ Total imágenes: **{len(image_files)}**")

if st.button("🚀 Ejecutar Inferencia en Vivo", type="primary", use_container_width=True):
    if not image_files:
        st.error("No hay imágenes válidas en la carpeta.")
    elif model is None:
        st.error("El modelo no está cargado.")
    else:
        bar = st.progress(0.0)
        names, probs, imgs, lat = [], [], [], []
        t_start = time.time()
        for idx, fname in enumerate(image_files):
            rgb = read_rgb(os.path.join(test_folder, fname))
            if rgb is None:
                continue
            if square_crop:
                rgb = center_square(rgb)
            t0 = time.perf_counter()
            tensor_img = infer_transform(Image.fromarray(rgb)).unsqueeze(0)
            p = predict_prob_ship(model, tensor_img, tta=use_tta)
            lat.append((time.perf_counter() - t0) * 1000)
            names.append(fname); probs.append(p); imgs.append(thumb(rgb))
            bar.progress((idx + 1) / len(image_files))
        ss["res"] = {"folder": test_folder, "names": names, "probs": np.array(probs), "imgs": imgs,
                     "lat": float(np.mean(lat)), "total": time.time() - t_start}

res = ss["res"]
if res is not None and res["folder"] == test_folder:
    names, probs, imgs = res["names"], res["probs"], res["imgs"]
    preds = (probs >= threshold).astype(int)     # el umbral se aplica sin re-ejecutar el modelo
    fps = 1000.0 / res["lat"] if res["lat"] > 0 else 0
    st.success(f"Inferencia MobileNetV2 completada en {res['total']:.2f} s | "
               f"Latencia: {res['lat']:.2f} ms/frame (~{fps:.1f} FPS)")

    # ----------------- PANEL 3: MÉTRICAS (E3 & E4) -----------------
    st.markdown("---")
    st.subheader("📊 3. Panel de Rendimiento (ABET RAE-144)")
    gt = get_ground_truth()
    idx = [i for i, n in enumerate(names) if n in gt]

    if idx:
        y_true = np.array([gt[names[i]] for i in idx], dtype=int)
        y_pred = preds[idx]
        if len(idx) < len(names):
            st.warning(f"Métricas calculadas sobre {len(idx)}/{len(names)} imágenes con etiqueta.")

        acc = accuracy_score(y_true, y_pred) * 100
        prec = precision_score(y_true, y_pred, zero_division=0) * 100
        rec = recall_score(y_true, y_pred, zero_division=0) * 100
        f1 = f1_score(y_true, y_pred, zero_division=0) * 100

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy en Vivo", f"{acc:.2f}%")
        m2.metric("Precision", f"{prec:.2f}%")
        m3.metric("Recall (Sensibilidad)", f"{rec:.2f}%")
        m4.metric("F1-Score", f"{f1:.2f}%")
        m5.metric("FPS Estimados", f"{fps:.1f}")

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['No Barco (0)', 'Barco (1)'],
                    yticklabels=['No Barco (0)', 'Barco (1)'], ax=ax)
        ax.set_xlabel("Predicción MobileNetV2")
        ax.set_ylabel("Etiqueta Real")
        st.pyplot(fig)

        tn, fp, fn, tp = cm.ravel()
        st.caption(f"TN={tn} | FP={fp} (falsas alarmas) | FN={fn} (barcos perdidos) | TP={tp}")

        err = [idx[k] for k in np.where(y_true != y_pred)[0]]
        st.markdown(f"#### 🔍 Análisis de Errores ({len(err)} detectados)")
        if err:
            ecols = st.columns(6)
            for k, i in enumerate(err[:12]):
                with ecols[k % 6]:
                    r_lbl = "BARCO" if gt[names[i]] == 1 else "NO BARCO"
                    p_lbl = "BARCO" if preds[i] == 1 else "NO BARCO"
                    st.image(imgs[i], use_container_width=True,
                             caption=f"{names[i]}\nReal: {r_lbl} | Pred: {p_lbl}\n"
                                     f"Prob: {probs[i]*100:.1f}%")
        else:
            st.info("¡Cero errores registrados!")
    else:
        st.warning("Sin etiquetas disponibles (modo ciego o CSV no encontrado): solo predicciones.")

    st.download_button("⬇️ Descargar predicciones (CSV)",
                       pd.DataFrame({"filename": names, "prob_barco": probs, "prediccion": preds})
                       .to_csv(index=False).encode("utf-8"), "predicciones.csv", "text/csv")

    # ----------------- PANEL 4: GALERÍA -----------------
    st.markdown("---")
    st.subheader("🖼️ 4. Galería de Inspección en Vuelo")
    gcols = st.columns(6)
    for i in range(min(12, len(names))):
        with gcols[i % 6]:
            tag = "🚢 BARCO" if preds[i] == 1 else "🌊 NO BARCO"
            st.image(imgs[i], use_container_width=True,
                     caption=f"{names[i]}\n{tag}\nConf: {probs[i]*100:.1f}%")