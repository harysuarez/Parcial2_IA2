# train.py  (baseline HOG + LBP + SVM, validación SIN leakage)
import json
import os
import cv2
import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

from features import extract_all_descriptors

SEED = 42
rng = np.random.default_rng(SEED)


def make_pipeline(probability=False):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='rbf', C=15.0, gamma='scale',
                           class_weight='balanced',
                           probability=probability, random_state=SEED)),
    ])


# ----------------- DATOS -----------------
def load_shipsnet(json_path="data/shipsnet.json"):
    print("[1/5] Cargando dataset shipsnet.json...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    raw = np.array(data['data'], dtype=np.uint8)
    images = raw.reshape(-1, 3, 80, 80).transpose(0, 2, 3, 1)  # N,80,80,3
    labels = np.array(data['labels'], dtype=np.int32)

    if 'scene_ids' in data:
        groups = np.array(data['scene_ids'])
        print(f"   {len(np.unique(groups))} escenas distintas -> CV agrupada por escena")
    else:
        print("   ⚠️ No hay 'scene_ids': la validación tendrá leakage (recortes vecinos).")
        groups = np.arange(len(labels))
    return np.ascontiguousarray(images), labels, groups


# ----------------- AUGMENTATION -----------------
def zoom_in_crop(image, zoom_factor=1.4):
    h, w = image.shape[:2]
    zh, zw = int(h / zoom_factor), int(w / zoom_factor)
    top, left = (h - zh) // 2, (w - zw) // 2
    cropped = image[top:top + zh, left:left + zw]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def make_variants(img, lbl):
    """Variantes sin esquinas negras: solo rot90, flips y zoom."""
    out = []
    if lbl == 1:
        out.append(np.fliplr(img))
        out.append(np.rot90(img, int(rng.integers(1, 4))))
        out.append(np.flipud(np.rot90(img, int(rng.integers(1, 4)))))
        zoomed = zoom_in_crop(img, 1.4)
        out.append(zoomed)
        out.append(np.rot90(zoomed, int(rng.integers(1, 4))))
    else:
        out.append(np.rot90(img, int(rng.integers(1, 4))))
    return [np.ascontiguousarray(v) for v in out]


def feat(img):
    return extract_all_descriptors(img)


def extract_batch(imgs):
    return np.array(Parallel(n_jobs=-1)(delayed(feat)(im) for im in imgs))


def metrics(y_true, y_pred):
    return (accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, zero_division=0),
            recall_score(y_true, y_pred, zero_division=0),
            f1_score(y_true, y_pred, zero_division=0))


def evaluate_external(pipe, folder="test_externo"):
    """Opcional: evalúa sobre imágenes propias (con ground_truth.csv) nunca usadas al entrenar."""
    csv = os.path.join(folder, "ground_truth.csv")
    if not os.path.exists(csv):
        print(f"\n(Sin '{csv}': se omite evaluación externa)")
        return
    df = pd.read_csv(csv)
    df.columns = [c.strip().lower() for c in df.columns]
    X, y = [], []
    for fn, lb in zip(df['filename'], df['label']):
        im = cv2.imread(os.path.join(folder, fn))
        if im is None:
            continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h, w = im.shape[:2]
        s = min(h, w)
        im = im[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
        X.append(feat(im))
        y.append(int(lb))
    pred = pipe.predict(np.array(X))
    a, p, r, f = metrics(y, pred)
    print(f"\n🌍 EXTERNO ({len(y)} imgs) -> Acc {a*100:.2f}% | Prec {p*100:.2f}% | "
          f"Rec {r*100:.2f}% | F1 {f*100:.2f}%")
    print(confusion_matrix(y, pred, labels=[0, 1]))


def main():
    images, labels, groups = load_shipsnet()
    n = len(images)

    # Se calculan UNA vez las features de originales y de variantes,
    # guardando el índice del original (padre) para filtrar por fold.
    print(f"[2/5] Extrayendo descriptores de {n} imágenes originales...")
    X_orig = extract_batch(images)

    print("[3/5] Generando variantes aumentadas y sus descriptores...")
    aug_imgs, aug_labels, aug_parent = [], [], []
    for i in range(n):
        for v in make_variants(images[i], labels[i]):
            aug_imgs.append(v)
            aug_labels.append(labels[i])
            aug_parent.append(i)
    aug_labels = np.array(aug_labels)
    aug_parent = np.array(aug_parent)
    X_aug = extract_batch(aug_imgs)
    print(f"   {len(aug_imgs)} variantes generadas")

    print("\n[4/5] Validación cruzada agrupada por escena (augment SOLO en train)...")
    gkf = GroupKFold(n_splits=5)
    accs, precs, recs, f1s = [], [], [], []
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_orig, labels, groups), 1):
        mask = np.isin(aug_parent, tr_idx)              # variantes de imágenes de train
        X_tr = np.vstack([X_orig[tr_idx], X_aug[mask]])
        y_tr = np.concatenate([labels[tr_idx], aug_labels[mask]])

        pipe = make_pipeline()
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_orig[val_idx])            # validación: solo originales
        a, p, r, f = metrics(labels[val_idx], pred)
        accs.append(a); precs.append(p); recs.append(r); f1s.append(f)
        print(f"  Fold {fold} -> Acc {a*100:.2f}% | Prec {p*100:.2f}% | "
              f"Rec {r*100:.2f}% | F1 {f*100:.2f}%")

    print(f"\nPromedio CV por escenas -> Acc {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f} | "
          f"Prec {np.mean(precs)*100:.2f}% | Rec {np.mean(recs)*100:.2f}% | "
          f"F1 {np.mean(f1s)*100:.2f}%")

    print("\n[5/5] Entrenando modelo final con todos los datos...")
    X_all = np.vstack([X_orig, X_aug])
    y_all = np.concatenate([labels, aug_labels])
    final_pipeline = make_pipeline(probability=True)
    final_pipeline.fit(X_all, y_all)

    os.makedirs("models", exist_ok=True)
    joblib.dump(final_pipeline, "models/ship_detector_pipeline.joblib")
    print("✅ Modelo guardado en models/ship_detector_pipeline.joblib")

    evaluate_external(final_pipeline)


if __name__ == '__main__':
    main()