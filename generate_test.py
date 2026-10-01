# generate_test.py
import os
import json
import cv2
import numpy as np
import pandas as pd

def create_mock_test_folder(output_folder="test_ciego", num_samples=100):
    os.makedirs(output_folder, exist_ok=True)
    with open("data/shipsnet.json", "r") as f:
        data = json.load(f)
        
    raw_data = np.array(data['data'], dtype=np.uint8)
    labels = np.array(data['labels'], dtype=np.int32)
    
    indices = np.random.choice(len(labels), size=num_samples, replace=False)
    
    records = []
    for i, idx in enumerate(indices):
        row = raw_data[idx]
        lbl = int(labels[idx])
        
        r = row[0:6400].reshape((80, 80))
        g = row[6400:12800].reshape((80, 80))
        b = row[12800:19200].reshape((80, 80))
        img_rgb = np.dstack((r, g, b))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        filename = f"sat_sample_{i:03d}.png"
        file_path = os.path.join(output_folder, filename)
        cv2.imwrite(file_path, img_bgr)
        
        records.append({'filename': filename, 'label': lbl})
        
    csv_path = os.path.join(output_folder, "ground_truth.csv")
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"Creadas {num_samples} imágenes en '{output_folder}/' y archivo '{csv_path}'")

if __name__ == '__main__':
    create_mock_test_folder()