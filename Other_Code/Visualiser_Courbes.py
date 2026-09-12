import json
from pathlib import Path
import matplotlib.pyplot as plt


DATA_DIR = Path("../Jeu de mesures")
materials = ["ACETATE", "COTON", "LIN", "PES"]


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes = axes.flatten()

for i, material in enumerate(materials):

    material_dir = DATA_DIR / material
    json_files = list(material_dir.rglob("*.json"))

    print(f"{material} : {len(json_files)} fichiers JSON")

    ax = axes[i]

    for json_file in json_files[:100]:

        with open(json_file, "r") as f:
            data = json.load(f)

        ys = data["ys"]

        ax.plot(ys, alpha=0.15)

    ax.set_title(f"Spectres - {material}")
    ax.set_xlabel("Indice de mesure")
    ax.set_ylabel("Intensité")
    ax.grid(True)

plt.tight_layout()

plt.show()