import json
from pathlib import Path
import numpy as np
import pandas as pd


DATA_DIR = Path("../Jeu de mesures")
materials = ["ACETATE", "COTON", "LIN", "PES"]
results = []

for material in materials:

    material_dir = DATA_DIR / material
    json_files = list(material_dir.rglob("*.json"))

    all_ys = []

    spectra = []

    for json_file in json_files:

        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            ys = np.array(data["ys"], dtype=float)

            all_ys.extend(ys)

            spectra.append({
                "file": str(json_file),
                "ys": ys
            })

        except Exception as e:
            print(f"ERREUR : {json_file}")
            print(e)

    all_ys = np.array(all_ys)

    results.append({
        "material": material,
        "nombre_fichiers": len(json_files),
        "nombre_total_points": len(all_ys),
        "min": np.min(all_ys),
        "max": np.max(all_ys),
        "mean": np.mean(all_ys),
        "std": np.std(all_ys),
        "nan": np.isnan(all_ys).sum(),
        "inf": np.isinf(all_ys).sum()
    })

    spectra_dict = {}

    for spectrum in spectra:

        key = tuple(spectrum["ys"])

        if key not in spectra_dict:
            spectra_dict[key] = []

        spectra_dict[key].append(spectrum["file"])

    duplicates = [
        files
        for files in spectra_dict.values()
        if len(files) > 1
    ]

    print(f"\n===== DOUBLONS {material} =====")

    if len(duplicates) == 0:

        print("Aucun doublon exact détecté.")

    else:

        print(
            f"Nombre de groupes de doublons : {len(duplicates)}"
        )

        for i, files in enumerate(duplicates, start=1):

            print(f"\nDoublon #{i} :")

            for file in files:
                print(f"  {file}")


df = pd.DataFrame(results)

print("\n========== STATISTIQUES PAR MATIÈRE ==========\n")

print(df.to_string(index=False))