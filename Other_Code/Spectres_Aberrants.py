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

    print(f"\n===== {material} =====")
    print(f"Nombre de fichiers : {len(json_files)}")

    for json_file in json_files:

        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            ys = np.array(data["ys"], dtype=float)

            if np.isnan(ys).any() or np.isinf(ys).any():
                print(f"Valeurs invalides : {json_file.name}")

            results.append({
                "material": material,
                "file": str(json_file),
                "min": np.min(ys),
                "max": np.max(ys),
                "mean": np.mean(ys),
                "std": np.std(ys),
                "n_points": len(ys)
            })

        except Exception as e:
            print(f"ERREUR : {json_file}")
            print(e)

df = pd.DataFrame(results)

outliers = []

for material in materials:

    data = df[df["material"] == material].copy()

    Q1_mean = data["mean"].quantile(0.25)
    Q3_mean = data["mean"].quantile(0.75)

    IQR_mean = Q3_mean - Q1_mean

    lower_mean = Q1_mean - 1.5 * IQR_mean
    upper_mean = Q3_mean + 1.5 * IQR_mean

    Q1_std = data["std"].quantile(0.25)
    Q3_std = data["std"].quantile(0.75)

    IQR_std = Q3_std - Q1_std

    lower_std = Q1_std - 1.5 * IQR_std
    upper_std = Q3_std + 1.5 * IQR_std

    data["outlier_mean"] = (
        (data["mean"] < lower_mean) |
        (data["mean"] > upper_mean)
    )

    data["outlier_std"] = (
        (data["std"] < lower_std) |
        (data["std"] > upper_std)
    )

    data["outlier"] = (
        data["outlier_mean"] |
        data["outlier_std"]
    )

    outliers.append(data)

df_results = pd.concat(outliers, ignore_index=True)

print("\n\n==============================================")
print("           SPECTRES ABERRANTS")
print("==============================================")

for material in materials:

    abnormal = df_results[
        (df_results["material"] == material) &
        (df_results["outlier"])
    ]

    print(f"\n===== {material} =====")

    print(f"Nombre de spectres aberrants : {len(abnormal)}")

    if len(abnormal) > 0:

        print(
            abnormal[
                [
                    "file",
                    "min",
                    "max",
                    "mean",
                    "std",
                    "outlier_mean",
                    "outlier_std"
                ]
            ].to_string(index=False)
        )

print("\n\n==============================================")
print("        RÉSUMÉ DE LA DÉTECTION")
print("==============================================")

for material in materials:

    total = len(
        df_results[df_results["material"] == material]
    )

    abnormal = len(
        df_results[
            (df_results["material"] == material) &
            (df_results["outlier"])
        ]
    )

    percentage = 100 * abnormal / total

    print(
        f"{material:10s} : "
        f"{abnormal:4d} aberrants / "
        f"{total:4d} "
        f"({percentage:.2f} %)"
    )

df_results.to_csv(
    "detection_spectres_aberrants.csv",
    index=False
)
print("\nRésultats sauvegardés dans :")
print("detection_spectres_aberrants.csv")