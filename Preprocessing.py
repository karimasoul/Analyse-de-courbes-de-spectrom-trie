import json
import numpy as np
from pathlib import Path


# PARAMÈTRES

DATA_DIR = Path(
    "Jeu de mesures"
)
MATERIALS = [
    "ACETATE",
    "COTON",
    "LIN",
    "PES"
]

# CHARGEMENT DES SPECTRES

def load_spectra(data_dir=DATA_DIR, materials=MATERIALS):
    spectra_by_material = {}
    for material in materials:
        material_dir = Path(data_dir) / material
        json_files = list(
            material_dir.rglob("*.json")
        )
        spectra = []
        for json_file in json_files:
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                ys = np.asarray(
                    data["ys"],
                    dtype=float
                )
                if np.isnan(ys).any():
                    continue
                if np.isinf(ys).any():
                    continue
                spectra.append({
                    "file": str(json_file),
                    "ys": ys
                })
            except Exception as e:

                print(
                    f"ERREUR : {json_file}"
                )
                print(e)
        spectra_by_material[material] = spectra
        print(
            f"{material} : "
            f"{len(spectra)} spectres chargés"
        )
    return spectra_by_material

# FUSION DES DOUBLONS EXACTS

def merge_duplicates(spectra_by_material):

    merged_spectra = {}
    duplicate_counts = {}
    for material, spectra in spectra_by_material.items():
        groups = {}
        for spectrum in spectra:
            key = tuple(
                spectrum["ys"]
            )
            if key not in groups:
                groups[key] = []
            groups[key].append(
                spectrum["ys"]
            )
        merged = []
        number_of_duplicates = 0
        for group in groups.values():
            mean_spectrum = np.mean(
                np.asarray(group),
                axis=0
            )
            merged.append({
                "ys": mean_spectrum,
                "n_merged": len(group)
            })
            if len(group) > 1:
                number_of_duplicates += (
                    len(group) - 1
                )
        merged_spectra[material] = merged
        duplicate_counts[material] = (
            number_of_duplicates
        )
        print(
            f"{material} : "
            f"{len(spectra)} → "
            f"{len(merged)} spectres "
            f"après fusion"
        )
    return (
        merged_spectra,
        duplicate_counts
    )

# DÉTECTION DES ABERRANTS

def remove_outliers(
    spectra_by_material,
    iqr_factor=1.5
):

    cleaned_spectra = {}
    outlier_counts = {}
    for material, spectra in spectra_by_material.items():

        means = np.array([
            np.mean(s["ys"])
            for s in spectra
        ])
        stds = np.array([
            np.std(s["ys"])
            for s in spectra
        ])

        Q1_mean = np.percentile(
            means,
            25
        )
        Q3_mean = np.percentile(
            means,
            75
        )
        IQR_mean = (
            Q3_mean - Q1_mean
        )
        lower_mean = (
            Q1_mean
            - iqr_factor * IQR_mean
        )
        upper_mean = (
            Q3_mean
            + iqr_factor * IQR_mean
        )

        Q1_std = np.percentile(
            stds,
            25
        )

        Q3_std = np.percentile(
            stds,
            75
        )

        IQR_std = (
            Q3_std - Q1_std
        )

        lower_std = (
            Q1_std
            - iqr_factor * IQR_std
        )

        upper_std = (
            Q3_std
            + iqr_factor * IQR_std
        )

        cleaned = []
        number_of_outliers = 0
        for spectrum, mean, std in zip(
            spectra,
            means,
            stds
        ):
            mean_outlier = (
                mean < lower_mean
                or
                mean > upper_mean
            )
            std_outlier = (
                std < lower_std
                or
                std > upper_std
            )

            if not mean_outlier and not std_outlier:

                cleaned.append(
                    spectrum
                )
            else:
                number_of_outliers += 1

        cleaned_spectra[material] = cleaned
        outlier_counts[material] = (
            number_of_outliers
        )
        print(
            f"{material} : "
            f"{len(spectra)} → "
            f"{len(cleaned)} spectres "
            f"après nettoyage"
        )

    return (
        cleaned_spectra,
        outlier_counts
    )

# CONVERSION EN X et y

def convert_to_ml(
    spectra_by_material,
    materials=MATERIALS
):

    X = []
    y = []

    for material in materials:
        for spectrum in spectra_by_material[material]:
            X.append(
                spectrum["ys"]
            )
            y.append(
                material
            )
    X = np.asarray(
        X,
        dtype=float
    )
    y = np.asarray(
        y
    )
    return X, y

# PIPELINE

def load_and_clean_data(
    data_dir=DATA_DIR,
    materials=MATERIALS,
    iqr_factor=1.5
):

    spectra = load_spectra(
        data_dir=data_dir,
        materials=materials
    )

    spectra, duplicate_counts = (
        merge_duplicates(spectra)
    )

    spectra, outlier_counts = (
        remove_outliers(
            spectra,
            iqr_factor=iqr_factor
        )
    )

    X, y = convert_to_ml(
        spectra,
        materials=materials
    )

    print("\n====================================")
    print("RÉSUMÉ DU PRÉTRAITEMENT")
    print("====================================")

    for material in materials:

        print(
            f"{material:10s} | "
            f"doublons fusionnés : "
            f"{duplicate_counts[material]:4d} | "
            f"aberrants retirés : "
            f"{outlier_counts[material]:4d}"
        )

    print("\nDonnées ML :")
    print("X :", X.shape)
    print("y :", y.shape)
    return X, y