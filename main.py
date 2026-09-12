import json
import numpy as np
import joblib


def predict_material(json_file):
    pls = joblib.load(
        "models/pls_da.pkl"
    )
    scaler = joblib.load(
        "models/scaler.pkl"
    )
    label_encoder = joblib.load(
        "models/label_encoder.pkl"
    )
    with open(json_file, "r") as f:
        data = json.load(f)

    ys = np.asarray(
        data["ys"],
        dtype=float
    )

    if np.isnan(ys).any():

        raise ValueError(
            "Le spectre contient des NaN."
        )

    if np.isinf(ys).any():

        raise ValueError(
            "Le spectre contient des Inf."
        )

    expected_points = scaler.n_features_in_

    if len(ys) != expected_points:

        raise ValueError(
            f"Nombre de points incorrect : "
            f"{len(ys)} au lieu de "
            f"{expected_points}."
        )

    X = ys.reshape(1, -1)
    X_scaled = scaler.transform(X)

    prediction = pls.predict(
        X_scaled
    )

    prediction = np.maximum(
        prediction,
        0
    )

    probabilities = (
        prediction /
        np.sum(
            prediction,
            axis=1,
            keepdims=True
        )
    )

    predicted_class = np.argmax(
        probabilities[0]
    )

    material = label_encoder.inverse_transform(
        [predicted_class]
    )[0]

    confidence = probabilities[
        0,
        predicted_class
    ]

    return material, confidence


if __name__ == "__main__":

    json_file = input(
        "Chemin du fichier JSON :"
    )

    material, confidence = predict_material(
        json_file
    )

    print("\n======================================")
    print("RÉSULTAT")
    print("======================================")

    print(
        f"Matière prédite : {material}"
    )

    print(
        f"Score           : {confidence:.2%}"
    )