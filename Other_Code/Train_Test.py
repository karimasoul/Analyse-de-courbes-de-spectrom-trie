import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Preprocessing import load_and_clean_data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.utils import to_categorical


# PRÉTRAITEMENT
X, y = load_and_clean_data()

print("\n======================================")
print("DONNÉES")
print("======================================")

print("X shape :", X.shape)
print("y shape :", y.shape)

print("\nNombre de spectres par matière :")

unique, counts = np.unique(y, return_counts=True)

for material, count in zip(unique, counts):
    print(f"{material:10s} : {count}")


# ENCODAGE DES CLASSES

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
classes = label_encoder.classes_
n_classes = len(classes)

print("\nClasses :")
print(classes)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\n======================================")
print("TRAIN / TEST")
print("======================================")

print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)

print("y_train :", y_train.shape)
print("y_test  :", y_test.shape)

# STANDARDISATION

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# FONCTION D'ÉVALUATION

results = []

def evaluate_model(
    name,
    y_true,
    y_pred,
    y_proba
):
    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    auc = roc_auc_score(
        y_true,
        y_proba,
        multi_class="ovr",
        average="weighted"
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "AUC-ROC": auc
    })

    print(f"\n===== {name} =====")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"AUC-ROC   : {auc:.4f}")

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=classes
    )

    disp.plot(
        xticks_rotation=45
    )

    plt.title(
        f"Matrice de confusion - {name}"
    )

    plt.tight_layout()
    plt.show()

# PLS-DA

print("\n======================================")
print("PLS-DA")
print("======================================")

y_train_onehot = to_categorical(
    y_train,
    num_classes=n_classes
)

n_components = min(
    10,
    X_train_scaled.shape[1],
    X_train_scaled.shape[0] - 1
)

pls = PLSRegression(
    n_components=n_components
)

pls.fit(
    X_train_scaled,
    y_train_onehot
)

pls_prediction = pls.predict(
    X_test_scaled
)

pls_prediction = np.maximum(
    pls_prediction,
    0
)

pls_proba = (
    pls_prediction /
    np.sum(
        pls_prediction,
        axis=1,
        keepdims=True
    )
)

pls_pred = np.argmax(
    pls_proba,
    axis=1
)

evaluate_model(
    "PLS-DA",
    y_test,
    pls_pred,
    pls_proba
)

# SUPPORT VECTOR MACHINE

print("\n======================================")
print("SVM")
print("======================================")

svm = SVC(
    kernel="rbf",
    probability=True,
    random_state=42
)

svm.fit(
    X_train_scaled,
    y_train
)

svm_pred = svm.predict(
    X_test_scaled
)

svm_proba = svm.predict_proba(
    X_test_scaled
)

evaluate_model(
    "SVM",
    y_test,
    svm_pred,
    svm_proba
)

# RANDOM FOREST

print("\n======================================")
print("RANDOM FOREST")
print("======================================")

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

rf.fit(
    X_train,
    y_train
)

rf_pred = rf.predict(
    X_test
)

rf_proba = rf.predict_proba(
    X_test
)

evaluate_model(
    "Random Forest",
    y_test,
    rf_pred,
    rf_proba
)


# CNN 1D

print("\n======================================")
print("CNN 1D")
print("======================================")

X_train_cnn = X_train_scaled[..., np.newaxis]

X_test_cnn = X_test_scaled[..., np.newaxis]

cnn = Sequential([

    Conv1D(
        filters=32,
        kernel_size=5,
        activation="relu",
        input_shape=(
            X_train_cnn.shape[1],
            1
        )
    ),

    MaxPooling1D(
        pool_size=2
    ),

    Conv1D(
        filters=64,
        kernel_size=5,
        activation="relu"
    ),

    MaxPooling1D(
        pool_size=2
    ),

    Flatten(),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        n_classes,
        activation="softmax"
    )
])

cnn.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

cnn.fit(
    X_train_cnn,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

cnn_proba = cnn.predict(
    X_test_cnn
)

cnn_pred = np.argmax(
    cnn_proba,
    axis=1
)

evaluate_model(
    "CNN 1D",
    y_test,
    cnn_pred,
    cnn_proba
)

# COMPARAISON DES MODÈLES

results_df = pd.DataFrame(
    results
)

print("\n\n======================================")
print("COMPARAISON DES MODÈLES")
print("======================================")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# CLASSEMENT PAR F1-SCORE

print("\n======================================")
print("CLASSEMENT")
print("======================================")

ranking = results_df.sort_values(
    by="F1-Score",
    ascending=False
)

print(
    ranking[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "AUC-ROC"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)