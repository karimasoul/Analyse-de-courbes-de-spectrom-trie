import joblib
import os
from Preprocessing import load_and_clean_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cross_decomposition import PLSRegression
from tensorflow.keras.utils import to_categorical


# CHARGEMENT DES DONNÉES

X, y = load_and_clean_data()

# ENCODAGE DES MATIÈRES

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
classes = label_encoder.classes_

# TRAIN et TEST

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

# STANDARDISATION

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PLS-DA

y_train_onehot = to_categorical(
    y_train,
    num_classes=len(classes)
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

# CRÉER LE DOSSIER MODELS

os.makedirs("models", exist_ok=True)

# SAUVEGARDER LE MODÈLE

joblib.dump(
    pls,
    "models/pls_da.pkl"
)
joblib.dump(
    scaler,
    "models/scaler.pkl"
)
joblib.dump(
    label_encoder,
    "models/label_encoder.pkl"
)

# INFORMATIONS

print("\n======================================")
print("MODÈLE SAUVEGARDÉ")
print("======================================")
print("Modèle      : models/pls_da.pkl")
print("Scaler      : models/scaler.pkl")
print("Classes     : models/label_encoder.pkl")

print("\nClasses utilisées :")
for i, classe in enumerate(classes):
    print(f"{i} → {classe}")
print("\nEntraînement terminé.")