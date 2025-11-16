# -*- coding: utf-8 -*-
"""
Dementia Prediction Model (TensorFlow + Keras)
Using Non-Medical Features from NACC dataset
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ================================================================
# 1. CONFIGURATION
# ================================================================

FILE_PATH = "./dataset.csv"  # change to your CSV path
TARGET_VARIABLE = "NACCIDEM"

# --- Master list of all possible non-medical features ---
ALL_POSSIBLE_FEATURES = [
    # A1: Demographics
    'NACCAGE','SEX','EDUC','MARISTAT','NACCLIVS','RESIDENC','HANDED',
    'HISPANIC','RACE','RACESEC','RACETER','PRIMLANG','INDEPEND',
    # A2: Co-participant
    'INRELTO','INLIVWTH','INVISITS','INCALLS',
    # A3: Family history
    'NACCFAM','NACCMOM','NACCDAD',
    # A4: Medications
    'ANYMEDS',
    # A5: Health history
    'TOBAC30','TOBAC100','SMOKYRS','PACKSPER','QUITSMOK','ALCOCCAS',
    'ALCFREQ','ALCOHOL','ABUSOTHR','CVHATT','CVAFIB','CVANGIO',
    'CVBYPASS','CVPACDEF','CVPACE','CVCHF','CVANGINA','CVHVALVE',
    'CVOTHR','HYPERTEN','HYPERCHO','CBSTROKE','NACCSTYR','CBTIA',
    'NACCTIYR','PD','PDYR','SEIZURES','NACCTBI','DIABETES','DIABTYPE',
    'B12DEF','THYROID','ARTHRIT','APNEA','RBD','INSOMN','OTHSLEEP',
    'PTSD','BIPOLAR','SCHIZ','DEP2YRS','DEPOTHR','ANXIETY','OCD',
    'INCONTU','INCONTF',
    # B1: Physical
    'HEIGHT','WEIGHT','NACCBMI','VISION','VISCORR','VISWCORR',
    'HEARING','HEARAID','HEARWAID',
    # B9: Self-Reported Decline
    'DECSUB','DECIN',
    # B7: Functional Activities
    'BILLS','TAXES','SHOPPING','GAMES','STOVE','MEALPREP',
    'EVENTS','PAYATTN','REMDATES','TRAVEL',
    # Milestones
    'NACCNURP',
    # CLS: Linguistic History
    'APREFLAN','AYRSPAN','AYRENGL','APCSPAN','APCENGL','NACCSPNL','NACCENGL'
]


# ================================================================
# 2. LOAD DATA
# ================================================================

print(f"Loading data from: {FILE_PATH}")
try:
    df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    raise SystemExit(f"ERROR: File not found at {FILE_PATH}")
except Exception as e:
    raise SystemExit(f"Error loading data: {e}")

if df.empty:
    raise SystemExit("ERROR: DataFrame is empty. Please check your CSV file.")

# Keep only columns available
NON_MEDICAL_FEATURES = [c for c in ALL_POSSIBLE_FEATURES if c in df.columns]
missing_from_csv = [c for c in ALL_POSSIBLE_FEATURES if c not in df.columns]

print(f"Found {len(NON_MEDICAL_FEATURES)} usable non-medical features.")
if missing_from_csv:
    print("\nMissing columns skipped:")
    print(missing_from_csv)

# Build working frame
cols_to_use = NON_MEDICAL_FEATURES + [TARGET_VARIABLE]
df = df[cols_to_use].copy()


# ================================================================
# 3. CLEANING & TARGET HANDLING
# ================================================================

# Treat 8 as "unknown" for NACCIDEM
df.loc[df[TARGET_VARIABLE] == 8, TARGET_VARIABLE] = np.nan

# Replace sentinel "unknown/not assessed" codes
SENTINELS = {
    TARGET_VARIABLE: [8],
    "HISPANIC": [9],
    "RACE": [9],
    "RACESEC": [9],
    "RACETER": [9],
    "VISION": [9, 99],
    "VISCORR": [9, 99],
    "VISWCORR": [9, 99],
    "HEARING": [9, 99],
    "HEARAID": [9, 99],
    "HEARWAID": [9, 99],
    "_default": [-4, 9, 99, 999, 9999],
}

for col in df.columns:
    bads = SENTINELS.get(col, SENTINELS["_default"])
    df[col] = df[col].replace(bads, np.nan)

# Drop missing target and keep only {0,1}
df = df.dropna(subset=[TARGET_VARIABLE])
df = df[df[TARGET_VARIABLE].isin([0, 1])]
df[TARGET_VARIABLE] = df[TARGET_VARIABLE].astype(int)
print(f"Rows after cleaning target: {len(df)}")


# ================================================================
# 4. DETERMINE FEATURE TYPES
# ================================================================

FORCE_NUMERIC = {
    'NACCAGE','EDUC','SMOKYRS','PACKSPER','QUITSMOK','HEIGHT','WEIGHT','NACCBMI',
    'AYRSPAN','AYRENGL'
}
FORCE_CATEGORICAL = {
    'SEX','MARISTAT','NACCLIVS','RESIDENC','HANDED','HISPANIC','RACE','RACESEC',
    'RACETER','PRIMLANG','INDEPEND','INRELTO','INLIVWTH','INVISITS','INCALLS',
    'NACCFAM','NACCMOM','NACCDAD','ANYMEDS','TOBAC30','TOBAC100','ALCOCCAS',
    'ALCFREQ','ALCOHOL','ABUSOTHR','CVHATT','CVAFIB','CVANGIO','CVBYPASS',
    'CVPACDEF','CVPACE','CVCHF','CVANGINA','CVHVALVE','CVOTHR','HYPERTEN',
    'HYPERCHO','CBSTROKE','NACCSTYR','CBTIA','NACCTIYR','PD','PDYR','SEIZURES',
    'NACCTBI','DIABETES','DIABTYPE','B12DEF','THYROID','ARTHRIT','APNEA','RBD',
    'INSOMN','OTHSLEEP','PTSD','BIPOLAR','SCHIZ','DEP2YRS','DEPOTHR','ANXIETY',
    'OCD','INCONTU','INCONTF','VISION','VISCORR','VISWCORR','HEARING','HEARAID',
    'HEARWAID','DECSUB','DECIN','BILLS','TAXES','SHOPPING','GAMES','STOVE',
    'MEALPREP','EVENTS','PAYATTN','REMDATES','TRAVEL','NACCNURP',
    'APREFLAN','APCSPAN','APCENGL','NACCSPNL','NACCENGL'
}

NUMERIC_FEATURES, CATEGORICAL_FEATURES = [], []

for col in NON_MEDICAL_FEATURES:
    if col in FORCE_NUMERIC:
        NUMERIC_FEATURES.append(col)
    elif col in FORCE_CATEGORICAL:
        CATEGORICAL_FEATURES.append(col)
    else:
        # Heuristic: objects -> categorical; low-cardinality ints also categorical
        if df[col].dtype == "O":
            CATEGORICAL_FEATURES.append(col)
        else:
            nunq = df[col].nunique(dropna=True)
            if nunq <= 20:
                CATEGORICAL_FEATURES.append(col)
            else:
                NUMERIC_FEATURES.append(col)

print(f"Numeric features   ({len(NUMERIC_FEATURES)}): {NUMERIC_FEATURES}")
print(f"Categorical fields ({len(CATEGORICAL_FEATURES)}): {CATEGORICAL_FEATURES}")


# ================================================================
# 5. SPLIT & IMPUTE
# ================================================================

X = df[NON_MEDICAL_FEATURES]
y = df[TARGET_VARIABLE].values

X_train_df, X_test_df, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train_df.shape[0]}  Test: {X_test_df.shape[0]}")

# --- Numeric median imputation ---
num_medians = {c: X_train_df[c].median() for c in NUMERIC_FEATURES}
for c in NUMERIC_FEATURES:
    X_train_df[c] = X_train_df[c].astype("float32").fillna(num_medians[c])
    X_test_df[c]  = X_test_df[c].astype("float32").fillna(num_medians[c])

# --- Categorical fill ---
for c in CATEGORICAL_FEATURES:
    X_train_df[c] = X_train_df[c].astype(str).fillna("Unknown")
    X_test_df[c]  = X_test_df[c].astype(str).fillna("Unknown")


# ================================================================
# 6. PREPROCESSING LAYERS
# ================================================================

normalizers = {}
for c in NUMERIC_FEATURES:
    norm = layers.Normalization(axis=None, name=f"norm_{c}")
    norm.adapt(X_train_df[c].values.astype("float32"))
    normalizers[c] = norm

lookups, encoders = {}, {}
for c in CATEGORICAL_FEATURES:
    lookup = layers.StringLookup(output_mode="int", num_oov_indices=1, name=f"lookup_{c}")
    lookup.adapt(X_train_df[c].values.astype(str))
    vocab_size = lookup.vocabulary_size()
    enc = layers.CategoryEncoding(num_tokens=vocab_size, output_mode="one_hot", name=f"onehot_{c}")
    lookups[c] = lookup
    encoders[c] = enc


# ================================================================
# 7. BUILD MODEL
# ================================================================

inputs, encoded = {}, []

for c in NUMERIC_FEATURES:
    inp = keras.Input(shape=(1,), name=c, dtype="float32")
    x = normalizers[c](inp)
    inputs[c] = inp
    encoded.append(x)

for c in CATEGORICAL_FEATURES:
    inp = keras.Input(shape=(1,), name=c, dtype="string")
    idx = lookups[c](inp)
    oh = encoders[c](idx)
    inputs[c] = inp
    encoded.append(oh)

x = layers.Concatenate(name="concat_all")(encoded)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation="relu")(x)
out = layers.Dense(1, activation="sigmoid", name="prob")(x)

model = keras.Model(inputs=inputs, outputs=out)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.AUC(name="AUC"),
        keras.metrics.Precision(name="Precision"),
        keras.metrics.Recall(name="Recall")
    ]
)
model.summary()


# ================================================================
# 8. TF.DATA PIPELINES
# ================================================================

def df_to_dict(frame):
    d = {}
    for c in NUMERIC_FEATURES:
        d[c] = frame[c].astype("float32").values
    for c in CATEGORICAL_FEATURES:
        d[c] = frame[c].astype(str).values
    return d

train_ds = tf.data.Dataset.from_tensor_slices((df_to_dict(X_train_df), y_train)).shuffle(4096).batch(64).prefetch(tf.data.AUTOTUNE)
test_ds  = tf.data.Dataset.from_tensor_slices((df_to_dict(X_test_df),  y_test)).batch(256).prefetch(tf.data.AUTOTUNE)


# ================================================================
# 9. CLASS WEIGHTS
# ================================================================

neg, pos = np.bincount(y_train)
total = neg + pos
class_weight = {0: total/(2.0*neg), 1: total/(2.0*pos)}
print("Class weights:", class_weight)


# ================================================================
# 10. TRAIN
# ================================================================

callbacks = [keras.callbacks.EarlyStopping(monitor="val_AUC", patience=5, restore_best_weights=True)]
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=50,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)


# ================================================================
# 11. EVALUATE
# ================================================================

proba = model.predict(test_ds).ravel()
y_pred = (proba >= 0.5).astype(int)

print("\n--- Evaluation (Test) ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC :", roc_auc_score(y_test, proba))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Dementia (0)", "Dementia (1)"]))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

plt.figure(figsize=(6, 4))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.colorbar()
ticks = np.arange(2)
plt.xticks(ticks, ["Pred No", "Pred Yes"], rotation=45, ha="right")
plt.yticks(ticks, ["Actual No", "Actual Yes"])
th = cm.max()/2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, f"{cm[i,j]:d}", ha="center", va="center",
                 color="white" if cm[i,j] > th else "black")
plt.tight_layout()
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.savefig("dementia_confusion_matrix_tf.png", bbox_inches="tight")
print("Saved confusion matrix: dementia_confusion_matrix_tf.png")


# ================================================================
# 12. SAVE MODEL + MEDIANS
# ================================================================

model.save("dementia_tf_model.keras")
with open("numeric_medians.json", "w") as f:
    json.dump(num_medians, f, indent=2)
print("Saved model to dementia_tf_model.keras and medians to numeric_medians.json")

print("\n✅ Done.")
