import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from typing import List
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.stats import loguniform, randint

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

CSV_PATH = "./dataset.csv"
TARGET_COL = "DEMENTED"

df = pd.read_csv(CSV_PATH)

ALL_POSSIBLE_FEATURES = [
    # A1: Demographics
    'NACCAGE', 'SEX', 'EDUC', 'MARISTAT', 'NACCLIVS', 'RESIDENC', 'HANDED',
    'HISPANIC', 'RACE', 'RACESEC', 'RACETER', 'PRIMLANG', 'INDEPEND',
    # A2: Co-participant
    'INRELTO', 'INLIVWTH', 'INVISITS', 'INCALLS',
    # A3: Family history
    'NACCFAM', 'NACCMOM', 'NACCDAD',
    # A4: Medications
    'ANYMEDS',
    # A5: Health history
    'TOBAC30', 'TOBAC100', 'SMOKYRS', 'PACKSPER', 'QUITSMOK', 'ALCOCCAS',
    'ALCFREQ', 'ALCOHOL', 'ABUSOTHR', 'CVHATT', 'CVAFIB', 'CVANGIO',
    'CVBYPASS', 'CVPACDEF', 'CVPACE', 'CVCHF', 'CVANGINA', 'CVHVALVE',
    'CVOTHR', 'HYPERTEN', 'HYPERCHO', 'CBSTROKE', 'NACCSTYR', 'CBTIA',
    'NACCTIYR', 'PD', 'PDYR', 'SEIZURES', 'NACCTBI', 'DIABETES', 'DIABTYPE',
    'B12DEF', 'THYROID', 'ARTHRIT', 'APNEA', 'RBD', 'INSOMN', 'OTHSLEEP',
    'PTSD', 'BIPOLAR', 'SCHIZ', 'DEP2YRS', 'DEPOTHR', 'ANXIETY', 'OCD',
    'INCONTU', 'INCONTF',
    # B1: Physical
    'HEIGHT', 'WEIGHT', 'NACCBMI', 'VISION', 'VISCORR', 'VISWCORR',
    'HEARING', 'HEARAID', 'HEARWAID',
    # B9: Self-Reported Decline
    'DECSUB', 'DECIN',
    # B7: Functional Activities
    'BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE', 'MEALPREP',
    'EVENTS', 'PAYATTN', 'REMDATES', 'TRAVEL',
    # Milestones
    'NACCNURP',
    # CLS: Linguistic History
    'APREFLAN', 'AYRSPAN', 'AYRENGL', 'APCSPAN', 'APCENGL', 'NACCSPNL', 'NACCENGL'
]

ID_COLS = []

# Drop IDs if present
df = df.drop(columns=[c for c in ID_COLS if c in df.columns])

FORCE_NUMERIC = {
    'NACCAGE', 'EDUC', 'SMOKYRS', 'PACKSPER', 'QUITSMOK', 'HEIGHT', 'WEIGHT', 'NACCBMI',
    'AYRSPAN', 'AYRENGL'
}
FORCE_CATEGORICAL = {
    'SEX', 'MARISTAT', 'NACCLIVS', 'RESIDENC', 'HANDED', 'HISPANIC', 'RACE', 'RACESEC',
    'RACETER', 'PRIMLANG', 'INDEPEND', 'INRELTO', 'INLIVWTH', 'INVISITS', 'INCALLS',
    'NACCFAM', 'NACCMOM', 'NACCDAD', 'ANYMEDS', 'TOBAC30', 'TOBAC100', 'ALCOCCAS',
    'ALCFREQ', 'ALCOHOL', 'ABUSOTHR', 'CVHATT', 'CVAFIB', 'CVANGIO', 'CVBYPASS',
    'CVPACDEF', 'CVPACE', 'CVCHF', 'CVANGINA', 'CVHVALVE', 'CVOTHR', 'HYPERTEN',
    'HYPERCHO', 'CBSTROKE', 'NACCSTYR', 'CBTIA', 'NACCTIYR', 'PD', 'PDYR', 'SEIZURES',
    'NACCTBI', 'DIABETES', 'DIABTYPE', 'B12DEF', 'THYROID', 'ARTHRIT', 'APNEA', 'RBD',
    'INSOMN', 'OTHSLEEP', 'PTSD', 'BIPOLAR', 'SCHIZ', 'DEP2YRS', 'DEPOTHR', 'ANXIETY',
    'OCD', 'INCONTU', 'INCONTF', 'VISION', 'VISCORR', 'VISWCORR', 'HEARING', 'HEARAID',
    'HEARWAID', 'DECSUB', 'DECIN', 'BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE',
    'MEALPREP', 'EVENTS', 'PAYATTN', 'REMDATES', 'TRAVEL', 'NACCNURP',
    'APREFLAN', 'APCSPAN', 'APCENGL', 'NACCSPNL', 'NACCENGL'
}

NUMERIC_COLS, CATEGORICAL_COLS = [], []

NON_MEDICAL_FEATURES = [c for c in ALL_POSSIBLE_FEATURES if c in df.columns]

for col in NON_MEDICAL_FEATURES:
    if col in FORCE_NUMERIC:
        NUMERIC_COLS.append(col)
    elif col in FORCE_CATEGORICAL:
        CATEGORICAL_COLS.append(col)
    else:
        # Heuristic: objects -> categorical; low-cardinality ints also categorical
        if df[col].dtype == "O":
            CATEGORICAL_COLS.append(col)
        else:
            nunq = df[col].nunique(dropna=True)
            if nunq <= 20:
                CATEGORICAL_COLS.append(col)
            else:
                NUMERIC_COLS.append(col)

print(f"Numeric features ({len(NUMERIC_COLS)}): {NUMERIC_COLS}")
print(f"Categorical fields ({len(CATEGORICAL_COLS)}): {CATEGORICAL_COLS}")


# ===========================================
# 2) FEATURE ENGINEERING HOOK
# ===========================================

def feature_engineering_func(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add or modify features here.
    """
    X = X.copy()
    return X