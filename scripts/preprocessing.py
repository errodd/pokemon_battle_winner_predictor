"""
Módulo de preprocesamiento para Pokémon Battle Winner Predictor.

Contiene los grupos de columnas, las representaciones (brazos) a comparar, la fábrica del
ColumnTransformer, la fábrica del pipeline completo y la regla de velocidad usada como baseline.
Lo importan el notebook de preparación (02) y el de modelado (03), para no copiar código ni constantes.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from features import create_pokemon_features

STAT_COLS: List[str] = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]

# Grupos de columnas que produce create_pokemon_features
RAW_NUMERIC: List[str] = [f"{stat}_{side}" for stat in STAT_COLS for side in ("first", "second")]
ENGINEERED_NUMERIC: List[str] = ["Speed_diff", "Stat_Total_Diff", "Atk_Def_Penetration_Diff", "Type_Advantage_Ratio"]
NOMINAL: List[str] = ["Type 1_first", "Type 2_first", "Type 1_second", "Type 2_second"]
ORDINAL: List[str] = ["Special_Form_Advantage"]
BINARY: List[str] = ["Legendary_first", "Legendary_second"]
EXCLUDED: Dict[str, str] = {
    "First_pokemon": "Fighter ID -> memorization hazard; used only for grouped partitioning",
    "Second_pokemon": "Fighter ID -> memorization hazard; used only for grouped partitioning",
    "Name_first": "Raw text; abstracted into Special_Form_Advantage",
    "Name_second": "Raw text; abstracted into Special_Form_Advantage",
    "form_first": "Internal categorical helper; abstracted into Special_Form_Advantage",
    "form_second": "Internal categorical helper; abstracted into Special_Form_Advantage",
    "Generation_first": "Release-era catalogue attribute, not a battle mechanic (domain-based)",
    "Generation_second": "Release-era catalogue attribute, not a battle mechanic (domain-based)",
}

# Representaciones a comparar en el modelado (con los mismos folds).
# "nominal_min_frequency": None = one-hot completo; 0.01 = tipos con < 1% de frecuencia agrupados.
FEATURE_SETS: Dict[str, Dict[str, Any]] = {
    "A_raw": {"numeric": RAW_NUMERIC},
    "B_engineered": {"numeric": ENGINEERED_NUMERIC, "passthrough": ORDINAL},
    "C_combined": {"numeric": RAW_NUMERIC + ENGINEERED_NUMERIC, "passthrough": ORDINAL},
    "full": {
        "numeric": RAW_NUMERIC + ENGINEERED_NUMERIC,
        "nominal": NOMINAL,
        "nominal_min_frequency": 0.01,
        "passthrough": ORDINAL + BINARY,
    },
    "full_onehot": {
        "numeric": RAW_NUMERIC + ENGINEERED_NUMERIC,
        "nominal": NOMINAL,
        "nominal_min_frequency": None,
        "passthrough": ORDINAL + BINARY,
    },
}


def make_nominal_pipeline(min_frequency=0.01) -> Pipeline:
    """Type 2 ausente = monotipo ("None"); one-hot completo (None) o con tipos infrecuentes agrupados."""
    if min_frequency is None:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        encoder = OneHotEncoder(min_frequency=min_frequency, handle_unknown="infrequent_if_exist",
                                sparse_output=False)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        ("encoder", encoder),
    ])


def to_float(X) -> np.ndarray:
    """Convierte columnas ordinales/booleanas a float para que la matriz final no quede como object."""
    return np.asarray(X, dtype=float)


def make_preprocessor(feature_set: str = "C_combined") -> ColumnTransformer:
    """Construye un ColumnTransformer nuevo y sin ajustar para el brazo indicado."""
    spec = FEATURE_SETS[feature_set]
    transformers = []
    if "numeric" in spec:
        transformers.append(("numeric", Pipeline([("scaler", StandardScaler())]), spec["numeric"]))
    if "nominal" in spec:
        transformers.append(("nominal", make_nominal_pipeline(spec["nominal_min_frequency"]), spec["nominal"]))
    if "passthrough" in spec:
        transformers.append(("passthrough", FunctionTransformer(to_float, feature_names_out="one-to-one"),
                             spec["passthrough"]))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def make_preparation_pipeline(catalog: pd.DataFrame, feature_set: str = "C_combined") -> Pipeline:
    """Pipeline sin modelo: pares crudos -> create_pokemon_features -> ColumnTransformer."""
    return Pipeline([
        ("feature_engineering", FunctionTransformer(create_pokemon_features, validate=False,
                                                    kw_args={"catalog": catalog.copy()})),
        ("preprocessor", make_preprocessor(feature_set)),
    ])


def build_model_pipeline(estimator, catalog: pd.DataFrame, feature_set: str = "C_combined") -> Pipeline:
    """
    Pipeline completo, nuevo y sin ajustar (pares crudos -> features -> preprocesamiento -> modelo).
    El estimador se clona, de modo que dos pipelines nunca comparten estado aprendido.
    """
    model = clone(estimator) if hasattr(estimator, "get_params") else estimator
    steps = make_preparation_pipeline(catalog, feature_set).steps
    return Pipeline(steps + [("model", model)])


def speed_rule_predict(features: pd.DataFrame) -> np.ndarray:
    """
    Baseline de velocidad del EDA: gana First si es estrictamente más rápido; en empate gana Second.
    Recibe la salida de create_pokemon_features. Debe evaluarse en los mismos folds y holdout que los modelos.
    """
    return (features["Speed_first"] > features["Speed_second"]).astype(int).to_numpy()
