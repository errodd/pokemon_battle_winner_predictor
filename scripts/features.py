"""
Módulo de Feature Engineering para Pokémon Battle Winner Predictor.

Contiene las funciones reutilizables y formalizadas para la extracción y cálculo
de las características maestras basadas en la mecánica de combate Pokémon.
"""

from typing import Optional, Set
import numpy as np
import pandas as pd

# Matriz estándar de efectividad elemental de tipos (Gen 6 standard - 120 enfrentamientos)
TYPE_CHART = {
    ('Normal', 'Rock'): 0.5, ('Normal', 'Ghost'): 0.0, ('Normal', 'Steel'): 0.5,
    ('Fire', 'Fire'): 0.5, ('Fire', 'Water'): 0.5, ('Fire', 'Grass'): 2.0, ('Fire', 'Ice'): 2.0,
    ('Fire', 'Bug'): 2.0, ('Fire', 'Rock'): 0.5, ('Fire', 'Dragon'): 0.5, ('Fire', 'Steel'): 2.0,
    ('Water', 'Fire'): 2.0, ('Water', 'Water'): 0.5, ('Water', 'Grass'): 0.5, ('Water', 'Ground'): 2.0,
    ('Water', 'Rock'): 2.0, ('Water', 'Dragon'): 0.5,
    ('Electric', 'Water'): 2.0, ('Electric', 'Electric'): 0.5, ('Electric', 'Grass'): 0.5,
    ('Electric', 'Ground'): 0.0, ('Electric', 'Flying'): 2.0, ('Electric', 'Dragon'): 0.5,
    ('Grass', 'Fire'): 0.5, ('Grass', 'Water'): 2.0, ('Grass', 'Grass'): 0.5, ('Grass', 'Poison'): 0.5,
    ('Grass', 'Ground'): 2.0, ('Grass', 'Flying'): 0.5, ('Grass', 'Bug'): 0.5, ('Grass', 'Rock'): 2.0,
    ('Grass', 'Dragon'): 0.5, ('Grass', 'Steel'): 0.5,
    ('Ice', 'Fire'): 0.5, ('Ice', 'Water'): 0.5, ('Ice', 'Grass'): 2.0, ('Ice', 'Ice'): 0.5,
    ('Ice', 'Ground'): 2.0, ('Ice', 'Flying'): 2.0, ('Ice', 'Dragon'): 2.0, ('Ice', 'Steel'): 0.5,
    ('Fighting', 'Normal'): 2.0, ('Fighting', 'Ice'): 2.0, ('Fighting', 'Poison'): 0.5,
    ('Fighting', 'Flying'): 0.5, ('Fighting', 'Psychic'): 0.5, ('Fighting', 'Bug'): 0.5,
    ('Fighting', 'Rock'): 2.0, ('Fighting', 'Ghost'): 0.0, ('Fighting', 'Dark'): 2.0,
    ('Fighting', 'Steel'): 2.0, ('Fighting', 'Fairy'): 0.5,
    ('Poison', 'Grass'): 2.0, ('Poison', 'Poison'): 0.5, ('Poison', 'Ground'): 0.5,
    ('Poison', 'Rock'): 0.5, ('Poison', 'Ghost'): 0.5, ('Poison', 'Steel'): 0.0, ('Poison', 'Fairy'): 2.0,
    ('Ground', 'Fire'): 2.0, ('Ground', 'Electric'): 2.0, ('Ground', 'Grass'): 0.5,
    ('Ground', 'Poison'): 2.0, ('Ground', 'Flying'): 0.0, ('Ground', 'Bug'): 0.5,
    ('Ground', 'Rock'): 2.0, ('Ground', 'Steel'): 2.0,
    ('Flying', 'Electric'): 0.5, ('Flying', 'Grass'): 2.0, ('Flying', 'Fighting'): 2.0,
    ('Flying', 'Bug'): 2.0, ('Flying', 'Rock'): 0.5, ('Flying', 'Steel'): 0.5,
    ('Psychic', 'Fighting'): 2.0, ('Psychic', 'Poison'): 2.0, ('Psychic', 'Psychic'): 0.5,
    ('Psychic', 'Dark'): 0.0, ('Psychic', 'Steel'): 0.5,
    ('Bug', 'Fire'): 0.5, ('Bug', 'Grass'): 2.0, ('Bug', 'Fighting'): 0.5, ('Bug', 'Poison'): 0.5,
    ('Bug', 'Flying'): 0.5, ('Bug', 'Psychic'): 2.0, ('Bug', 'Ghost'): 0.5, ('Bug', 'Dark'): 2.0,
    ('Bug', 'Steel'): 0.5, ('Bug', 'Fairy'): 0.5,
    ('Rock', 'Fire'): 2.0, ('Rock', 'Ice'): 2.0, ('Rock', 'Fighting'): 0.5, ('Rock', 'Ground'): 0.5,
    ('Rock', 'Flying'): 2.0, ('Rock', 'Bug'): 2.0, ('Rock', 'Steel'): 0.5,
    ('Ghost', 'Normal'): 0.0, ('Ghost', 'Psychic'): 2.0, ('Ghost', 'Ghost'): 2.0, ('Ghost', 'Dark'): 0.5,
    ('Dragon', 'Dragon'): 2.0, ('Dragon', 'Steel'): 0.5, ('Dragon', 'Fairy'): 0.0,
    ('Dark', 'Fighting'): 0.5, ('Dark', 'Psychic'): 2.0, ('Dark', 'Ghost'): 2.0,
    ('Dark', 'Dark'): 0.5, ('Dark', 'Fairy'): 0.5,
    ('Steel', 'Fire'): 0.5, ('Steel', 'Water'): 0.5, ('Steel', 'Electric'): 0.5, ('Steel', 'Ice'): 2.0,
    ('Steel', 'Rock'): 2.0, ('Steel', 'Steel'): 0.5, ('Steel', 'Fairy'): 2.0,
    ('Fairy', 'Fire'): 0.5, ('Fairy', 'Fighting'): 2.0, ('Fairy', 'Poison'): 0.5,
    ('Fairy', 'Dragon'): 2.0, ('Fairy', 'Dark'): 2.0, ('Fairy', 'Steel'): 0.5,
}

SPECIAL_FORM_CATEGORIES: Set[str] = {
    "Mega-Evolution",
    "Primal Reversion",
    "Alt. Combat Form",
    "Legendary"
}


def calculate_speed_diff(df: pd.DataFrame) -> pd.Series:
    """Calcula el diferencial de velocidad: Speed_first - Speed_second."""
    return df['Speed_first'] - df['Speed_second']


def calculate_attack_diff(df: pd.DataFrame) -> pd.Series:
    """Calcula el diferencial de ataque: Attack_first - Attack_second."""
    return df['Attack_first'] - df['Attack_second']


def calculate_defense_diff(df: pd.DataFrame) -> pd.Series:
    """Calcula el diferencial de defensa: Defense_first - Defense_second."""
    return df['Defense_first'] - df['Defense_second']


def calculate_stat_total_diff(df: pd.DataFrame) -> pd.Series:
    """
    Calcula el diferencial de estadísticas totales: Total_first - Total_second.
    Si no existen las columnas Total_first / Total_second, se calculan sumando las 6 base stats.
    """
    stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    if 'Total_first' in df.columns and 'Total_second' in df.columns:
        total_1 = df['Total_first']
        total_2 = df['Total_second']
    elif 'Total_diff' in df.columns:
        return df['Total_diff']
    else:
        total_1 = sum(df[f"{s}_first"] for s in stats)
        total_2 = sum(df[f"{s}_second"] for s in stats)
    return total_1 - total_2


def calculate_atk_def_penetration_diff(df: pd.DataFrame) -> pd.Series:
    """
    Calcula la penetración física diferencial:
    (Attack_1 / Defense_2) - (Attack_2 / Defense_1).
    """
    penetration_1 = df['Attack_first'] / df['Defense_second'].replace(0, 1)
    penetration_2 = df['Attack_second'] / df['Defense_first'].replace(0, 1)
    return penetration_1 - penetration_2


def classify_form(name: str, legendary: bool) -> str:
    """
    Clasifica taxonómicamente la forma de un Pokémon según su nombre y su estatus legendario.
    """
    n = str(name)
    if "Mega " in n:
        return "Mega-Evolution"
    if "Primal " in n:
        return "Primal Reversion"
    if any(f in n for f in ["Forme", "Mode", "Unbound", "Black Kyurem", "White Kyurem", "Rotom"]):
        return "Alt. Combat Form"
    if any(v in n for v in ["Size", "Cloak", "Male", "Female"]):
        return "Variant"
    if legendary:
        return "Legendary"
    return "Standard"


def calculate_special_form_advantage(df: pd.DataFrame,
                                      special_forms: Optional[Set[str]] = None) -> pd.Series:
    """
    Calcula la ventaja binaria/ternaria de forma especial entre ambos contendientes:
    is_special_first - is_special_second in {-1, 0, 1}.
    """
    forms = special_forms if special_forms is not None else SPECIAL_FORM_CATEGORIES
    
    if 'form_first' in df.columns:
        f1 = df['form_first']
    else:
        f1 = [classify_form(n, l) for n, l in zip(df['Name_first'], df['Legendary_first'])]
        
    if 'form_second' in df.columns:
        f2 = df['form_second']
    else:
        f2 = [classify_form(n, l) for n, l in zip(df['Name_second'], df['Legendary_second'])]
        
    s1 = pd.Series(f1, index=df.index).isin(forms).astype(int)
    s2 = pd.Series(f2, index=df.index).isin(forms).astype(int)
    return s1 - s2


def calculate_type_multiplier(atk_t1: str, atk_t2: str, def_t1: str, def_t2: str,
                              type_chart: Optional[dict] = None) -> float:
    """
    Calcula el multiplicador ofensivo máximo de un atacante contra un defensor considerando tipos duales.
    Cualquier matchup no explícito en la tabla se evalúa como neutral (1.0).
    """
    chart = type_chart if type_chart is not None else TYPE_CHART
    def val(a, d):
        if not isinstance(a, str) or not isinstance(d, str) or pd.isna(a) or pd.isna(d):
            return 1.0
        return chart.get((a, d), 1.0)
    m1 = val(atk_t1, def_t1) * val(atk_t1, def_t2)
    m2 = val(atk_t2, def_t1) * val(atk_t2, def_t2) if (isinstance(atk_t2, str) and not pd.isna(atk_t2)) else 1.0
    return max(m1, m2)


def calculate_type_advantage_ratio(df: pd.DataFrame, type_chart: Optional[dict] = None) -> pd.Series:
    """
    Calcula la razón de ventaja elemental simétrica:
    log2((Eff_1v2 + 0.1) / (Eff_2v1 + 0.1)).
    """
    chart = type_chart if type_chart is not None else TYPE_CHART
    
    if 'Eff_1v2' in df.columns and 'Eff_2v1' in df.columns:
        eff_1v2 = df['Eff_1v2']
        eff_2v1 = df['Eff_2v1']
    else:
        eff_1v2 = [
            calculate_type_multiplier(t1_1, t2_1, t1_2, t2_2, chart)
            for t1_1, t2_1, t1_2, t2_2 in zip(
                df['Type 1_first'], df['Type 2_first'], df['Type 1_second'], df['Type 2_second']
            )
        ]
        eff_2v1 = [
            calculate_type_multiplier(t1_2, t2_2, t1_1, t2_1, chart)
            for t1_1, t2_1, t1_2, t2_2 in zip(
                df['Type 1_first'], df['Type 2_first'], df['Type 1_second'], df['Type 2_second']
            )
        ]
        eff_1v2 = pd.Series(eff_1v2, index=df.index)
        eff_2v1 = pd.Series(eff_2v1, index=df.index)
        
    return np.log2((eff_1v2 + 0.1) / (eff_2v1 + 0.1))


def add_master_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica y añade las 5 master engineered features a una copia del DataFrame:
    1. Speed_diff
    2. Stat_Total_Diff
    3. Atk_Def_Penetration_Diff
    4. Special_Form_Advantage
    5. Type_Advantage_Ratio
    """
    df_out = df.copy()
    df_out['Speed_diff'] = calculate_speed_diff(df_out)
    df_out['Stat_Total_Diff'] = calculate_stat_total_diff(df_out)
    df_out['Atk_Def_Penetration_Diff'] = calculate_atk_def_penetration_diff(df_out)
    df_out['Special_Form_Advantage'] = calculate_special_form_advantage(df_out)
    df_out['Type_Advantage_Ratio'] = calculate_type_advantage_ratio(df_out)
    return df_out


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Alias compatible para add_master_features."""
    return add_master_features(df)
