import pandas as pd
import numpy as np

def calculate_speed_diff(df):
    return df['Speed_first'] - df['Speed_second']

def calculate_stat_total_diff(df):
    # Asume que existe una columna 'Total_Stats_first' o se calcula sumando las base stats
    # Por seguridad calcularemos asumiendo las columnas base si no existe Total
    stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    for stat in stats:
        if f"{stat}_first" not in df.columns:
            # Revert to standard naming if different
            pass
            
    # Asumiendo que las columnas existen:
    if 'Total_first' in df.columns and 'Total_second' in df.columns:
         return df['Total_first'] - df['Total_second']
    
    return np.zeros(len(df)) # Placeholder

def calculate_atk_def_penetration_diff(df):
    # (Attack_1 / Defense_2) - (Attack_2 / Defense_1)
    penetration_1 = df['Attack_first'] / df['Defense_second'].replace(0, 1)
    penetration_2 = df['Attack_second'] / df['Defense_first'].replace(0, 1)
    return penetration_1 - penetration_2

def calculate_special_form_advantage(df):
    # Asume que Is_Special_first es boolean/int
    if 'Is_Special_first' in df.columns and 'Is_Special_second' in df.columns:
        return df['Is_Special_first'].astype(int) - df['Is_Special_second'].astype(int)
    return np.zeros(len(df))

def add_engineered_features(df):
    """
    Applies all engineered features to the unified dataframe.
    """
    df = df.copy()
    
    if 'Speed_first' in df.columns and 'Speed_second' in df.columns:
        df['Speed_Diff'] = calculate_speed_diff(df)
        
    if 'Attack_first' in df.columns and 'Defense_second' in df.columns:
        df['Atk_Def_Penetration_Diff'] = calculate_atk_def_penetration_diff(df)
        
    # Agrega más features comprobando primero la existencia de las columnas
    
    return df
