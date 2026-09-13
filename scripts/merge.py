#!/usr/bin/env python3
"""
Pipeline de consolidación y unificación de datos para Pokémon Battle Winner Predictor.
Construye el dataset analítico 'pokemon_combats_unified.csv' a partir de 'pokemon.csv' y 'combats.csv',
aplicando estrictas validaciones de schema, integridad referencial, calidad de datos e invariantes.
"""

import sys
from pathlib import Path
import pandas as pd

# Permitir importación de scripts/features.py independientemente del cwd
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from features import calculate_speed_diff, calculate_attack_diff, calculate_defense_diff


def resolve_data_paths():
    """Resuelve dinámicamente las rutas de los archivos CSV de entrada y salida."""
    cwd = Path.cwd()
    if (cwd / 'pokemon.csv').exists() and (cwd / 'combats.csv').exists():
        data_dir = cwd
    elif (cwd / 'data' / 'pokemon.csv').exists():
        data_dir = cwd / 'data'
    elif (PROJECT_ROOT / 'data' / 'pokemon.csv').exists():
        data_dir = PROJECT_ROOT / 'data'
    else:
        data_dir = Path('data')

    pokemon_path = data_dir / 'pokemon.csv'
    combats_path = data_dir / 'combats.csv'
    output_path = data_dir / 'pokemon_combats_unified.csv'

    return pokemon_path, combats_path, output_path


def main():
    pokemon_path, combats_path, output_path = resolve_data_paths()
    print(f"[1/6] Cargando datos desde: {pokemon_path.parent}...")
    
    assert pokemon_path.exists(), f"Error: No se encontró {pokemon_path}"
    assert combats_path.exists(), f"Error: No se encontró {combats_path}"

    pokemon = pd.read_csv(pokemon_path)
    combats = pd.read_csv(combats_path)

    # -------------------------------------------------------------
    # 1. Validaciones Pre-merge: Schema y Columnas Obligatorias
    # -------------------------------------------------------------
    expected_pokemon_cols = [
        '#', 'Name', 'Type 1', 'Type 2', 'HP', 'Attack',
        'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 'Legendary'
    ]
    expected_combats_cols = ['First_pokemon', 'Second_pokemon', 'Winner']

    assert set(expected_pokemon_cols).issubset(pokemon.columns), (
        f"Schema error: Faltan columnas en {pokemon_path.name}. "
        f"Esperadas: {expected_pokemon_cols}, Encontradas: {list(pokemon.columns)}"
    )
    assert set(expected_combats_cols).issubset(combats.columns), (
        f"Schema error: Faltan columnas en {combats_path.name}. "
        f"Esperadas: {expected_combats_cols}, Encontradas: {list(combats.columns)}"
    )

    # -------------------------------------------------------------
    # 2. Imputación de Datos Faltantes Críticos
    # -------------------------------------------------------------
    # Especie #63 posee Name ausente (NaN); corresponde exactamente a Primeape
    primeape_mask = pokemon['#'] == 63
    if pokemon.loc[primeape_mask, 'Name'].isna().any():
        pokemon.loc[primeape_mask, 'Name'] = 'Primeape'
        print("  -> Imputado nombre ausente de especie #63 como 'Primeape'.")

    # -------------------------------------------------------------
    # 3. Validaciones de Integridad de Pokémon
    # -------------------------------------------------------------
    assert pokemon['#'].is_unique, "Invariante violado: Los IDs de Pokémon ('#') no son únicos."
    assert (pokemon['#'] > 0).all(), "Invariante violado: Existen IDs de Pokémon no positivos."
    assert pokemon['Name'].notna().all(), "Invariante violado: Existen Pokémon con Name nulo tras imputación."

    # Columnas de estadísticas base no pueden contener nulos ni valores negativos
    stat_cols = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    for stat in stat_cols:
        assert pokemon[stat].notna().all(), f"Invariante violado: La estadística '{stat}' contiene nulos."
        assert (pokemon[stat] > 0).all(), f"Invariante violado: La estadística '{stat}' contiene valores <= 0."

    # -------------------------------------------------------------
    # 4. Validaciones de Integridad de Combats
    # -------------------------------------------------------------
    initial_combats_rows = len(combats)
    assert initial_combats_rows == 50000, (
        f"Advertencia/Invariante: Se esperaban 50,000 combates, encontrados: {initial_combats_rows}"
    )

    valid_pokemon_ids = set(pokemon['#'])
    assert set(combats['First_pokemon']).issubset(valid_pokemon_ids), (
        "Invariante violado: Existen IDs en 'First_pokemon' que no existen en el catálogo de Pokémon."
    )
    assert set(combats['Second_pokemon']).issubset(valid_pokemon_ids), (
        "Invariante violado: Existen IDs en 'Second_pokemon' que no existen en el catálogo de Pokémon."
    )
    assert set(combats['Winner']).issubset(valid_pokemon_ids), (
        "Invariante violado: Existen IDs en 'Winner' que no existen en el catálogo de Pokémon."
    )

    # Ningún combate puede ser un Pokémon contra sí mismo
    assert (combats['First_pokemon'] != combats['Second_pokemon']).all(), (
        "Invariante violado: Existen combates donde un Pokémon lucha contra sí mismo."
    )

    # El ganador debe ser estrictamente First_pokemon o Second_pokemon
    valid_winner_mask = (combats['Winner'] == combats['First_pokemon']) | (combats['Winner'] == combats['Second_pokemon'])
    assert valid_winner_mask.all(), (
        "Invariante violado: Existen registros donde Winner no coincide ni con First_pokemon ni con Second_pokemon."
    )

    # Verificación de duplicados exactos en combats
    n_exact_duplicates = combats.duplicated().sum()
    print(f"  -> Duplicados exactos detectados en {combats_path.name}: {n_exact_duplicates} (3.9%).")
    assert n_exact_duplicates == 1952, (
        f"Advertencia: Duplicados exactos esperados: 1,952, encontrados: {n_exact_duplicates}."
    )

    # -------------------------------------------------------------
    # 5. Fusión Relacional (Merge)
    # -------------------------------------------------------------
    print("[2/6] Ejecutando merge de combatientes...")
    pokemon_renamed = pokemon.rename(columns={'#': 'id'})

    # Fusión para First_pokemon
    unified = combats.merge(pokemon_renamed, left_on='First_pokemon', right_on='id', how='left')
    unified = unified.rename(columns={col: f"{col}_first" for col in pokemon_renamed.columns if col != 'id'})

    # Fusión para Second_pokemon
    unified = unified.merge(pokemon_renamed, left_on='Second_pokemon', right_on='id', how='left')
    unified = unified.rename(columns={col: f"{col}_second" for col in pokemon_renamed.columns if col != 'id'})

    # Limpieza de columnas temporales de id
    unified = unified.drop(columns=['id_x', 'id_y'], errors='ignore')

    # -------------------------------------------------------------
    # 6. Validaciones Post-merge
    # -------------------------------------------------------------
    print("[3/6] Verificando invariantes post-merge...")
    assert len(unified) == initial_combats_rows, (
        f"Invariante violado: El número de filas cambió tras el merge. Esperado: {initial_combats_rows}, Obtenido: {len(unified)}"
    )
    assert unified['Name_first'].notna().all(), "Invariante violado: Nulos introducidos en Name_first."
    assert unified['Name_second'].notna().all(), "Invariante violado: Nulos introducidos en Name_second."

    for stat in stat_cols:
        assert unified[f"{stat}_first"].notna().all(), f"Invariante violado: Nulos introducidos en {stat}_first."
        assert unified[f"{stat}_second"].notna().all(), f"Invariante violado: Nulos introducidos en {stat}_second."

    # -------------------------------------------------------------
    # 7. Construcción de Target y Validación de Dominio
    # -------------------------------------------------------------
    print("[4/6] Creando target binario...")
    unified['Target_First_Wins'] = (unified['Winner'] == unified['First_pokemon']).astype(int)

    assert set(unified['Target_First_Wins'].unique()).issubset({0, 1}), (
        f"Invariante violado: Target_First_Wins contiene valores fuera de {{0, 1}}: {unified['Target_First_Wins'].unique()}"
    )
    assert unified['Target_First_Wins'].notna().all(), "Invariante violado: Target_First_Wins contiene valores nulos."

    # -------------------------------------------------------------
    # 8. Feature Engineering: Características Diferenciales Básicas
    # -------------------------------------------------------------
    print("[5/6] Calculando características diferenciales vía scripts/features.py...")
    unified['Speed_diff'] = calculate_speed_diff(unified)
    unified['Attack_diff'] = calculate_attack_diff(unified)
    unified['Defense_diff'] = calculate_defense_diff(unified)

    # Validar que no haya nulos en las características creadas
    assert unified['Speed_diff'].notna().all(), "Error: Nulos en Speed_diff."
    assert unified['Attack_diff'].notna().all(), "Error: Nulos en Attack_diff."
    assert unified['Defense_diff'].notna().all(), "Error: Nulos en Defense_diff."

    # -------------------------------------------------------------
    # 9. Exportación y Verificación Final
    # -------------------------------------------------------------
    print(f"[6/6] Guardando tabla unificada en: {output_path}...")
    unified.to_csv(output_path, index=False)

    assert output_path.exists(), f"Error: No se pudo generar el archivo {output_path}"
    assert len(unified) == 50000, f"Error: Filas finales inesperadas: {len(unified)}"
    assert len(unified.columns) == 29, f"Error: Columnas finales inesperadas: {len(unified.columns)} (esperadas 29)"

    print(f"Pipeline completado exitosamente: {len(unified):,} filas x {len(unified.columns)} columnas.")


if __name__ == '__main__':
    main()
