import pandas as pd

# 1. Cargar los archivos originales
pokemon = pd.read_csv('pokemon.csv')
combats = pd.read_csv('combats.csv')

# Renombrar la columna '#' a 'id' para mayor claridad
pokemon = pokemon.rename(columns={'#': 'id'})

# 2. Unir datos para el primer Pokémon (First_pokemon)
unified = combats.merge(pokemon, left_on='First_pokemon', right_on='id', how='left')
unified = unified.rename(columns={col: f"{col}_first" for col in pokemon.columns if col != 'id'})

# 3. Unir datos para el segundo Pokémon (Second_pokemon)
unified = unified.merge(pokemon, left_on='Second_pokemon', right_on='id', how='left')
unified = unified.rename(columns={col: f"{col}_second" for col in pokemon.columns if col != 'id'})

# Eliminar columnas auxiliares redundantes
unified = unified.drop(columns=['id_x', 'id_y'], errors='ignore')

# 4. Crear la variable objetivo binaria (1 = Gana First_pokemon, 0 = Gana Second_pokemon)
unified['Target_First_Wins'] = (unified['Winner'] == unified['First_pokemon']).astype(int)

# 5. Feature Engineering: Crear características relativas/diferenciales
unified['Speed_diff'] = unified['Speed_first'] - unified['Speed_second']
unified['Attack_diff'] = unified['Attack_first'] - unified['Attack_second']
unified['Defense_diff'] = unified['Defense_first'] - unified['Defense_second']

# 6. Exportar a un solo CSV unificado
unified.to_csv('pokemon_combats_unified.csv', index=False)
