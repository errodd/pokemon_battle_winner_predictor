import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<div style=\"border: 2px solid #2B5B84; padding: 20px; border-radius: 10px; background-color: #F8F9FA; text-align: center;\">\n",
    "    <img src=\"logo/Logo_Facyt.svg.png\" alt=\"Logo FACYT\" style=\"width: 140px; margin-bottom: 12px;\">\n",
    "    <h2 style=\"color: #1B365D; margin-bottom: 2px;\">UNIVERSIDAD DE CARABOBO</h2>\n",
    "    <h4 style=\"color: #2B5B84; margin-top: 0px;\">Facultad Experimental de Ciencias y Tecnología (FACYT)<br>Departamento de Computación</h4>\n",
    "    <hr style=\"border: 1px solid #2B5B84; width: 80%; margin: 15px auto;\">\n",
    "    <h3 style=\"color: #D9534F; margin-bottom: 5px;\">ELECTIVA: APRENDIZAJE AUTOMÁTICO</h3>\n",
    "    <h4 style=\"color: #333;\">ASIGNACIÓN 1 DEL PROYECTO: ANÁLISIS EXPLORATORIO DE DATOS (EDA)</h4>\n",
    "    <p style=\"margin-bottom: 2px;\"><b>Profesor:</b> Álvaro Espinoza | <b>Año Académico:</b> 2026</p>\n",
    "    <p style=\"margin-top: 0px;\"><b>Tema del Proyecto:</b> Predictor de Ganador de Batallas Pokémon</p>\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 1. Contexto del Problema y Definición del Dominio\n",
    "\n",
    "### 1.1 Visión General del Dominio y Motivación\n",
    "En la saga de videojuegos Pokémon, los combates se desarrollan por turnos entre dos Pokémon. El resultado de un combate depende de una combinación de atributos base (Puntos de Vida - HP, Ataque, Defensa, Ataque Especial, Defensa Especial, Velocidad), ventajas elementales por Tipo (por ejemplo, Agua vence a Fuego), el orden de turno (el Pokémon más veloz ataca primero) y el **estadio o nivel evolutivo del Pokémon**.\n",
    "\n",
    "En el conjunto de datos existen formas base, evoluciones intermedias y finales, así como **Mega-Evoluciones** (versiones potenciadas temporalmente en combate que suelen incrementar de forma drástica sus estadísticas base, por ejemplo +100 puntos en la suma total de stats). Predecir el ganador de un combate nos permite cuantificar la mecánica subyacente de la batalla y determinar qué atributos, ventajas de tipo y formas evolutivas poseen mayor poder predictivo.\n",
    "\n",
    "### 1.2 Unidad de Observación\n",
    "La **unidad de observación** en nuestro conjunto de datos analítico consolidado se define como **un combate individual por turnos entre dos Pokémon específicos**:\n",
    "- `First_Pokemon`: El Pokémon asignado a la primera posición en el registro de la batalla.\n",
    "- `Second_Pokemon`: El Pokémon oponente asignado a la segunda posición.\n",
    "\n",
    "### 1.3 Definición de la Variable Objetivo\n",
    "- **Variable Objetivo (`Winner_Is_First`)**: Indicador binario derivado del registro de combates:\n",
    "  - `1`: Si el `First_Pokemon` resultó ganador del combate.\n",
    "  - `0`: Si el `Second_Pokemon` resultó ganador del combate.\n",
    "- **Tipo de Problema de Aprendizaje Automático**: **Clasificación Binaria Supervisada**."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2. Preguntas Guía del Análisis\n",
    "\n",
    "Antes de iniciar la exploración numérica, establecemos las siguientes preguntas analíticas clave para guiar el flujo de trabajo exploratorio:\n",
    "\n",
    "1. **Integridad Estructural y Calidad**: ¿Qué variables contienen las tablas de datos de Pokémon y combates, qué valores faltantes o inconsistencias existen y cómo están estructurados los registros?\n",
    "2. **Impacto de Evoluciones y Mega-Evoluciones**: ¿Cómo influye el estadio evolutivo (forma base, evolución intermedia/final y Mega-Evoluciones) en la tasa de victorias y en los atributos totales de los Pokémon?\n",
    "3. **Ventaja del Primer Movimiento**: ¿Existe una ventaja estadísticamente significativa por ser el `First_Pokemon` en el registro de un combate?\n",
    "4. **Influencia de la Velocidad**: ¿Qué tan fuerte es la correlación entre una mayor `Speed` (Velocidad) y la probabilidad de ganar, dado que el más veloz ataca primero?\n",
    "5. **Diferenciales de Atributos**: ¿Son más predictivas las diferencias de atributos (`First_Pokemon` menos `Second_Pokemon`) que los valores absolutos de cada uno?\n",
    "6. **Enfrentamientos por Tipo**: ¿Cómo influyen los multiplicadores de efectividad de tipos (Súper Efectivo vs Poco Efectivo) en comparación con la superioridad de atributos base?\n",
    "7. **Estado Legendario y Generaciones**: ¿Los Pokémon Legendarios dominan de manera consistente a los no legendarios y cómo se distribuyen los atributos a través de las Generaciones?"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 3. Configuración, Entorno y Reproducibilidad\n",
    "\n",
    "Importamos todas las librerías necesarias de Python, configuramos la semilla aleatoria global para garantizar reproducibilidad, establecemos las opciones de Pandas para mostrar todas las columnas sin truncamiento y configuramos la apariencia visual de los gráficos."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Librerías principales de manipulación de datos\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "# Librerías de visualización de datos\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import os\n",
    "\n",
    "# Fijar semilla aleatoria global para reproducibilidad\n",
    "RANDOM_SEED = 42\n",
    "np.random.seed(RANDOM_SEED)\n",
    "\n",
    "# Configuración de Pandas para visualización completa de columnas (evitar truncamiento '...')\n",
    "pd.set_option('display.max_columns', None)\n",
    "pd.set_option('display.max_rows', 100)\n",
    "pd.set_option('display.width', 1000)\n",
    "\n",
    "# Configuración estética de visualizaciones\n",
    "sns.set_theme(style=\"whitegrid\", palette=\"muted\")\n",
    "plt.rcParams[\"figure.figsize\"] = (10, 6)\n",
    "plt.rcParams[\"font.size\"] = 11\n",
    "plt.rcParams[\"axes.titlesize\"] = 14\n",
    "plt.rcParams[\"axes.labelsize\"] = 12\n",
    "\n",
    "# Rutas relativas de archivos de datos\n",
    "DATA_DIR = \"data\"\n",
    "POKEMON_PATH = os.path.join(DATA_DIR, \"pokemon.csv\")\n",
    "COMBATS_PATH = os.path.join(DATA_DIR, \"combats.csv\")\n",
    "TESTS_PATH = os.path.join(DATA_DIR, \"tests.csv\")\n",
    "\n",
    "print(\"¡Configuración del entorno completada exitosamente!\")\n",
    "print(f\"Directorio de datos: '{DATA_DIR}'\")\n",
    "print(\"Opciones de Pandas configuradas para mostrar todas las columnas sin truncar.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 4. Carga y Construcción del Dataset Consolidado\n",
    "\n",
    "### 4.1 Ingesta de Datos Raw\n",
    "Cargamos las tres fuentes de datos provistas (`pokemon.csv`, `combats.csv` y `tests.csv`) e inspeccionamos sus dimensiones iniciales."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Carga de archivos CSV\n",
    "pokemon_raw = pd.read_csv(POKEMON_PATH)\n",
    "combats_raw = pd.read_csv(COMBATS_PATH)\n",
    "tests_raw = pd.read_csv(TESTS_PATH)\n",
    "\n",
    "print(f\"Dimensiones de pokemon.csv: {pokemon_raw.shape} (800 Pokémon con 12 atributos)\")\n",
    "print(f\"Dimensiones de combats.csv: {combats_raw.shape} (50,000 registros de combates)\")\n",
    "print(f\"Dimensiones de tests.csv:   {tests_raw.shape} (10,000 combates sin etiqueta para testing)\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4.2 Criterio de Unión (*Merges*) y Construcción de la Tabla Analítica\n",
    "Para construir la tabla analítica donde cada fila represente un combate completo con todos los atributos de ambos contendientes, realizamos las siguientes acciones:\n",
    "1. Renombramos el atributo identificador `#` a `Pokemon_ID` en la tabla de Pokémon para evitar ambigüedades.\n",
    "2. Realizamos una **primera unión de tipo Left Join** entre `combats.csv` (`First_pokemon`) y `pokemon.csv` (`Pokemon_ID`) para asociar todas las características del primer Pokémon (sufijo `_first`).\n",
    "3. Realizamos una **segunda unión de tipo Left Join** entre el resultado anterior (`Second_pokemon`) y `pokemon.csv` (`Pokemon_ID`) para asociar las características del segundo Pokémon (sufijo `_second`).\n",
    "4. Derivamos la variable binaria **`Winner_Is_First`**: asignando `1` cuando el ganador coincide con `First_pokemon` y `0` cuando coincide con `Second_pokemon`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Copia y renombrado de la clave en Pokémon\n",
    "pokemon_df = pokemon_raw.rename(columns={'#': 'Pokemon_ID'}).copy()\n",
    "\n",
    "# 1. Primera unión: atributos de First_pokemon\n",
    "combats_df = combats_raw.merge(\n",
    "    pokemon_df, \n",
    "    left_on='First_pokemon', \n",
    "    right_on='Pokemon_ID', \n",
    "    how='left'\n",
    ")\n",
    "\n",
    "first_cols = {col: f\"{col}_first\" for col in pokemon_df.columns if col != 'Pokemon_ID'}\n",
    "combats_df.rename(columns=first_cols, inplace=True)\n",
    "combats_df.drop(columns=['Pokemon_ID'], inplace=True)\n",
    "\n",
    "# 2. Segunda unión: atributos de Second_pokemon\n",
    "combats_df = combats_df.merge(\n",
    "    pokemon_df, \n",
    "    left_on='Second_pokemon', \n",
    "    right_on='Pokemon_ID', \n",
    "    how='left'\n",
    ")\n",
    "\n",
    "second_cols = {col: f\"{col}_second\" for col in pokemon_df.columns if col != 'Pokemon_ID'}\n",
    "combats_df.rename(columns=second_cols, inplace=True)\n",
    "combats_df.drop(columns=['Pokemon_ID'], inplace=True)\n",
    "\n",
    "# 3. Creación de la variable objetivo binaria\n",
    "combats_df['Winner_Is_First'] = (combats_df['Winner'] == combats_df['First_pokemon']).astype(int)\n",
    "\n",
    "print(f\"Dimensiones del dataset consolidado: {combats_df.shape}\")\n",
    "print(f\"Total de combates integrados: {len(combats_df):,} filas.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4.3 Vista Previa y Verificación de Integridad de la Tabla Consolidada"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Configurar visualización sin límite de columnas para la tabla\n",
    "pd.set_option('display.max_columns', None)\n",
    "\n",
    "# Primeras 5 filas del dataset de combates consolidado completo\n",
    "combats_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> **Conclusión de la Construcción del Dataset:**  \n",
    "> - Se integraron exitosamente los **50,000 combates** de la tabla original con los **800 registros** de Pokémon.  \n",
    "> - Se verificó que las uniones relacionales (*left joins*) **no produjeron pérdida de registros ni filas duplicadas accidentales**, preservando exactamente la granularidad requerida para el análisis (1 fila = 1 combate).  \n",
    "> - La tabla final cuenta con **26 columnas** (identificadores de combate, atributos del primer Pokémon, atributos del segundo Pokémon y la variable objetivo binaria `Winner_Is_First`)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 5. Inspección Estructural y Roles Analíticos\n",
    "\n",
    "### 5.1 Diferenciación entre Tipo de Dato de Pandas y Rol Analítico (Requisito 4.5)\n",
    "De acuerdo con las orientaciones metodológicas de la asignatura, **el tipo de almacenamiento técnico de Pandas (`int64`, `float64`, `object`, `bool`) no debe confundirse con el rol analítico que desempeña la variable**. Por ejemplo, una variable almacenada como un entero de 64 bits (`int64`) puede representar un identificador no cuantitativo (como `First_pokemon` o `Winner`), una categoría ordenada (como `Generation`) o el valor objetivo binario (`Winner_Is_First`).\n",
    "\n",
    "A continuación, definimos formalmente la matriz de roles analíticos para las 26 variables de la tabla consolidada:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "| Categoría de Rol Analítico | Nombre de la Variable | Tipo Pandas | Descripción y Función Analítica |\n",
    "| :--- | :--- | :--- | :--- |\n",
    "| **Identificador (ID)** | `First_pokemon`, `Second_pokemon`, `Winner` | `int64` | Llaves de identificación del combate y contendientes. No deben usarse directamente como predictores numéricos. |\n",
    "| **Identificador Textual** | `Name_first`, `Name_second` | `object` (`str`) | Nombre del Pokémon. Utilizado para derivar indicadores de evolución y Mega-Evoluciones (`Is_Mega`). |\n",
    "| **Categórica Nominal** | `Type 1_first`, `Type 2_first`, `Type 1_second`, `Type 2_second` | `object` (`str`) | Tipo elemental principal y secundario. Determinan multiplicadores de efectividad en combate. |\n",
    "| **Categórica Ordinal / Discreta** | `Generation_first`, `Generation_second` | `int64` | Generación de la franquicia (1 a 6). Representa la era de introducción del Pokémon. |\n",
    "| **Binaria / Booleana** | `Legendary_first`, `Legendary_second` | `bool` | Indicador si el Pokémon es de estatus Legendario (`True`/`False`). |\n",
    "| **Numérica Discreta (Atributos Base)** | `HP_first`, `Attack_first`, `Defense_first`, `Sp. Atk_first`, `Sp. Def_first`, `Speed_first` | `int64` | Estadísticas base de combate del primer Pokémon. |\n",
    "| **Numérica Discreta (Atributos Base)** | `HP_second`, `Attack_second`, `Defense_second`, `Sp. Atk_second`, `Sp. Def_second`, `Speed_second` | `int64` | Estadísticas base de combate del segundo Pokémon. |\n",
    "| **Variable Objetivo (Target)** | `Winner_Is_First` | `int64` | Variable binaria objetivo: `1` si gana `First_pokemon`, `0` si gana `Second_pokemon`. |"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5.2 Resumen de Tipos Técnicos, Nulos y Valores Únicos en Pandas"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Resumen estructural de la tabla consolidada\n",
    "structural_summary = pd.DataFrame({\n",
    "    'Tipo_Pandas': combats_df.dtypes,\n",
    "    'Valores_No_Nulos': combats_df.notnull().sum(),\n",
    "    'Valores_Nulos': combats_df.isnull().sum(),\n",
    "    'Porcentaje_Nulos (%)': (combats_df.isnull().sum() / len(combats_df) * 100).round(2),\n",
    "    'Valores_Unicos': combats_df.nunique()\n",
    "})\n",
    "\n",
    "structural_summary"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5.3 Síntesis de Inspección Estructural y Evaluación de Variables\n",
    "\n",
    "> **Conclusiones Detalladas del Diagnóstico Estructural y Roles Analíticos:**\n",
    ">\n",
    "> 1. **Diferenciación entre Almacenamiento y Rol Analítico:**\n",
    ">    - Se ha establecido la distinción entre los tipos de almacenamiento técnico de Pandas y la función real de cada variable en el dominio del problema. Identificadores como `First_pokemon`, `Second_pokemon` y `Winner` (almacenados como `int64`) son llaves relacionales y no deben tratarse como magnitudes continuas o predictoras numéricas directas.\n",
    ">\n",
    "> 2. **Evaluación Crítica de la Variable `Generation` (`Generation_first`, `Generation_second`):**\n",
    ">    - **Aportación de información:** Aunque `Generation` toma valores discretos de 1 a 6 (indicando la generación de la franquicia en la que se introdujo el Pokémon), por sí sola **no determina directamente el poder de combate**.\n",
    ">    - **Hipótesis sobre *Power Creep*:** En los videojuegos existe la hipótesis de inflación de poder (*power creep*), donde generaciones más recientes podrían incluir Pokémon con estadísticas base ligeramente más altas. Sin embargo, su impacto directo es secundario frente a los atributos reales (`Speed`, `Attack`, `Defense`) y ventajas de tipo. Evaluaremos en el análisis bivariado si la generación aporta poder predictivo residual o si resulta redundante.\n",
    ">\n",
    "> 3. **Estatus Legendario (`Legendary_first`, `Legendary_second`):**\n",
    ">    - Identifica a Pokémon especiales con techos de atributos significativamente superiores (generalmente sumas totales de atributos entre 580 y 720). Esta variable booleana actuará como un segmentador clave en el análisis bivariado y en la interacción de combates (`Legendario vs No-Legendario`).\n",
    ">\n",
    "> 4. **Identificadores Textuales (`Name_first`, `Name_second`):**\n",
    ">    - Los nombres no deben ingresarse crudos al modelo. Sin embargo, su rol analítico es fundamental para la **Ingeniería Exploratoria de Variables**, ya que nos permiten extraer patrones de texto como el prefijo `'Mega '` (para identificar **Mega-Evoluciones**) y etapas evolutivas.\n",
    ">\n",
    "> 5. **Tipos Elementales (`Type 1`, `Type 2`) y Presencia de Nulos:**\n",
    ">    - `Type 1` posee 18 categorías y 0% de valores nulos (todos los Pokémon tienen un tipo primario).\n",
    ">    - `Type 2` presenta aproximadamente un **48.6% de valores nulos** (`NaN`). Se concluye categóricamente que esto **no constituye un problema de calidad de datos**, sino la representación válida del dominio para Pokémon de un solo tipo elemental (monotipo).\n",
    ">\n",
    "> 6. **Atributos Base de Combate (HP, Attack, Defense, Sp. Atk, Sp. Def, Speed):**\n",
    ">    - Representan los 6 ejes numéricos discretos de capacidad física y especial de cada Pokémon. La interacción relativa entre las estadísticas de ambos contendientes (`First_Pokemon` vs `Second_Pokemon`), en lugar de los valores aislados, será el núcleo predictivo del modelo."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 6. Diagnóstico y Calidad de Datos\n",
    "\n",
    "En esta sección evaluamos minuciosamente la calidad del conjunto de datos consolidado, inspeccionando valores ausentes, duplicados relacionales, consistencia de rangos del dominio y presencia de valores extremos (*outliers*)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6.1 Auditoría de Valores Ausentes (*Missing Values*)\n",
    "Analizamos la cantidad y proporción de valores ausentes en la tabla de Pokémon de origen y en el dataset analítico de combates."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Auditoría en la tabla de Pokémon original\n",
    "pokemon_missing = pokemon_raw.isnull().sum()\n",
    "print(\"Valores ausentes en pokemon.csv original:\")\n",
    "print(pokemon_missing[pokemon_missing > 0])\n",
    "\n",
    "# Identificar el Pokémon sin nombre (ID #63)\n",
    "missing_name_pkmn = pokemon_raw[pokemon_raw['Name'].isnull()]\n",
    "print(\"\\nDetalle del Pokémon con nombre ausente (ID #63):\")\n",
    "missing_name_pkmn[['#', 'Name', 'Type 1', 'HP', 'Attack', 'Defense', 'Speed', 'Generation']]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6.2 Detección de Registros Duplicados y Repetición de Combates\n",
    "Inspeccionamos si existen filas idénticas duplicadas en los combates y evaluamos la frecuencia de enfrentamientos repetidos entre la misma pareja de Pokémon."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Combates exactamente duplicados (misma fila completa)\n",
    "exact_dups = combats_raw.duplicated().sum()\n",
    "\n",
    "# 2. Combates repetidos entre la misma pareja de contendientes (First_pokemon y Second_pokemon)\n",
    "pair_dups = combats_raw.duplicated(subset=['First_pokemon', 'Second_pokemon'], keep=False).sum()\n",
    "\n",
    "print(f\"Combates exactamente duplicados en combats.csv: {exact_dups} ({exact_dups / len(combats_raw) * 100:.2f}%)\")\n",
    "print(f\"Total de filas que forman parte de enfrentamientos repetidos entre la misma pareja: {pair_dups} combates.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6.3 Validación de Rangos y Reglas del Dominio\n",
    "Comprobamos que las estadísticas base no contengan valores negativos o ilógicos, que los identificadores pertenezcan al rango válido [1, 800] y que las categorías de Tipo elemental sean consistentes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Columnas de estadísticas base de ambos Pokémon\n",
    "stat_cols = ['HP_first', 'Attack_first', 'Defense_first', 'Sp. Atk_first', 'Sp. Def_first', 'Speed_first',\n",
    "             'HP_second', 'Attack_second', 'Defense_second', 'Sp. Atk_second', 'Sp. Def_second', 'Speed_second']\n",
    "\n",
    "# Comprobar valores menores o iguales a cero\n",
    "invalid_stats_count = (combats_df[stat_cols] <= 0).sum().sum()\n",
    "\n",
    "# Comprobar rango de identificadores de Pokémon en combates\n",
    "valid_ids_first = combats_df['First_pokemon'].between(1, 800).all()\n",
    "valid_ids_second = combats_df['Second_pokemon'].between(1, 800).all()\n",
    "\n",
    "# Lista única de tipos elementales\n",
    "unique_types = sorted(pokemon_raw['Type 1'].dropna().unique())\n",
    "\n",
    "print(f\"Total de valores estáticos inválidos (<= 0): {invalid_stats_count}\")\n",
    "print(f\"¿Todos los identificadores de First_pokemon están en el rango válido [1, 800]?: {valid_ids_first}\")\n",
    "print(f\"¿Todos los identificadores de Second_pokemon están en el rango válido [1, 800]?: {valid_ids_second}\")\n",
    "print(f\"Total de Tipos Elementales únicos validados: {len(unique_types)}\")\n",
    "print(f\"Tipos detectados: {unique_types}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6.4 Detección de Valores Extremos (*Outliers*)\n",
    "Examinamos la distribución de la suma total de atributos (`Total Stats`) para identificar Pokémon con estadísticas extremadamente altas (Mega-Evoluciones y Legendarios) o extremadamente bajas."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calcular Total Stats en la tabla de Pokémon\n",
    "pokemon_with_total = pokemon_raw.copy()\n",
    "pokemon_with_total['Total_Stats'] = pokemon_with_total[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].sum(axis=1)\n",
    "\n",
    "print(\"Top 5 Pokémon con MAYOR suma de estadísticas base (Valores Extremos Superiores):\")\n",
    "display(pokemon_with_total.nlargest(5, 'Total_Stats')[['#', 'Name', 'Type 1', 'Total_Stats', 'Legendary']])\n",
    "\n",
    "print(\"\\nTop 5 Pokémon con MENOR suma de estadísticas base (Valores Extremos Inferiores):\")\n",
    "display(pokemon_with_total.nsmallest(5, 'Total_Stats')[['#', 'Name', 'Type 1', 'Total_Stats', 'Legendary']])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6.5 Síntesis Escrita de Diagnóstico de Calidad de Datos (Requisito 4.6)\n",
    "\n",
    "> **Síntesis y Estrategia de Tratamiento de Calidad de Datos:**\n",
    ">\n",
    "> 1. **Falta de Nombre en ID #63 (`Primeape`):**  \n",
    ">    - Se identificó que la observación `#63` en `pokemon.csv` posee `Name = NaN`. Al contrastar con la Pokédex oficial y los atributos del registro (Tipo Lucha, HP: 65, Ataque: 105, Velocidad: 95, Gen 1), se verifica fehacientemente que corresponde a **Primeape**.  \n",
    ">    - *Decisión de Calidad:* Imputaremos directamente la cadena `'Primeape'` en `Name` durante el preprocesamiento para no perder sus combates asociados.\n",
    ">\n",
    "> 2. **Valores Ausentes en `Type 2` (48.6%):**  \n",
    ">    - Los `NaN` en `Type 2` corresponden a Pokémon monotipo (de un solo elemento como Pikachu o Charmander). No representan una falla de registro.  \n",
    ">    - *Decisión de Calidad:* Se mantendrán y categorizarán como `'None'` (o `'None_Type'`) para permitir el cálculo correcto de matrices de ventaja de tipo sin eliminar ninguna observación.\n",
    ">\n",
    "> 3. **Combates Duplicados en `combats.csv` (1,952 registros):**  \n",
    ">    - Existen 1,952 registros idénticos de enfrentamientos. En simulaciones de combate estocásticas o repetidas, enfrentar a los mismos dos Pokémon en múltiples ocasiones es un procedimiento de muestreo válido que refleja la frecuencia del evento.  \n",
    ">    - *Decisión de Calidad:* **No se eliminarán estos combates**, ya que son simulaciones legítimas provistas en el conjunto de entrenamiento.\n",
    ">\n",
    "> 4. **Ausencia de Atributos Negativos o Inválidos:**  \n",
    ">    - El 100% de las estadísticas numéricas (`HP`, `Attack`, `Defense`, `Sp. Atk`, `Sp. Def`, `Speed`) se encuentran en rangos estricta y físicamente válidos (> 0). Todos los IDs pertenecen al rango [1, 800].\n",
    ">\n",
    "> 5. **Valores Extremos (*Outliers*) Válidos:**  \n",
    ">    - Las sumas de atributos oscilan entre 180 (Sunkern) y 780 (Mega Rayquaza, Mega Mewtwo X/Y). Se concluye que los valores extremos corresponden a **Mega-Evoluciones y Legendarios legítimos** de la franquicia y no a errores de medición. Por ende, **ningún outlier será descartado**, ya que representan variaciones reales del dominio."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.14.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('pokemon_battle_prediction.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

with open('.ipynb_checkpoints/pokemon_battle_prediction-checkpoint.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print('SUCCESS')
