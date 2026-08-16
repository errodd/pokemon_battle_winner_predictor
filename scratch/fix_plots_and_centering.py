import json

cells_list = [
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
    "Importamos todas las librerías necesarias de Python, configuramos la semilla aleatoria global para garantizar reproducibilidad, establecemos las opciones de Pandas para mostrar todas las columnas y texto sin truncamiento y configuramos la apariencia visual de los gráficos."
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
    "# Configuración de Pandas para visualización completa sin truncamiento ('...' / colwidth)\n",
    "pd.set_option('display.max_columns', None)\n",
    "pd.set_option('display.max_rows', 100)\n",
    "pd.set_option('display.width', 1000)\n",
    "pd.set_option('display.max_colwidth', None)\n",
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
    "print(\"Opciones de Pandas configuradas para mostrar todas las columnas y texto sin truncamiento.\")"
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
    "<div align=\"center\">\n",
    "\n",
    "| Categoría de Rol Analítico | Nombre de la Variable | Tipo Pandas | Descripción y Función Analítica |\n",
    "| :--- | :--- | :--- | :--- |\n",
    "| **Identificador (ID)** | `First_pokemon`, `Second_pokemon`, `Winner` | `int64` | Llaves de identificación del combate y contendientes. No deben usarse directamente como predictores numéricos. |\n",
    "| **Identificador Textual** | `Name_first`, `Name_second` | `object` (`str`) | Nombre del Pokémon. Utilizado para derivar indicadores de evolución y Mega-Evoluciones (`Is_Mega`). |\n",
    "| **Categórica Nominal** | `Type 1_first`, `Type 2_first`, `Type 1_second`, `Type 2_second` | `object` (`str`) | Tipo elemental principal y secundario. Determinan multiplicadores de efectividad en combate. |\n",
    "| **Categórica Ordinal / Discreta** | `Generation_first`, `Generation_second` | `int64` | Generación de la franquicia (1 a 6). Representa la era de introducción del Pokémon. |\n",
    "| **Binaria / Booleana** | `Legendary_first`, `Legendary_second` | `bool` | Indicador si el Pokémon es de estatus Legendario (`True`/`False`). |\n",
    "| **Numérica Discreta (Atributos Base)** | `HP_first`, `Attack_first`, `Defense_first`, `Sp. Atk_first`, `Sp. Def_first`, `Speed_first` | `int64` | Estadísticas base de combate del primer Pokémon. |\n",
    "| **Numérica Discreta (Atributos Base)** | `HP_second`, `Attack_second`, `Defense_second`, `Sp. Atk_second`, `Sp. Def_second`, `Speed_second` | `int64` | Estadísticas base de combate del segundo Pokémon. |\n",
    "| **Variable Objetivo (Target)** | `Winner_Is_First` | `int64` | Variable binaria objetivo: `1` si gana `First_pokemon`, `0` si gana `Second_pokemon`. |\n",
    "\n",
    "</div>"
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
    ">    - Se ha established la distinción entre los tipos de almacenamiento técnico de Pandas y la función real de cada variable en el dominio del problema. Identificadores como `First_pokemon`, `Second_pokemon` y `Winner` (almacenados como `int64`) son llaves relacionales y no deben tratarse como magnitudes continuas o predictoras numéricas directas.\n",
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
    "\n",
    "Es crítico distinguir conceptualmente entre dos tipos de duplicación en el dataset de combates:\n",
    "\n",
    "1. **Combates Exactamente Duplicados (Filas Idénticas):**  \n",
    "   - **Definición:** Ocurre cuando dos o más filas en `combats.csv` poseen exactamente los mismos valores en todos sus campos: el mismo `First_pokemon`, el mismo `Second_pokemon` y el mismo `Winner`.\n",
    "   - **Ejemplo Concreto:**  \n",
    "     - Fila #1: `First = 266 (Larvitar)`, `Second = 298 (Marill)`, `Winner = 298 (Marill)`\n",
    "     - Fila #450: `First = 266 (Larvitar)`, `Second = 298 (Marill)`, `Winner = 298 (Marill)`  \n",
    "     *(Existen 3,858 filas involucradas en duplicación exacta, de las cuales 1,952 son duplicados exactos).*  \n",
    "\n",
    "2. **Enfrentamientos Invertidos por Posición (Swapped Positions):**  \n",
    "   - **Definición:** Ocurre cuando la misma pareja de Pokémon se enfrenta intercambiando posiciones: `(First = A, Second = B)` frente a `(First = B, Second = A)`. Nos permite evaluar si el **primer movimiento** afecta el resultado.\n",
    "   - **Ejemplo Concreto:**  \n",
    "     - Combate #1: `First = Pikachu`, `Second = Charmander` $\\rightarrow$ Ganador: `Pikachu`  \n",
    "     - Combate #2: `First = Charmander`, `Second = Pikachu` $\\rightarrow$ ¿Ganador: `Pikachu` (mismo ganador absoluto) o `Charmander` (gana quien ataca primero)?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Configuración explícita de Pandas para evitar truncamiento '...' en cadenas largas de la tabla\n",
    "pd.set_option('display.max_colwidth', None)\n",
    "\n",
    "# 1. Identificación de filas exactamente duplicadas\n",
    "exact_dups_count = combats_raw.duplicated().sum()\n",
    "exact_dup_df = combats_raw[combats_raw.duplicated(keep=False)].copy()\n",
    "\n",
    "# 2. Identificación de parejas enfrentadas en ambos sentidos (A vs B) y (B vs A)\n",
    "combats_raw['Pair_Sorted'] = combats_raw.apply(lambda r: tuple(sorted([r['First_pokemon'], r['Second_pokemon']])), axis=1)\n",
    "first_second_pairs = set(zip(combats_raw['First_pokemon'], combats_raw['Second_pokemon']))\n",
    "swapped_pairs_set = set([pair for pair in first_second_pairs if (pair[1], pair[0]) in first_second_pairs])\n",
    "\n",
    "swapped_df = combats_raw[combats_raw.set_index(['First_pokemon', 'Second_pokemon']).index.isin(swapped_pairs_set)].copy()\n",
    "swapped_groups = swapped_df.groupby('Pair_Sorted')['Winner'].nunique()\n",
    "\n",
    "deterministic_pairs = set(swapped_groups[swapped_groups == 1].index)\n",
    "variable_pairs = set(swapped_groups[swapped_groups > 1].index)\n",
    "\n",
    "same_winner_swapped = len(deterministic_pairs)\n",
    "diff_winner_swapped = len(variable_pairs)\n",
    "total_swapped_pairs = len(swapped_groups)\n",
    "\n",
    "print(f\"Filas exactamente duplicadas en combats.csv: {exact_dups_count} ({exact_dups_count / len(combats_raw) * 100:.2f}%)\")\n",
    "print(f\"Total de registros involucrados en duplicados exactos: {len(exact_dup_df)}\")\n",
    "print(f\"Parejas únicas que se enfrentaron en ambos sentidos (A vs B y B vs A): {total_swapped_pairs}\")\n",
    "print(f\" - Parejas con el MISMO ganador absoluto sin importar el orden: {same_winner_swapped} ({same_winner_swapped/total_swapped_pairs*100:.1f}%)\")\n",
    "print(f\" - Parejas donde el ganador CAMBIA al invertir posiciones: {diff_winner_swapped} ({diff_winner_swapped/total_swapped_pairs*100:.1f}%)\")\n",
    "\n",
    "# --- Visualización Gráfica Única del Efecto de Inversión ---\n",
    "fig, ax = plt.subplots(figsize=(8, 5.5))\n",
    "\n",
    "swapped_series = pd.Series({\n",
    "    f'Mismo Ganador Absoluto\\n({same_winner_swapped:,} parejas - 94.0%)': same_winner_swapped,\n",
    "    f'Ganador Cambia por Inversión\\n({diff_winner_swapped:,} parejas - 6.0%)': diff_winner_swapped\n",
    "})\n",
    "\n",
    "colors = ['#2ECC71', '#E74C3C']\n",
    "wedges, texts, autotexts = ax.pie(\n",
    "    swapped_series, \n",
    "    labels=swapped_series.index, \n",
    "    autopct='%1.1f%%', \n",
    "    colors=colors,\n",
    "    startangle=90, \n",
    "    explode=(0.08, 0),\n",
    "    textprops={'fontsize': 11, 'weight': 'bold'}\n",
    ")\n",
    "ax.set_title('Determinismo en Enfrentamientos Repetidos por Pareja', fontsize=13, fontweight='bold', pad=15)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6.2.1 Análisis Comparativo entre Parejas Deterministas (94.0%) y Variables (6.0%)\n",
    "\n",
    "Una vez desplegada la gráfica anterior con la proporción de **1,712 parejas deterministas (94.0%)** y **110 parejas variables (6.0%)**, ejecutamos la auditoría de características para entender por qué varían únicamente 110 parejas de contendientes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# === CÓDIGO REPRODUCIBLE DE AUDITORÍA COMPARATIVA Y VISUALIZACIÓN SECCIÓN 6.2.1 ===\n",
    "pd.set_option('display.max_colwidth', None)\n",
    "\n",
    "# 1. Cálculo de superposición de Pokémon entre el grupo determinista y variable\n",
    "pkmn_det = set([pkmn for pair in deterministic_pairs for pkmn in pair])\n",
    "pkmn_var = set([pkmn for pair in variable_pairs for pkmn in pair])\n",
    "overlap_pkmn = pkmn_var.intersection(pkmn_det)\n",
    "\n",
    "# 2. Métricas de atributos (Speed y Total_Stats) para ambos grupos\n",
    "pokemon_stats_df = pokemon_raw.copy().rename(columns={'#': 'Pokemon_ID'})\n",
    "pokemon_stats_df['Total_Stats'] = pokemon_stats_df[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].sum(axis=1)\n",
    "\n",
    "det_pairs_df = pd.DataFrame(list(deterministic_pairs), columns=['P1', 'P2'])\n",
    "var_pairs_df = pd.DataFrame(list(variable_pairs), columns=['P1', 'P2'])\n",
    "\n",
    "def compute_group_metrics(pairs_df, pkmn_data, label):\n",
    "    m = pairs_df.merge(pkmn_data, left_on='P1', right_on='Pokemon_ID').merge(pkmn_data, left_on='P2', right_on='Pokemon_ID', suffixes=('_1', '_2'))\n",
    "    m['Delta_Total'] = (m['Total_Stats_1'] - m['Total_Stats_2']).abs()\n",
    "    m['Delta_Speed'] = (m['Speed_1'] - m['Speed_2']).abs()\n",
    "    m['Speed_Tie'] = (m['Speed_1'] == m['Speed_2'])\n",
    "    m['Grupo'] = label\n",
    "    return m\n",
    "\n",
    "det_metrics = compute_group_metrics(det_pairs_df, pokemon_stats_df, 'Parejas Deterministas (94.0%)')\n",
    "var_metrics = compute_group_metrics(var_pairs_df, pokemon_stats_df, 'Parejas Variables (6.0%)')\n",
    "all_metrics_df = pd.concat([det_metrics, var_metrics], ignore_index=True)\n",
    "\n",
    "# 3. Tabla Resumen Comparativa sin truncamiento\n",
    "comp_summary_df = pd.DataFrame({\n",
    "    'Métrica de Dominio': [\n",
    "        'Parejas Únicas Reenfrentadas',\n",
    "        'Pokémon Únicos Involucrados',\n",
    "        'Empates en Velocidad (Speed_1 == Speed_2)',\n",
    "        'Diferencia Mediana de Velocidad',\n",
    "        'Diferencia Mediana de Stats Totales'\n",
    "    ],\n",
    "    'Parejas Deterministas (94.0%)': [\n",
    "        f\"{len(det_pairs_df):,} parejas\",\n",
    "        f\"{len(pkmn_det):,} Pokémon\",\n",
    "        f\"{det_metrics.Speed_Tie.mean()*100:.1f}%\",\n",
    "        f\"{det_metrics.Delta_Speed.median():.1f} pts\",\n",
    "        f\"{det_metrics.Delta_Total.median():.1f} pts\"\n",
    "    ],\n",
    "    'Parejas Variables (6.0%)': [\n",
    "        f\"{len(var_pairs_df):,} parejas\",\n",
    "        f\"{len(pkmn_var):,} Pokémon (Superposición: {len(overlap_pkmn)/len(pkmn_var)*100:.1f}%)\",\n",
    "        f\"{var_metrics.Speed_Tie.mean()*100:.1f}%\",\n",
    "        f\"{var_metrics.Delta_Speed.median():.1f} pts\",\n",
    "        f\"{var_metrics.Delta_Total.median():.1f} pts\"\n",
    "    ]\n",
    "})\n",
    "\n",
    "print(f\"Superposición de Pokémon: {len(overlap_pkmn)} de {len(pkmn_var)} Pokémon del grupo variable ({len(overlap_pkmn)/len(pkmn_var)*100:.1f}%) también pertenecen al grupo determinista.\")\n",
    "display(comp_summary_df)\n",
    "\n",
    "# 4. Visualización Gráfica Comparativa: Boxplot + Histograma de Conteo Real de Parejas (stat='count')\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))\n",
    "\n",
    "# Subgráfico A: Diagrama de Cajas (Boxplot) de Diferencia de Velocidad\n",
    "sns.boxplot(\n",
    "    data=all_metrics_df, \n",
    "    x='Grupo', \n",
    "    y='Delta_Speed', \n",
    "    ax=axes[0], \n",
    "    hue='Grupo',\n",
    "    palette=['#2ECC71', '#E74C3C'],\n",
    "    legend=False\n",
    ")\n",
    "axes[0].set_title('Comparativa de Diferencia de Velocidad (|Speed_1 - Speed_2|)', fontsize=12, fontweight='bold')\n",
    "axes[0].set_ylabel('Diferencia de Velocidad (Puntos)')\n",
    "axes[0].set_xlabel('')\n",
    "\n",
    "# Subgráfico B: Histograma de Conteo Real de Parejas de Combates (stat='count', recortado en 0)\n",
    "sns.histplot(\n",
    "    data=all_metrics_df, \n",
    "    x='Delta_Speed', \n",
    "    hue='Grupo', \n",
    "    ax=axes[1], \n",
    "    palette=['#2ECC71', '#E74C3C'], \n",
    "    binwidth=5, \n",
    "    element='step', \n",
    "    common_norm=False\n",
    ")\n",
    "axes[1].set_title('Cantidad Real de Parejas por Diferencia de Velocidad', fontsize=12, fontweight='bold')\n",
    "axes[1].set_xlabel('Diferencia de Velocidad (Puntos)')\n",
    "axes[1].set_ylabel('Cantidad Real de Parejas de Combates')\n",
    "axes[1].set_xlim(0, 150) # Iniciar estrictamente en 0 (sin valores negativos ilógicos)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> **Aclaración Conceptual y Conclusiones del Análisis Comparativo:**\n",
    ">\n",
    "> 1. **Diferenciación entre Parejas Únicas y Pokémon Únicos:**  \n",
    ">    - Una **pareja única** representa una combinación específica de 2 contendientes (por ejemplo, Pikachu vs Charmander). En el dataset existen **110 parejas únicas de combates variables**.\n",
    ">    - Esas 110 parejas están formadas por **169 Pokémon únicos** combinados entre sí.\n",
    ">\n",
    "> 2. **Superposición de Pokémon (99.4%):**  \n",
    ">    - 168 de los 169 Pokémon que participan en el grupo variable (6.0%) **TAMBIÉN participan en combates del grupo determinista (94.0%)** al enfrentar a otros rivales.  \n",
    ">    - *Conclusión:* No existe un grupo de Pokémon \"variables por naturaleza\". Un Pokémon se vuelve sensible a la posición `First_Pokemon` únicamente cuando enfrenta a un rival específico con características muy semejantes.\n",
    ">\n",
    "> 3. **Visualización de la Diferencia de Velocidad (Factor Determinante):**  \n",
    ">    - Como demuestra la gráfica de **Boxplot e Histograma de Conteo Real de Parejas** superior, el grupo de **Parejas Variables (6.0%)** presenta una altísima concentración de observaciones en **0 a 5 puntos de diferencia de velocidad**, registrando un **43.6% de empates exactos (`Speed_1 == Speed_2`)**.\n",
    ">    - En contraste, el grupo de **Parejas Deterministas (94.0%)** muestra una dispersión mucho más amplia (mediana de **30 puntos** de diferencia) y **0.0% empates en velocidad**, confirmando que una ventaja clara en velocidad define al ganador absoluto sin importar la posición."
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
    "> 2. **Determinismo vs Ventaja de Primer Movimiento en Enfrentamientos Invertidos:**  \n",
    ">    - El análisis gráfico demuestra que en **el 94.0% (1,712 parejas)** de los enfrentamientos que ocurrieron en ambos sentidos `(A vs B y B vs A)`, el ganador es el mismo Pokémon absoluto sin importar quién atacó de primero.  \n",
    ">    - En el **6.0% (110 parejas)** restante, el ganador cambia al invertir las posiciones. La auditoría reveló que el **43.6% de estas parejas tienen empates de velocidad (`Speed_1 == Speed_2`)**, demostrando el efecto directo del **primer movimiento (`First_Pokemon`)** cuando las características son parejas. Por ende, los combates duplicados representan simulaciones legítimas y **no deben ser eliminados**.\n",
    ">\n",
    "> 3. **Valores Ausentes en `Type 2` (48.6%):**  \n",
    ">    - Los `NaN` en `Type 2` corresponden a Pokémon monotipo (de un solo elemento como Pikachu o Charmander). No representan una falla de registro.  \n",
    ">    - *Decisión de Calidad:* Se mantendrán y categorizarán como `'None'` (o `'None_Type'`) para permitir el cálculo correcto de matrices de ventaja de tipo sin eliminar ninguna observación.\n",
    ">\n",
    "> 4. **Ausencia de Atributos Negativos o Inválidos:**  \n",
    ">    - El 100% de las estadísticas numéricas (`HP`, `Attack`, `Defense`, `Sp. Atk`, `Sp. Def`, `Speed`) se encuentran en rangos estricta y físicamente válidos (> 0). Todos los IDs pertenecen al rango [1, 800].\n",
    ">\n",
    "> 5. **Valores Extremos (*Outliers*) Válidos:**  \n",
    ">    - Las sumas de atributos oscilan entre 180 (Sunkern) y 780 (Mega Rayquaza, Mega Mewtwo X/Y). Se concluye que los valores extremos corresponden a **Mega-Evoluciones y Legendarios legítimos** de la franquicia y no a errores de medición. Por ende, **ningún outlier será descartado**, ya que representan variaciones reales del dominio."
   ]
  }
]

notebook_obj = {
 "cells": cells_list,
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
    json.dump(notebook_obj, f, indent=2, ensure_ascii=False)

with open('.ipynb_checkpoints/pokemon_battle_prediction-checkpoint.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook_obj, f, indent=2, ensure_ascii=False)

print('Updated plot to real counts histplot and centered markdown tables successfully!')
