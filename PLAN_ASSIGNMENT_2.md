# Plan — Assignment 2: Feature Engineering & Pipeline (notebook `02_feature_engineering_pokemon.ipynb`)

**Base:** `downloads/02_data_preparation.ipynb` (Titanic, teaching template) + `notebooks/01_eda_pokemon.ipynb` (EDA actual, ya mergeado en `main` vía PR #1)
**Estado:** análisis exploratory completo. Todas las cifras de abajo están medidas sobre `data/pokemon_combats_unified.csv` (50 000 × 29).

---

## 1. Qué hace el notebook 2 del Titanic y qué se transfiere

Es una plantilla de 13 secciones con una arquitectura muy deliberada. El valor no está en Titanic, está en el **esqueleto**:

| # | Sección Titanic | Principio transferible |
|---|---|---|
| 2 | Load and validate | Validación explícita → `raise`, no warnings silenciosos |
| 3 | Split target + partition | **Split antes de fitear** nada; `stratify=y` |
| 4 | Tabla EDA→decisiones | Cada hallazgo del EDA se traduce a una decisión **con su alternativa a comparar** |
| 5 | `create_titanic_features()` | Función **stateless**: solo combina valores de la misma fila, no aprende nada |
| 7–9 | `ColumnTransformer` + sub-pipelines | Agrupar por tipo de transformación; `remainder="drop"` hace la selección **auditable** |
| 10 | `FunctionTransformer` | El FE vive **dentro** del pipeline, no horneado en el CSV |
| 11 | Asserts estructurales | nº de filas no cambia, todo finito |
| 12 | `build_model_pipeline(est)` + `clone()` | Fábrica de pipelines **sin estado compartido** entre experimentos |
| 13 | Manifiesto + metadata JSON | Se guarda el **split**, no la matriz transformada |

**Los 4 que más importan aquí:** stateless FE, FE dentro del pipeline, `build_model_pipeline` con `clone`, y split-before-fit.

---

## 2. Dónde Pokémon diverge del Titanic (5 adaptaciones obligatorias)

### 2.1 La fila cruda de Pokémon no es una fila de Titanic ← **la más importante**

Titanic: una fila = un pasajero. `create_titanic_features(X_train)` ya es stateless.

Pokémon: una fila = **una batalla entre 2 filas de `pokemon.csv`**. El FE sobre la tabla ya mergeada es trivialmente stateless (restar columnas), pero eso **no sirve**: `data/tests.csv` nunca fue mergeado y no tiene stats.

**Diseño correcto:** el pipeline debe tomar `(First_pokemon, Second_pokemon)` crudos y **mergear internamente** desde `pokemon.csv` + imputar el nombre de #63 + extraer FE. Así el **mismo objeto** se puede pasar con `combats.csv` para entrenar y con `tests.csv` para predecir, sin ningún paso intermedio manual. Esto es la razón real por la que el patrón Titanic vale la pena aquí.

→ `scripts/merge.py` se **reutiliza como transformer**, no se reimplementa. `merge.py` ya valida schema/unicidad/integridad referencial (`scripts/merge.py:1-209`); ese código se encaja dentro de un `FunctionTransformer`.

### 2.2 La imputación numérica desaparece — y `Type 2` NO se imputa

No hay NaN en HP/Attack/Defense/Sp.Atk/Sp.Def/Speed. El `SimpleImputer` numérico del Titanic **no aplica**.

El análogo correcto del "Age + AgeMissing" de Titanic es `Type 2`, y **no** es una imputación: los NaN (48%) significan *monotype*, que es información. Imputar con `"None"`/moda destruiría señal. → bandera explícita `Is_Monotype_first/second`, nunca imputación.

### 2.3 Titanic no tiene problema de dependencia entre filas; Pokémon sí ← **crítico**

`PassengerId` es único → `train_test_split` estratificado es suficiente. En Pokémon las batallas **no son independientes**:

- Split estratificado ingenuo → **11,80 %** de las filas de test ya tienen su par no ordenado en train.
- Tras eliminar las 1 952 duplicadas exactas → **sigue habiendo 6,16 %**.

`GroupShuffleSplit` / `GroupKFold` por **par no ordenado** `(min(id), max(id))`. Es laScenario C del EDA (§11.6)BIOelevada a **protocolo por defecto**, no a escenario opcional.

Mitigación real: el modelo **no usa IDs**, así que no puede memorizar pares. Verificado: CV agrupado 95,19 % vs split ingenuo 95,53 % → la inflación del leakage es ~0,3 pp. El resultado es robusto, pero el número que se reporte debe salir del split agrupado.

### 2.4 Antisimetría: la etiqueta lo es, las features deben serlo

- Reversor de orden → el ganador **cambia** en el 93,87 % de los solapes.
- El **99,76 %** de los matchups no ordenados tiene **un solo** ganador.

La etiqueta es antisimétrica bajo el intercambio. Por tanto las features deben serlo por construcción (diferencias, no pares). Esto es el argumento **teórico** de por qué el brazo C (Raw+Engineered) supera al A, y ya se confirma empíricamente (C 95,27 % vs A 94,31 %).

### 2.5 El encoding categórico sí tiene lugar aquí

El Titanic codifica `Sex`/`Embarked`/`Pclass`/`Title`. Pokémon tiene categóricos legítimos hoy **descartados por completo**: `Generation`, la taxonomía `classify_form`, el matchup de formas, `Type 1`. Se transfiere `OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)`.

**Pero medido: aporta casi nada** (§4.3). Se incluye por completitud y se reporta como resultado negativo honesto.

---

## 3. Corrección importante al EDA (sección 11.6)

El EDA §11.6 concluye: *"...extraer señal de ventajas elementales, ratios de penetración y formas para resolver la **zona de paridad/conflicto** de ~6 %"*.

**Eso está invertido.** El error de la regla de velocidad **crece** con el hueco de velocidad:

| `|Speed_diff|` | n | error de la regla |
|---|---|---|
| 0 (empate) | 1 328 | **0,00 %** |
| (0, 1] | 504 | 2,58 % |
| (10, 20] | 8 722 | 4,14 % |
| (20, 30] | 7 709 | 5,82 % |
| (30, 50] | 11 865 | 7,37 % |
| (50, 200] | 10 975 | **9,39 %** |

La zona de paridad (empates) está **perfectamente resuelta**. El residuo del 5,95 % es el **regimen del underdog**: un Pokémon mucho más lento pero mucho más fuerte ganando un combate de hueco amplio. Eso cambia el措辞 de las conclusiones y la hipótesis que se testean.

### 3.1 La regla de empate es posicional, no de initiative

- En los 1 328 empates, gana `Second_pokemon` el **100 %**, en las 6 generaciones.
- De los 103 empates cuyo par invertido también aparece, la **posición 2ª** gana en ambos casos → el ganador es un Pokémon distinto según el orden.

Es un artefacto de desempate del dataset, no "quien tiene la iniciativa". Es 100 % aprendible (2,66 pp de accuracy gratis), y un árbol lo captura solo desde `Speed_diff` (verificado: `Is_Speed_Tie` explícito añade **+0,0003**, redundante). → **no añadir la feature**, pero sí documentar la regla y su naturaleza artificial.

---

## 4. Evidencia empírica: qué representation funciona

CV agrupado por par no ordenado, `GroupKFold(5)`, sobre 50 000 filas (`n` de test aparte en split agrupado 80/20):

| Brazo | Features | groupedCV | test (pair-disjoint) |
|---|---|---|---|
| Baseline velocidad | regla determinista | 0,9405 | 0,9403 |
| **A** raw | 12 | 0,9431 | 0,9444 |
| **B** engineered | 5 | 0,9441 | 0,9426 |
| **C** combined | 17 | **0,9527** | **0,9520** |

**El objetivo del EDA (superar 94,05 %) es alcanzable y la ganancia es real: ~+1,2 pp, no un artefacto de leakage.** La增益 se sostiene en los dos regímenes de split.

### 4.1 Rankings y ratios: resultado **negativo** (no añadir)

| Cambio | groupedCV | Δ |
|---|---|---|
| C actual (17) | 0,9527 | — |
| + `Is_Speed_Tie` | 0,9530 | +0,0003 (redundante) |
| `Speed_Rank_Ratio`, `Stat_Total_Ratio` | 0,9474 | **−0,0053** |

Los ratios logarítmicos **empeoran**. La señal vive en diferencias **aditivas**, no en razones. Mantener `Speed_diff`/`Stat_Total_Diff`; no añadir ratios.

### 4.2 `Is_Monotype` + `Generation_diff`

Entran como features numéricas válidas (protegen contra imputación ingenua de `Type 2` y aprovechan datos hoy desperdiciados). Se prueban en el pipeline final; se mantienen solo si no degradan.

### 4.3 Categóricos: aporte marginal

RF + 5 categóricos (form×2, `Form_Matchup`, `Type 1`×2) + `Generation_diff` + monotype → **0,9530** (+0,0003). LogReg → **0,8832**, muy por debajo.

Consecuencia: la relación es **fundamentalmente no lineal** y está dominada por la regla de velocidad. **Logistic Regression debe reportarse como baseline lineal débil, no como candidato principal** — el EDA §11.4 lo listaba sin advertencia.

---

## 5. Bug en `scripts/features.py` (inerte, pero corregir)

`calculate_type_multiplier` (`scripts/features.py:157`):

```python
m2 = val(atk_t2, def_t1) * val(atk_t2, def_t2) if (isinstance(atk_t2, str) and not pd.isna(atk_t2)) else 1.0
```

El `else 1.0` debería ser `None` (o excluir `m2`): fuerza `max(m1, m2) >= 1.0` para todo atacante monotype.

- Ejemplo: Venusaur (Grass/None) contra Charizard (Fire/Dragon) devuelve **1,0**; correcto es **0,5**.
- 12 % de las filas discrepan de una implementación correcta; el multiplicador queda inflado a ≥1 en el 96 % de las filas.
- **Es inerte hoy**: tras el `log2((m+0.1)/(m'+0.1))` de `Type_Advantage_Ratio` el signo nunca cambia (0 filas). Pero es incorrecto y explotaría si el multiplicador se usara crudo.
- Aparte: `TYPE_CHART` tiene 120 de 324 entradas (37 %); el resto sale neutral 1,0. Ya documentado como aproximación en el EDA §7, pero conviene que quede explícito en el docstring.

**Acción:** arreglar el `else 1.0` → `None` + test unitario, **antes** de construir el pipeline (si no, el FE del notebook y el de `merge.py` divergen).

---

## 6. Estructura propuesta del notebook `notebooks/02_feature_engineering_pokemon.ipynb`

Espejo de las 13 secciones del Titanic, adaptadas:

| # | Sección | Contenido |
|---|---|---|
| 1 | Imports y rutas | igual que Titanic; `PROJECT_ROOT` + `sys.path` a `scripts/` |
| 2 | **Load + validate crudos** | valida `combats.csv`, `pokemon.csv`, `tests.csv`; imputa nombre de #63 |
| 3 | **Target + split agrupado** | `Target_First_Wins`; `GroupShuffleSplit` por par no ordenado; **`stratify` no aplica → verificar balanceo**; **descartar `Winner` aquí mismo** |
| 4 | **Tabla EDA→decisiones** | la del Titanic, con la corrección de §3 (zona de underdog, no de paridad) |
| 5 | **FE stateless** | `create_pokemon_features(raw_battles)`: merge + FE. **Stateless** (el chart de tipos y `classify_form` son constantes) |
| 6 | Preview | 5–10 filas para verificar meanings |
| 7–9 | `ColumnTransformer` | numérico (StandardScaler), categórico (OHE + `min_frequency`), binario (passthrough); `remainder="drop"` |
| 10 | `FunctionTransformer` | envuelve el FE; el **merge ocurre dentro** del pipeline |
| 11 | **Asserts + leakage asserts** | filas constantes, finitos, **y `Winner ∉ features`**, **y par no ordenado ∉ train∩test** |
| 12 | `build_model_pipeline(est)` + `clone` | y el **splitter** por par, para que CV y test usen la misma garantía |
| 13 | Manifiesto + metadata | `pair → split`, `random_state`, versiones, **y la nota de que el test no es *untouched*** (mismo disclaimer que Titanic) |
| 14 | Handoff | métricas fijadas **antes** de mirar resultados;Benchmark vs 52,80 % y 94,05 % |

**Punto clave de la sección 3:** Titanic usa `train_test_split(stratify=y)`. Aquí eso **no es válido** — agrupar por par y estratificar son objetivos en conflicto. Se usa `GroupShuffleSplit` y se **verifica explícitamente** el balanceo de clases resultante (el EDA ya mide 47,20 / 52,80 %).

---

## 7. Orden de ejecución

1. Arreglar el bug de `calculate_type_multiplier` + test.
2. Notebook `02` secciones 1–5 (validación, split agrupado, FE stateless) — **el FE debe entrar por `tests.csv` y producir las mismas columnas** que por `combats.csv`. Smoke test decisivo.
3. Secciones 6–13 (ColumnTransformer, pipeline, asserts, manifiesto).
4. Solo después: notebook `03` de modelado con CV agrupada y los 3 brazos.
5. Corregir §11.6 del `01_eda` para el hallazgo de §3 (el enunciado de la "zona de paridad" está factualmente mal).

---

## 8. Riesgos

- **`GroupShuffleSplit` no estratifica** → verificar el desbalance; si degrada, agrupar por par + `StratifiedGroupKFold` (disponible en sklearn ≥1.1).
- **Coste de cómputo**: el merge dentro del pipeline re-mergea en cada fold. Con 50 000 filas y 5 folds es aceptable, pero `n_jobs` debe quedar acotado dentro de CV.
- **El FE no puede aprender nada del fold** (stateless). Si se añade alguna feature *aprendida* (frecuencias, target encoding por especie), debe ir en un transformer con `fit`/`transform` propio — y el EDA §11.5 lo prohíbe para IDs, así que no se hará.
