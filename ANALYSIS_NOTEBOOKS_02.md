# Análisis comparativo — Notebooks 02 de Pokémon

**Fecha:** 28 de septiembre de 2026
**Rama:** `main` (`4d952d6`, merge de PR #1)
**Objeto:** comparar las dos versiones del notebook de *feature engineering / data preparation* para la Asignación 2 y decidir cuál consolidar.

---

## 1. Las dos versiones

| Archivo | Celdas | Ejecutado | Errores | Idioma |
|---|---|---|---|---|
| `notebooks/02_feature_engineering_pokemon.ipynb` | 30 (17 md / 13 código) | sí, 100 % | 0 | Inglés |
| `notebooks/02_data_preparation(1).ipynb` | 48 (27 md / 21 código) | sí, 100 % | 0 | Español |
| `notebooks/.ipynb_checkpoints/02_feature_engineering_pokemon-checkpoint.ipynb` | 30 | sí, 100 % | 0 | Inglés |

**El checkpoint es un duplicado.** Sus 30 fuentes son **idénticas** a las de `02_feature_eng` (diff unificado vacío); solo difieren los metadatos de ejecución. Es un archivo de sobrescritura, no una tercera versión. No cuenta como candidato.

Ambos ejecutan de punta a punta con `jupyter nbconvert --execute` **sin un solo error**.

### 1.1 Entorno verificado

```
Python 3.12.3 | scikit-learn 1.9.0 | pandas 3.0.5 | numpy 2.5.1
```

El entorno coincide con el que registran los outputs de los notebooks, así que **las diferencias de accuracy que se reportan abajo no provienen de un desajuste de versiones**.

---

## 2. Veredicto

**`02_feature_engineering_pokemon.ipynb` es la mejor base.**

No es una preferencia de estilo: la otra versión **no puede cumplir el objetivo de la asignación**. El detalle está en §4.

---

## 3. Qué hace bien cada una

### 3.1 `02_feature_eng` — arquitectura

Su decisión de diseño central es que **el merge ocurre dentro del transformer**, no antes:

```
celdas del pipeline:
  create_pokemon_features(battles)   # recibe SOLO (First_pokemon, Second_pokemon)
      -> merge con el catálogo (closure)
      -> calcula las 5 master features
  -> ColumnTransformer(StandardScaler, remainder="drop")
```

Como consecuencia, el pipeline consume **IDs crudos** y es idéntico para entrenar y para predecir. Verificado: `tests.csv` (10 000 × 2) → `(10000, 17)`.

Aporta además:

- **`build_model_pipeline(estimator, feature_set)`** con `clone(preprocessor)` — el elemento que la plantilla Titanic coloca como eje de la sección 12 y que aquí falta por completo en la otra versión. Garantiza que cada experimento arranque sin estado aprendido compartido.
- **Manifiesto de split + metadata JSON** en `artifacts/` (reproducibilidad del particionado entre notebooks).
- **Justificación de los diferenciales por antisimetría**, no por intuición: medir que al intercambiar combatientes la etiqueta se invierte (93,87 % de los solapes) justifica usar $X_1 - X_2$ en lugar de pares de valores crudos.
- **Documenta un resultado negativo**: los ratios empeoran −0,53 pp, la señal es aditiva. Esto evita que el notebook de modelado repita el error.
- **Registra la corrección del bug de tipos** en la tabla de decisiones (§5), no en silencio.

### 3.2 `02_data_prep` — rigor de validación

Su fortaleza es la disciplina de datos, y es real:

- **Valida `TYPE_CHART` contra la tabla oficial de la Generación VI** (celda 16). Transcribe la referencia aparte y compara entrada por entrada: `Faltantes: [] | Sobrantes: [] | Valor distinto: []`. **Esto atiende directamente la corrección que pidió el profesor en la Asignación 1.** La otra versión solo anota la cobertura y no valida nada.
- **One-hot de tipos con `min_frequency=0.01` + `handle_unknown="infrequent_if_exist"`** —agrupa categorías raras y tolera tipos nuevos. Justifica el umbral con las frecuencias observadas en train. Verificado: solo 4 especies de 800 tienen `Type 1 = Flying` (0,48 % de los combates), lo que confirma que la categoría es inestable y debe agruparse.
- **48 comprobaciones con bitácora acumulativa** (`validation_log`), cada una con mensaje explícito. Se puede auditar qué se verificó.
- **Deduplica** (1 952 exactos → 48 048 filas) antes del split.
- **Tabla rol-por-columna** con un `check` que falla si aparece una columna sin rol asignado.
- **Detecta redundancia lineal exacta**: demuestra que `Speed_diff` y `Stat_Total_Diff` son combinaciones lineales exactas de los stats crudos, y advierte que en el brazo C eso produce **multicolinealidad perfecta** para modelos lineales. Es la observación que gobierna la comparación A/B/C.
- **Justifica la exclusión de `Generation` con evidencia medida**, no por descarte. Confirmado: `corr(Generation_diff, target) = +0.0147`.

---

## 4. El error que invalida a `02_data_prep`

**Su pipeline no puede procesar `tests.csv`.**

Motivo: su `FunctionTransformer` envuelve `add_master_features`, que espera una tabla **ya mergeada** (`Speed_first`, `Type 1_first`, …). El merge se ejecuta en la celda 18, **en el cuerpo del notebook, fuera del pipeline**. Replicando su pipeline exacto sobre los datos de test:

```
tests.csv columnas de entrada: ['First_pokemon', 'Second_pokemon']
FALLA:  KeyError: 'Speed_first'
```

Además, la versión en español **excluye `tests.csv` explícitamente**. Sus dos únicas menciones lo tratan como descartado:

- Celda 47: ``y `tests.csv` (sin etiqueta)`` → listado de exclusiones.
- Celda 1: *"A partir de este notebook no tomamos ninguna decisión con el test"*.

Es decir, no es un descuido puntual: es una decisión de diseño. Y es incompatible con el objetivo. El entregable de la Asignación 2 es **predecir `tests.csv`**, y su pipeline no puede generar esa entrega sin un paso de merge manual no documentado entre el notebook 02 y el 03.

Ningún otro problema de `02_data_prep` se acerca en gravedad a este. Todo lo demás son defectos subsanables en un cuarto de hora.

---

## 5. Elementos a portar de `02_data_prep` → `02_feature_eng`

| # | Elemento | Por qué |
|---|---|---|
| 1 | **Validación de `TYPE_CHART` contra Gen-6** (celda 16) | Es la corrección pedida por el profesor. `02_feature_eng` solo anota "120 de 324" y no verifica nada |
| 2 | **Bloque one-hot de tipos** con `min_frequency` + `infrequent_if_exist` | Falta por completo. Aporta `Is_Monotype` implícito y grupa categorías raras |
| 3 | **Justificación medida de la exclusión de `Generation`** | En `02_feature_eng` la palabra aparece **1 sola vez**, en la lista de `remainder="drop"`. Sin evidencia. Medido: `corr = +0.0147` → la exclusión es correcta, pero debe justificarse |
| 4 | **Análisis de redundancia lineal exacta** | El notebook de modelado necesita saber que el brazo C induce colinealidad perfecta en modelos lineales |
| 5 | **Tabla rol-por-columna** con `check` de cobertura | Hace auditable la selección de features, que es el objetivo de la sección 8 de la plantilla Titanic |
| 6 | **Bitácora de validaciones** | 48 checks con mensaje explícito > asserts sueltos |

**Y en sentido contrario**, de `02_feature_eng` → `02_data_prep`: el merge dentro del pipeline, que es lo que elimina el `KeyError`.

---

## 6. Cifras declaradas: no reproducen exactamente

`02_feature_eng` publica en §14 (tabla *"EDA-validated cross-validation scores"*):

| Brazo | Declarado | Reproducido |
|---|---|---|
| A (raw, 12) | 94,49 % | **94,39 %** |
| B (engineered, 5) | 94,34 % | **94,59 %** |
| C (combined, 17) | 95,30 % | **95,63 %** |

Reproducción con el protocolo declarado del propio notebook (`GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)` → `StratifiedGroupKFold(5, shuffle=True, random_state=42)`, `RandomForestClassifier(n_estimators=100)`, merge dentro del pipeline, sin deduplicar, desviación típica):

**Ordenamiento preservado** (C > B > A) y **C sigue ganando con claridad**, así que la conclusión del notebook es correcta. Pero los números publicados **hay que regenerarlos** antes de entregarlos, y la dispersión entre A y B (±0,2 pp) está dentro del ruido de fold, de modo que la comparación A-vs-B no debe presentarse como una diferencia real.

Como los entornos coinciden (§1.1), la discrepancia viene del estimador o la semilla exactos usados en la corrida original, que el notebook no documenta. **Recomendación: regenerar la tabla in-notebook**, con el código a la vista.

---

## 7. Matiz sobre la cobertura de `TYPE_CHART`

Ambos notebooks reportan la tabla como "120 de 324 entradas (37 %)". El número es correcto, pero la **implicación es engañosa**: `TYPE_CHART` no almacena las 324 porque solo se guardan los cruces **no neutros**. Las 204 entradas restantes son exactamente los neutros ×1.0, que se resuelven por defecto en `chart.get(pair, 1.0)`.

No es una tabla incompleta, es una tabla *sparsely populated* por diseño. `02_feature_eng` incluso lo admite en la metadata, pero luego escribe `"TYPE_CHART has 120 of 324 possible matchups (37%). Unspecified matchups default to neutral (1.0)."`, que se lee como deficiencia.

**Recomendación:** reformular como *"la tabla declara los 120 cruces no neutros de la Gen-6; todo par no declarado es neutro por definición"*, en ambos.

---

## 8. Higiene de repositorio

1. **Tres archivos para lo mismo.** Tras resolver el contenido, conservar **una sola versión en inglés** y eliminar el checkpoint (fuentes idénticas) y el sufijo `(1)` de descarga de Windows.
2. **`scripts/features.py` está modificado sin commitear.** Contiene el fix de `calculate_type_multiplier` (`git diff` limpio y correcto), del que **depende** la corrección del multiplicador de tipos. Debe commitearse: si no, la entrega no es reproducible desde `main`.
3. **`PLAN.md` figura como borrado sin commitear** (ya restaurado por el merge de la PR #1) — decidir si se conserva.
4. **Archivos sin trackear**: `PLAN_ASSIGNMENT_2.md`, `artifacts/`, y los dos notebooks 02.

---

## 9. Plan de consolidación

1. Partir de **`02_feature_eng`** (merge end-to-end + `build_model_pipeline` + manifiesto + inglés).
2. Portar los 6 elementos de §5.
3. Regenerar la tabla de CV in-notebook (§6) y reportar A-vs-B como empate dentro del ruido.
4. Reformular la nota de cobertura de tipos en ambos (§7).
5. Commitear el fix de `features.py` **antes** que cualquier cosa que dependa de él.
6. Normalizar a un único archivo; borrar checkpoint y sufijo `(1)`.
7. Reejecutar completo y confirmar 0 errores.

**Resultado esperado:** ~55 celdas, un único pipeline que va de IDs crudos a predicción, validado contra la Gen-6, con la entrega de `tests.csv` reproducible.
