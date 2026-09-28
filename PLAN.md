# Plan de Corrección y Estado Actual — PR #1 (Assignment 1)

**Proyecto:** Pokémon Battle Winner Predictor  
**Rama:** `feat/eda`  
**Fecha:** 13 de Septiembre de 2026  

---

## 📊 Estado Actual del Proyecto (Actualizado post-implementación)

| Área | Estado | Diagnóstico |
|---|---|---|
| **`scripts/merge.py`** | 🟢 **Corregido y Validado** | Imputa la especie `#63` (`Primeape`) inmediatamente, aplica validaciones estrictas de schema, unicidad, integridad referencial y target domain, calcula features diferenciales importando `features.py` y genera `pokemon_combats_unified.csv` (50,000 filas × 29 columnas) sin errores tanto desde `data/` como desde la raíz. |
| **`scripts/features.py`** | 🟢 **Completo** | Módulo completamente implementado con tipado, `TYPE_CHART` de 120 matchups, cálculo de diferenciales (`Speed_diff`, `Attack_diff`, `Defense_diff`), `Stat_Total_Diff`, `Atk_Def_Penetration_Diff`, taxonomía de `classify_form`, `Special_Form_Advantage`, `calculate_type_multiplier`, `Type_Advantage_Ratio` y funciones agregadoras `add_master_features` / `add_engineered_features`. |
| **`notebooks/01_eda_pokemon.ipynb`** | 🟢 **Completamente Actualizado y 100% Pre-ejecutado** | Se reemplazó *"predictive ceiling"* por *"Strong Speed-Based Baseline"* en títulos, gráficos, celdas y conclusiones. Se moderó el lenguaje hacia asociación lineal univariada ($r = +0.678$). Se encuadró como hipótesis con posible confusión la ventaja de formas especiales vs base stats. Se documentaron los 3 escenarios de batalla no vista (A, B, C), la estrategia 3-arm (Raw vs Engineered vs Combined) y el protocolo de deduplicación de 1,952 combates en prevención de Data Leakage. Ejecutado exitosamente de punta a punta vía `nbconvert`. |
| **`README.md`** | 🟢 **Sincronizado al 100%** | Alineado completamente con los hallazgos del notebook: regla de desempate en empates de velocidad (Segunda posición gana el 100% de los 1,328 casos), baseline fuerte de velocidad (94.05%), protocolo de manejo de los 1,952 duplicados, definición de escenarios de evaluación no vistos (A, B, C) y estrategia de modelado 3-arm en el roadmap de Assignment 2. |

---

## 🎯 Plan de Acción y Lista de Cambios

### Fase 1: Corrección y Robustecimiento de `scripts/merge.py`
- [x] Imputar el nombre ausente para el Pokémon `#63` (`Primeape`) inmediatamente después de cargar `pokemon.csv` y antes de las aserciones schema.
- [x] Incorporar validación de tipos de datos y verificación de duplicados exactos en `combats.csv`.
- [x] Importar e integrar la generación de las características diferenciales desde `scripts/features.py`.
- [x] Validar que la ejecución de `python ../scripts/merge.py` desde `data/` y `python scripts/merge.py` desde la raíz complete exitosamente generando `pokemon_combats_unified.csv`.

### Fase 2: Implementación Completa de `scripts/features.py`
- [x] Implementar la función `calculate_speed_diff(df)` ($\text{Speed}_1 - \text{Speed}_2$).
- [x] Implementar `calculate_stat_total_diff(df)` ($\text{Total}_1 - \text{Total}_2$).
- [x] Implementar `calculate_atk_def_penetration_diff(df)` ($\frac{\text{Attack}_1}{\text{Defense}_2} - \frac{\text{Attack}_2}{\text{Defense}_1}$).
- [x] Implementar `calculate_special_form_advantage(df)` ($\text{Is\_Special}_1 - \text{Is\_Special}_2$).
- [x] Implementar `calculate_type_advantage_ratio(df)` ($\log_2\left(\frac{\text{Eff}_{1 \rightarrow 2} + 0.1}{\text{Eff}_{2 \rightarrow 1} + 0.1}\right)$) incluyendo la matriz completa de tipos.
- [x] Crear la función integradora `add_master_features(df)` que agrega las 5 columnas al DataFrame.

### Fase 3: Depuración y Refinamiento del Notebook (`notebooks/01_eda_pokemon.ipynb`)
- [x] Reemplazar todas las apariciones persistentes del término *"predictive ceiling"* por *"Strong Speed-Based Baseline"* (incluyendo el título del gráfico en la Celda 44 y el texto en las Celdas 45, 49 y 50).
- [x] Moderar afirmaciones categóricas sobre la dominancia de `Speed` para reflejar correctamente la *"asociación lineal univariada más fuerte"* ($r = +0.678$).
- [x] Encuadrar explícitamente como hipótesis la relación entre la ventaja de formas especiales y sus estadísticas base en las celdas de análisis.
- [x] Documentar explícitamente la tabla de efectividad de tipos como una matriz simplificada/aproximada en la celda 25/26.
- [x] Definir explícitamente los 3 escenarios de *"batalla no vista"* (Escenario A: nuevo enfrentamiento entre Pokémon conocidos; Escenario B: generalización a Pokémon nunca observados; Escenario C: nueva ejecución de un matchup previo) en la Sección 11.6.
- [x] Incluir la estrategia de eliminación de duplicados exactos antes del split en las notas de prevención de Data Leakage (Sección 11.5 y 11.6).

### Fase 4: Sincronización Completa del `README.md`
- [x] Añadir la definición de los escenarios de *"batalla no vista"* (Escenarios A, B, C) en la Sección de Roadmap de Modelado.
- [x] Actualizar el diagrama y el texto del roadmap para incluir la estrategia de evaluación 3-arm:
  - **Modelo A:** Raw Base Stats (12 atributos).
  - **Modelo B:** Master Engineered Features (5 atributos).
  - **Modelo C:** Raw + Engineered Features (17 atributos).
- [x] Documentar el protocolo de prevención de Data Leakage relativo a los 1,952 duplicados (eliminación previa al train/test split).

### Fase 5: Verificación de Calidad y Reproducibilidad
- [x] Ejecutar `merge.py` y verificar que genera `pokemon_combats_unified.csv` sin errores.
- [x] Ejecutar todas las celdas del notebook para asegurar reproducibilidad 100% y visualizaciones alineadas.
- [x] Validar consistencia absoluta entre `README.md`, `merge.py`, `features.py` y `notebooks/01_eda_pokemon.ipynb`.
