 # Predicción del MVP de la NBA - Módulo de Machine Learning

 ## Descripción del proyecto

 Este módulo implementa modelos de machine learning para predecir el porcentaje de votos (share) del MVP (Most Valuable Player) de la NBA a partir de estadísticas de jugadores y rendimiento de equipo. El objetivo es identificar con precisión los principales candidatos al MVP en cada temporada.

 ---

 ## Índice

 1. [Carga y preparación de datos](#1-carga-y-preparaci%C3%B3n-de-datos)
 2. [Selección de features](#2-selecci%C3%B3n-de-features)
 3. [Entrenamiento inicial del modelo](#3-entrenamiento-inicial-del-modelo)
 4. [Métricas de evaluación](#4-m%C3%A9tricas-de-evaluaci%C3%B3n)
 5. [Framework de backtesting](#5-framework-de-backtesting)
 6. [Ingeniería de features](#6-ingenier%C3%ADa-de-features)
 7. [Comparación de modelos](#7-comparaci%C3%B3n-de-modelos)

 ---

 ## 1. Carga y preparación de datos

 ### Qué hace:
 ```python
 stats = pd.read_csv("../data cleaning/combined_stats_master.csv")
 del stats["Unnamed: 0"]
 ```

 **Propósito:** Carga el conjunto de datos de estadísticas de la NBA ya limpiado y combinado, que contiene métricas de rendimiento de jugadores, estadísticas de equipo y los shares de votación del MVP desde 1991 hasta 2024.

 **Por qué:** Estos datos son la base de todo el análisis. Los datos fueron obtenidos y limpiados en módulos previos.

 **Eliminación de columna:** `Unnamed: 0` se elimina porque es un índice generado al exportar el CSV y no aporta valor predictivo.

 ---

 ## 2. Selección de features

 ### Qué hace:
 ```python
 predictor_features = ['Age','G', 'GS', 'MP', 'FG', 'FGA', 'FG%',
        '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%',
        'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Year',
         'W', 'L', 'W/L%', 'GB', 'PS/G', 'PA/G', 'SRS']
 ```

 **Propósito:** Define las estadísticas que se usarán para predecir el share del MVP.

 **Por qué estas features:**
 - **Estadísticas del jugador:** Rendimiento individual (puntos, asistencias, robos, tapones, porcentajes de tiro)
 - **Rendimiento del equipo:** Victorias/derrotas, porcentaje de victorias, diferencial de puntos
 - **Tiempo de juego:** Partidos jugados, minutos disputados
 - **Métricas avanzadas:** eFG% (effective field goal %), SRS (Simple Rating System)

 **Qué NO estamos incluyendo:** La variable objetivo `Share` (porcentaje de votos del MVP) y columnas relacionadas como `Pts Won`, `Pts Max` — queremos predecirlas, no usarlas como entradas.

 ---

 ## 3. Entrenamiento inicial del modelo

 ### División Train/Test:
 ```python
 train_data = stats[stats["Year"] < 2024]
 test_data = stats[stats["Year"] == 2024]
 ```

 **Propósito:** Separa datos históricos (1991-2023) para entrenar y datos recientes (2024) para evaluar.

 **Por qué:** Simula un escenario real donde entrenamos con datos pasados y predecimos la temporada más reciente. Esta separación temporal evita fugas de información (data leakage).

 ### Modelo Ridge Regression:
 ```python
 ridge_model = Ridge(alpha=0.1)
 ridge_model.fit(train_data[predictor_features], train_data["Share"])
 ```

 **Qué es Ridge Regression:**
 - Un modelo de regresión lineal con regularización L2
 - Evita el sobreajuste penalizando coeficientes grandes
 - El parámetro `alpha=0.1` controla la fuerza de la regularización

 **Por qué Ridge en lugar de regresión lineal simple:**
 - Muchas features están correlacionadas (p. ej. FG y PTS)
 - Ridge trata mejor la multicolinealidad
 - Reduce la varianza en las predicciones

 ### Realizar predicciones:
 ```python
 predictions_2024 = ridge_model.predict(test_data[predictor_features])
 predictions_df = pd.DataFrame(predictions_2024, columns=["Predicted"], index=test_data.index)
 comparison_df = pd.concat([test_data[["Player","Share"]], predictions_df], axis=1)
 ```

 **Propósito:** Generar predicciones del share del MVP para 2024 y crear una tabla de comparación.

 **Por qué:** Permite ver cómo se comparan nuestras predicciones con los resultados reales de la votación.

 ---

 ## 4. Métricas de evaluación

 ### Error cuadrático medio (MSE):
 ```python
 mean_squared_error(comparison_df["Share"], comparison_df["Predicted"])
 ```

 **Qué mide:** La media de los cuadrados de la diferencia entre valores predichos y reales.

 **Limitación:** Como se señala en el código, MSE no es la mejor métrica para este problema porque:
 - La mayoría de jugadores tienen Share = 0 (no recibieron votos para MVP)
 - Nos interesa más identificar correctamente a los CANDIDATOS PRINCIPALES
 - Un pequeño error en el ganador importa mucho más que un error en el puesto 50

 ### Métrica personalizada - Precisión media en Top 7:
 ```python
 def calculate_top7_average_precision(comparison):
     actual_top_7 = comparison.sort_values("Share", ascending=False).head(7)
     predicted_ranking = comparison.sort_values("Predicted", ascending=False)
     
     precision_scores = []
     correct_found = 0 
     players_seen = 1
     
     for index, row in predicted_ranking.iterrows():
         if row["Player"] in actual_top_7["Player"].values:
             correct_found += 1
             precision_scores.append(correct_found / players_seen)
         players_seen += 1
     
     return sum(precision_scores) / len(precision_scores)
 ```

 **Cómo funciona:**
 1. Identifica los 7 jugadores reales con mayor share
 2. Recorre las predicciones ordenadas de mayor a menor
 3. Cada vez que encuentra un jugador correcto dentro del top-7, calcula la precisión en ese punto
 4. Promedia todas las precisiones obtenidas

 **Ejemplo:**
 - Si nuestras 3 primeras predicciones son correctas: precision = (1/1 + 2/2 + 3/3) / 3 = 1.0
 - Si los aciertos aparecen en posiciones 2, 5 y 7: precision = (1/2 + 2/5 + 3/7) / 3 ≈ 0.48

 **Por qué esta métrica:**
 - Se centra en la calidad del ranking para los candidatos principales
 - Premia encontrar jugadores correctos pronto en la lista
 - Penaliza colocar jugadores del top-7 real en posiciones bajas
 - Representa mejor el objetivo real: identificar a los aspirantes al MVP

 ---

 ## 5. Framework de backtesting

 ### Propósito del backtesting:
 ```python
 years_range = list(range(1991, 2025))

 def backtest_model(stats, model, years, features):
     average_precisions = []
     all_predictions = []

     for year in years[5:]:  # Skip first 5 years
         train_data = stats[stats["Year"] < year]
         test_data = stats[stats["Year"] == year]
         
         model.fit(train_data[features], train_data["Share"])
         year_predictions = model.predict(test_data[features])
         
         predictions_df = pd.DataFrame(year_predictions, columns=["Predicted"], index=test_data.index)
         comparison = pd.concat([test_data[["Player","Share"]], predictions_df], axis=1)
         comparison = add_ranking_columns(comparison)
         
         all_predictions.append(comparison)
         average_precisions.append(calculate_top7_average_precision(comparison))
         
     mean_ap = sum(average_precisions) / len(average_precisions)
     return mean_ap, average_precisions, pd.concat(all_predictions)
 ```

 **Qué hace:**
 - Evalúa el modelo en múltiples años (1996-2024)
 - Para cada año, entrena con todos los años anteriores y predice ese año
 - Calcula la precisión promedio (Top7 AP) para cada año
 - Devuelve la precisión promedio general

 **Por qué se omiten los primeros 5 años:**
 - Se necesita suficiente historial para entrenar (al menos 5 años)
 - Garantiza un entrenamiento estable

 **Beneficios:**
 - Evaluación más robusta que usar un único año de prueba
 - Muestra si el modelo funciona de forma consistente entre distintas épocas
 - Revela patrones temporales (¿funciona mejor en épocas recientes?)

 ### Análisis de ranking:
 ```python
 def add_ranking_columns(comparison):
     ranked_comparison = comparison.sort_values("Share", ascending=False)
     ranked_comparison["Rank"] = list(range(1, ranked_comparison.shape[0] + 1))
     
     ranked_comparison = ranked_comparison.sort_values("Predicted", ascending=False)
     ranked_comparison["Predicted Rank"] = list(range(1, ranked_comparison.shape[0] + 1))
     
     ranked_comparison["Difference"] = ranked_comparison["Rank"] - ranked_comparison["Predicted Rank"]
     
     return ranked_comparison
 ```

 **Propósito:** Añade la clasificación real, la clasificación predicha y la diferencia entre ambas.

 **Interpretación de `Difference`:**
 - **Diferencia positiva:** El modelo subestimó al jugador (su ranking real es mejor que el predicho)
 - **Diferencia negativa:** El modelo sobreestimó al jugador (su ranking predicho es mejor que el real)
 - **Cero:** Predicción perfecta para la posición de ese jugador

 **Por qué es útil:** Ayuda a identificar sesgos sistemáticos (¿el modelo favorece/desfavorece cierto tipo de jugadores?)

 ---

 ## 6. Ingeniería de features

 ### Estadísticas normalizadas por año:
 ```python
 normalized_stats = stats.groupby("Year")[['PTS', 'AST', 'STL', 'BLK', '3P']].apply(
     lambda x: x / x.mean(), include_groups=False
 )

 stats[["PTS_R", "AST_R", "STL_R", "BLK_R", "3P_R"]] = normalized_stats[["PTS", "AST", "STL", "BLK", "3P"]]
 ```

 **Qué hace:** Crea features de ratio dividiendo cada estadística entre la media del año correspondiente.

 **Por qué es crítico:**
 - **Ajuste por época:** El juego de la NBA ha cambiado (años 90 vs 2020s)
 - **Cambios de ritmo:** Los equipos anotan más hoy por el mayor ritmo de juego
 - **Revolución del triple:** Los jugadores modernos lanzan muchos más triples
 - **Normalización:** Un jugador con 30 PPG en 1995 era más destacable que uno con 30 PPG en 2024

 **Ejemplo:**
 - 1995: Media liga = 20 PPG, jugador 30 → Ratio = 1.5
 - 2024: Media liga = 25 PPG, jugador 30 → Ratio = 1.2
 - El jugador de 1995 es relativamente más excepcional

 **Impacto:** Estas features normalizadas mejoran la robustez del modelo entre distintas épocas.

 ### Codificación categórica:
 ```python
 stats["Position_Encoded"] = stats["Pos"].astype("category").cat.codes
 stats["Team_Encoded"] = stats["Team"].astype("category").cat.codes
 ```

 **Qué hace:** Convierte categorías de texto (posiciones como "PG", "SF" y nombres de equipos) en códigos numéricos.

 **Por qué:** Los modelos de machine learning requieren entradas numéricas. Esta es una codificación simple donde cada categoría recibe un número.

 **Nota:** Estas features se generan pero no se han añadido aún a `predictor_features`; podrían incorporarse en futuras mejoras.

 ---

 ## 7. Comparación de modelos

 ### Alternativa: Random Forest:
 ```python
 random_forest_model = RandomForestRegressor(n_estimators=50, random_state=1, min_samples_split=5)
 ```

 **Qué es Random Forest:**
 - Un ensamblado (ensemble) de 50 árboles de decisión
 - Cada árbol aprende patrones distintos
 - La predicción final es el promedio de todos los árboles
 - `min_samples_split=5`: mínimo 5 muestras para dividir un nodo

 **Por qué probar Random Forest:**
 - Puede capturar relaciones no lineales (Ridge es lineal)
 - Maneja interacciones entre features automáticamente
 - Menos sensible a outliers
 - Puede modelar patrones complejos en la votación del MVP

 **Comparación:**
 ```python
 # Random Forest en años recientes (1991 + 28 = 2019 en adelante)
 mean_avg_precision_rf, _, _ = backtest_model(stats, random_forest_model, years_range[28:], predictor_features)

 # Ridge en el mismo periodo para una comparación justa
 mean_avg_precision_ridge, _, _ = backtest_model(stats, ridge_model, years_range[28:], predictor_features)
 ```

 **Por qué probar `years_range[28:]` (2019-2024):**
 - Random Forest puede sobreajustar con pocos datos
 - Probar en años recientes con todo el historial como entrenamiento
 - Comparación más justa entre modelos
 - Período más relevante para predicciones actuales

 ---

 ## Ideas clave y estrategia del modelo

 ### El problema del MVP:
 1. **La narrativa importa:** La votación para MVP no es puramente estadística — narrativas, récord de equipo y "story" influyen
 2. **Distribución concentrada:** Solo 5-7 jugadores reciben consideración seria cada año
 3. **Evolución temporal:** Lo que define a un MVP ha cambiado a lo largo de las décadas

 ### Por qué esta aproximación funciona:
 1. **Validación temporal:** El backtesting asegura que el modelo funcione entre distintas épocas
 2. **Métrica adecuada:** La Precisión media en Top-7 se enfoca en lo que importa (encontrar candidatos)
 3. **Ingeniería de features:** Las estadísticas normalizadas corrigen diferencias entre épocas
 4. **Diversidad de modelos:** Se prueban enfoques lineales (Ridge) y no lineales (Random Forest)

 ### Mejoras potenciales:
 1. Añadir features categóricas (posición, equipo) a los predictores
 2. Crear features de interacción (p. ej. PTS × W/L%)
 3. Incluir rendimiento del año anterior (momentum)
 4. Añadir métricas de cobertura mediática si están disponibles
 5. Ensamblar (ensemble) ambos modelos para la predicción final

 ---

 ## Cómo usar este módulo

 1. **Cargar datos:** Asegúrate de que `combined_stats_master.csv` esté en `../data cleaning/`
 2. **Ejecutar en orden:** Ejecuta las celdas del notebook de arriba hacia abajo
 3. **Interpretar resultados:**
    - Mayor `mean_avg_precision` = mejor modelo (valor máximo = 1.0)
    - Revisa la columna `Difference` para ver jugadores sobre/subestimados
    - Observa la importancia de features para entender qué impulsa la votación

 4. **Hacer predicciones:**
    - Entrena con todos los datos disponibles
    - Usar para predecir la próxima temporada
    - Monitoriza estadísticas tempranas de la temporada y actualiza predicciones

 ---

 ## Convenciones de nombres de variables

 **Antes → Después (propósito):**
 - `prediction` → `predictor_features` (lista de features de entrada)
 - `train` → `train_data` (dataset de entrenamiento)
 - `test` → `test_data` (dataset de prueba)
 - `reg` → `ridge_model` (modelo de Ridge)
 - `test_predictions` → `predictions_2024` (predicciones de 2024)
 - `compare` → `comparison_df` (comparación real vs predicha)
 - `aps` → `average_precisions` (lista de AP por año)
 - `mean_ap` → `mean_avg_precision` (precisión media total)
 - `sc` → `scaler` (objeto StandardScaler)
 - `rf` → `random_forest_model` (modelo Random Forest)
 - `ranks_predicted()` → `add_ranking_columns()` (función con nombre más claro)
 - `find_top_7_accuracy()` → `calculate_top7_average_precision()` (nombre más preciso)
 - `back_test()` → `backtest_model()` (nomenclatura estándar)

 ---

 ## Dependencias

 ```python
 import pandas as pd
 from sklearn.linear_model import Ridge
 from sklearn.metrics import mean_squared_error
 from sklearn.preprocessing import StandardScaler
 from sklearn.ensemble import RandomForestRegressor
 ```

 ---

 ## Archivos de salida

 Este módulo produce resultados en memoria. Para guardar predicciones:
 ```python
 comparison_df.to_csv('mvp_predictions_2024.csv', index=False)
 all_predictions.to_csv('historical_predictions.csv', index=False)
 ```

 ---

 ## Preguntas o problemas?

 Revisa este README junto con los comentarios del código. Cada sección se construye sobre la anterior, así que la comprensión fluye de arriba hacia abajo.