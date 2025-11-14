# Predicción del MVP de la NBA

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Licencia](https://img.shields.io/badge/Licencia-MIT-green.svg)](LICENSE)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)](https://scikit-learn.org/)

Un proyecto completo de machine learning que predice los resultados de la votación del Jugador Más Valioso (MVP) de la NBA utilizando estadísticas históricas de jugadores, datos de rendimiento de equipos y métricas de evaluación personalizadas.

## Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Características Principales](#-características-principales)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Metodología](#-metodología)
- [Resultados](#-resultados)
- [Tecnologías Utilizadas](#-tecnologías-utilizadas)
- [Mejoras Futuras](#-mejoras-futuras)
- [Contribuir](#-contribuir)
- [Licencia](#-licencia)
- [Contacto](#-contacto)

## Descripción del Proyecto

Este proyecto implementa un pipeline completo de machine learning para predecir el porcentaje de votos del MVP de la NBA. Combina web scraping, limpieza de datos, ingeniería de características y modelado predictivo para identificar candidatos al MVP basándose en 34 años de datos de la NBA (1991-2024).

El proyecto demuestra habilidades en:
- **Web Scraping avanzado** con manejo de JavaScript y protección anti-bot
- **Limpieza de datos robusta** con manejo de casos especiales (traspasos, codificaciones)
- **Análisis exploratorio** con visualizaciones significativas
- **Machine Learning** con validación temporal y métricas personalizadas
- **Documentación profesional** siguiendo mejores prácticas

### ¿Por qué este Proyecto?

La votación del MVP en la NBA está influenciada por:
- Rendimiento individual del jugador (anotación, asistencias, rebotes, etc.)
- Éxito del equipo (porcentaje de victorias, clasificación)
- Métricas avanzadas (eficiencia, win shares)
- Factores específicos de la era (ritmo de juego, cambios de reglas)

Este proyecto aborda estos desafíos utilizando enfoques basados en datos y métricas de evaluación personalizadas centradas en la precisión del ranking.

## Características Principales

- **Web Scraping Automatizado**: Extrae 34 años de datos de la NBA desde Basketball-Reference.com
- **Limpieza Robusta de Datos**: Maneja traspasos de jugadores, problemas de codificación de nombres y valores faltantes
- **Ingeniería de Características**: Estadísticas normalizadas por año para tener en cuenta las diferencias entre eras
- **Métrica de Evaluación Personalizada**: Precisión Promedio Top-7 para evaluación centrada en rankings
- **Backtesting Temporal**: Prueba los modelos en múltiples temporadas para garantizar robustez
- **Múltiples Modelos**: Compara enfoques de Regresión Ridge, Random Forest y Gradient Boosting

## Estructura del Proyecto

```
NBA Predictions/
│
├── web_scraping/
│   ├── web_scraping_nba.ipynb          # Recolección de datos desde Basketball-Reference
│   ├── mvps.csv                         # Resultados de votación MVP
│   ├── players.csv                      # Estadísticas de jugadores
│   ├── teams.csv                        # Datos de rendimiento 
│   ├── mvps_data/                       # Archivos HTML guardados
│   ├── players_data/                    # Archivos HTML guardados
│   └── team_data/                       # Archivos HTML guardados
│
├── data cleaning/
│   ├── data_cleaning.ipynb              # Preprocesamiento e integración de datos
│   ├── combined_stats_master.csv        # Dataset final limpio
│   └── nicknames.txt                    # Mapeo de nombres de equipos
│
├── machine_learning/
│   └── machine_learning.ipynb           # Entrenamiento y evaluación de modelos
│
└── README.md                            # Este archivo
```

## Instalación

### Requisitos Previos

- Python 3.8 o superior
- Navegador Chrome/Chromium (para web scraping)
- ChromeDriver (incluido en `web_scraping/chromedriver-linux64/`)

### Configuración

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/Super1232/MVP-NBA-Predictions-.git
   cd "MVP-NBA-Predictions-"
   ```

2. **Crear un entorno virtual** (recomendado)
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Instalar paquetes necesarios**
   ```bash
   pip install -r requirements.txt
   ```

### Paquetes Python Requeridos

```txt
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.1.0
beautifulsoup4>=4.11.0
requests>=2.28.0
cloudscraper>=1.2.60
selenium>=4.5.0
matplotlib>=3.6.0
jupyter>=1.0.0
```

## Uso

### 1. Web Scraping (Opcional - Datos Ya Proporcionados)

```bash
cd "web_scraping"
jupyter notebook web_scraping_nba.ipynb
```

Ejecuta todas las celdas para extraer datos actualizados desde Basketball-Reference.com. **Nota**: El proyecto incluye datos ya extraídos, por lo que este paso es opcional.

### 2. Limpieza de Datos

```bash
cd "data cleaning"
jupyter notebook data_cleaning.ipynb
```

Este notebook:
- Limpia los datos de votación MVP
- Procesa las estadísticas de jugadores
- Maneja traspasos de jugadores y normalización de nombres
- Combina datos de rendimiento de equipos
- Exporta combined_stats_master.csv

### 3. Machine Learning

```bash
cd machine_learning
jupyter notebook machine_learning.ipynb
```

Este notebook:
- Carga el dataset limpio
- Entrena modelos de Regresión Ridge, Random Forest y Gradient Boosting
- Realiza backtesting temporal
- Crea características normalizadas por año
- Evalúa usando la métrica de Precisión Promedio Top-7

## Metodología

### 1. Recolección de Datos

**Fuente**: [Basketball-Reference.com](https://www.basketball-reference.com/)

**Datos Extraídos**:
- Resultados de votación MVP (1991-2024)
- Estadísticas por partido de jugadores
- Clasificación y rendimiento de equipos

**Desafíos Resueltos**:
- Tablas renderizadas con JavaScript (usando Selenium)
- Protección Cloudflare (usando cloudscraper)
- Múltiples fuentes de datos que requieren integración

### 2. Preprocesamiento de Datos

**Pasos Clave**:

1. **Manejo de Traspasos de Jugadores**: Los jugadores traspasados a mitad de temporada tienen múltiples entradas. Guardamos los totales de temporada y registramos su equipo final.

2. **Normalización de Nombres**: Corrección de problemas de codificación con nombres de jugadores internacionales (ej., "Jokić", "Giannis Antetokounmpo")

3. **Mapeo de Nombres de Equipos**: Conversión de abreviaturas (ej., "LAL") a nombres completos (ej., "Los Angeles Lakers")

4. **Tratamiento de Valores Faltantes**: 
   - Columnas de votación MVP: NaN → 0 (el jugador no recibió votos)
   - Porcentajes de tiro: NaN → 0 (sin intentos)

### 3. Ingeniería de Características

**Estadísticas Normalizadas por Año**:

Las estadísticas brutas no son comparables entre eras debido a:
- Diferencias en el ritmo de juego
- Cambios de reglas
- Revolución del triple

**Solución**: Crear características de ratio
```
Estadística_Normalizada = Estadística_Jugador / Promedio_Año
```

**Ejemplo**:
- 1995: Jugador anota 30 PPG (promedio liga: 20) → Ratio = 1.5
- 2024: Jugador anota 30 PPG (promedio liga: 25) → Ratio = 1.2

El jugador de 1995 es más excepcional relativo a su era.

### 4. Métrica de Evaluación

**Problema con Métricas Estándar**:
- **MSE/RMSE**: Tratan todos los errores por igual, pero nos importan más los candidatos principales

**Nuestra Solución: Precisión Promedio Top-7**

Se centra en clasificar correctamente a los 7 principales candidatos al MVP (aquellos que típicamente reciben votos).

**Cómo funciona**:
1. Identificar a los 7 mejores jugadores reales por porcentaje de votos
2. Clasificar a todos los jugadores por porcentaje predicho
3. Calcular la precisión cada vez que encontramos un jugador correcto del top-7
4. Promediar todos los valores de precisión

### 5. Backtesting Temporal

Para asegurar que los modelos funcionen en diferentes eras:

```python
for year in range(1996, 2025):
    train_data = stats[stats["Year"] < year]
    test_data = stats[stats["Year"] == year]
    
    model.fit(train_data, train_labels)
    predictions = model.predict(test_data)
    
    calculate_average_precision(predictions, actual_results)
```

Esto simula el uso en el mundo real: entrenar con datos pasados, predecir temporadas futuras.


## Resultados

### Comparación de Modelos

El proyecto evaluó tres algoritmos diferentes con optimización de hiperparámetros:

| Modelo | Cross-Validation (Top-7 AP) | Backtesting (Top-7 AP) | Desviación Estándar |
|--------|----------------------------|------------------------|---------------------|
| **Random Forest** | 51.21% | **77.82%** | ±10.2% |
| **Ridge Regression** | 51.87% | 76.55% | ±14.1% |
| **Gradient Boosting** | 56.62% | 72.33% | ±12.6% |

**Modelo Ganador: Random Forest Optimizado**
- **Parámetros**: n_estimators=200, max_depth=15, min_samples_split=2
- **Precisión en Backtesting**: 77.82% (29 temporadas: 1996-2024)
- **Precisión en Temporada 2024**: 93.79%
- **Estabilidad**: Mayor consistencia entre temporadas (menor desviación estándar)

### ¿Por qué Random Forest es Superior?

Aunque Gradient Boosting muestra mejor rendimiento en cross-validation (56.62%), **Random Forest domina en backtesting** (77.82% vs 72.33%) porque:

1. **Mejor generalización temporal**: Aprovecha conjuntos de datos históricos grandes
2. **Mayor robustez**: Predice mejor en situaciones completamente nuevas
3. **Más estable**: Menor variabilidad entre temporadas (σ = 0.102 vs σ = 0.126 de GB)

El backtesting completo simula el escenario de producción real (entrenar con toda la historia para predecir el futuro), haciéndolo la métrica definitiva.

### Insights Clave

**Características Más Predictivas** (análisis de importancia de Random Forest):

1. **PTS_R (Puntos Normalizados)** - 28.26%
   - Rendimiento ofensivo relativo a la época
   - Casi 3 veces más importante que la segunda característica

2. **W/L% (Porcentaje de Victorias)** - 12.81%
   - El MVP casi siempre viene de un equipo ganador
   - Factor crítico en la votación

3. **SRS (Simple Rating System)** - 6.94%
   - Métrica avanzada de rendimiento del equipo
   - Refuerza la importancia del contexto del equipo

4. **FG (Canastas de Campo)** - 5.40%
   - Volumen de producción ofensiva

5. **PTS (Puntos Totales)** - 4.81%
   - Anotación bruta sigue siendo relevante

**Hallazgo Crítico**: Las características normalizadas (PTS_R, AST_R, STL_R) dominan en importancia, validando que la normalización temporal fue esencial para comparar jugadores de diferentes épocas.

### Rendimiento en Temporada 2024

El modelo identificó correctamente a los principales candidatos al MVP:

**Métricas de Predicción 2024**:
- **Top-7 Average Precision**: 93.79%
- **MSE**: 0.000639
- **MAE**: 0.009134
- **R² Score**: 0.9821

El modelo rankeó casi perfectamente a los candidatos principales, demostrando su capacidad para identificar a los jugadores más probables de ganar el MVP basándose únicamente en estadísticas.

*Nota: Para ver los rankings detallados y predicciones específicas, ejecuta el notebook machine_learning.ipynb.*

## 🛠️ Tecnologías Utilizadas

### Programación y Ciencia de Datos
- **Python**: Lenguaje de programación principal
- **Pandas**: Manipulación y análisis de datos
- **NumPy**: Cálculos numéricos
- **Scikit-learn**: Modelos de machine learning y evaluación
  - Ridge Regression
  - Random Forest Regressor (modelo ganador)
  - Gradient Boosting Regressor
  - GridSearchCV con TimeSeriesSplit para optimización

### Web Scraping
- **BeautifulSoup4**: Análisis de HTML
- **Requests**: Biblioteca HTTP
- **Cloudscraper**: Evitar protección Cloudflare
- **Selenium**: Automatización de navegador para contenido renderizado con JavaScript

### Visualización y Desarrollo
- **Matplotlib**: Visualización de datos
- **Seaborn**: Visualizaciones estadísticas avanzadas
- **Jupyter Notebook**: Entorno de desarrollo interactivo

### Control de Versiones
- **Git**: Gestión de código fuente
- **GitHub**: Alojamiento de repositorio

## Mejoras Futuras

- [ ] Integrar análisis de sentimiento de medios deportivos
- [ ] Crear API REST para servir predicciones
- [ ] Expandir a otros premios NBA (Mejor Defensor, Novato del Año, Sexto Hombre)
- [ ] Incorporar datos de lesiones y cargas de trabajo

## Contacto

**Javier Poza Garijo**
- GitHub: [@Super1232](https://github.com/Super1232)
- LinkedIn: [Javier Poza Garijo](https://www.linkedin.com/in/javier-poza-garijo-ba3b88302)
- Email: javierpozagarijo.7@gmail.com

## Agradecimientos

- **Basketball-Reference.com** por proporcionar estadísticas completas de la NBA
- **Scikit-learn** comunidad por excelentes herramientas de machine learning
- **NBA** por décadas de baloncesto emocionante y datos ricos para análisis

## Referencias

1. **Dataquest** - [YouTube Channel](https://www.youtube.com/@Dataquestio)
2. **Basketball-Reference.com** - Estadísticas e Historia de la NBA
3. **Scikit-learn Documentation** - Machine Learning en Python
4. Artículos académicos sobre analítica deportiva y sistemas de ranking
5. **NBA Advanced Stats** - Conceptos de métricas avanzadas

---

**¡Si este proyecto te pareció útil, considera darle una estrella!** ⭐

*Última Actualización: Noviembre 2024*
