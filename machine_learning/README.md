# NBA MVP Prediction - Machine Learning Module

## Project Overview

This module implements machine learning models to predict NBA MVP (Most Valuable Player) voting shares based on player statistics and team performance. The goal is to accurately identify the top MVP candidates each season.

---

## Table of Contents

1. [Data Loading and Preparation](#1-data-loading-and-preparation)
2. [Feature Selection](#2-feature-selection)
3. [Initial Model Training](#3-initial-model-training)
4. [Evaluation Metrics](#4-evaluation-metrics)
5. [Backtesting Framework](#5-backtesting-framework)
6. [Feature Engineering](#6-feature-engineering)
7. [Model Comparison](#7-model-comparison)

---

## 1. Data Loading and Preparation

### What it does:
```python
stats = pd.read_csv("../data cleaning/combined_stats_master.csv")
del stats["Unnamed: 0"]
```

**Purpose:** Loads the cleaned and combined NBA statistics dataset that contains player performance metrics, team stats, and MVP voting shares from 1991-2024.

**Why:** This is the foundation of the entire analysis. The data has been previously scraped and cleaned in other modules.

**Column Removal:** The `Unnamed: 0` column is removed because it's just an index column created during CSV export and provides no predictive value.

---

## 2. Feature Selection

### What it does:
```python
predictor_features = ['Age','G', 'GS', 'MP', 'FG', 'FGA', 'FG%',
       '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%',
       'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Year',
        'W', 'L', 'W/L%', 'GB', 'PS/G', 'PA/G', 'SRS']
```

**Purpose:** Defines which statistics will be used to predict MVP share.

**Why these features:**
- **Player Stats:** Individual performance (points, assists, steals, blocks, shooting percentages)
- **Team Performance:** Team wins/losses, win percentage, point differential
- **Playing Time:** Games played, minutes played
- **Advanced Metrics:** eFG% (effective field goal %), SRS (Simple Rating System)

**What we're NOT including:** The target variable `Share` (MVP voting share percentage) and related columns like `Pts Won`, `Pts Max` - we want to predict these, not use them as inputs.

---

## 3. Initial Model Training

### Train/Test Split:
```python
train_data = stats[stats["Year"] < 2024]
test_data = stats[stats["Year"] == 2024]
```

**Purpose:** Separates historical data (1991-2023) for training from recent data (2024) for testing.

**Why:** This simulates a real-world scenario where we train on past data and predict the most recent season. This is called temporal splitting and prevents data leakage.

### Ridge Regression Model:
```python
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(train_data[predictor_features], train_data["Share"])
```

**What is Ridge Regression:**
- A linear regression model with L2 regularization
- Prevents overfitting by penalizing large coefficients
- The `alpha=0.1` parameter controls the regularization strength

**Why Ridge instead of Linear Regression:**
- Many features are correlated (e.g., FG and PTS)
- Ridge handles multicollinearity better
- Reduces variance in predictions

### Making Predictions:
```python
predictions_2024 = ridge_model.predict(test_data[predictor_features])
predictions_df = pd.DataFrame(predictions_2024, columns=["Predicted"], index=test_data.index)
comparison_df = pd.concat([test_data[["Player","Share"]], predictions_df], axis=1)
```

**Purpose:** Generate MVP share predictions for 2024 and create a comparison table.

**Why:** Allows us to see how our predictions compare to actual MVP voting results.

---

## 4. Evaluation Metrics

### Mean Squared Error (MSE):
```python
mean_squared_error(comparison_df["Share"], comparison_df["Predicted"])
```

**What it measures:** Average squared difference between predicted and actual values.

**Limitation:** As noted in the code, MSE isn't ideal for this problem because:
- Most players have Share = 0 (didn't receive MVP votes)
- We care more about correctly identifying the TOP candidates
- A small error on the winner matters more than on rank 50

### Custom Metric - Top 7 Average Precision:
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

**How it works:**
1. Identify the actual top 7 MVP candidates
2. Go through predictions in order (from highest to lowest predicted share)
3. Each time we find a correct top-7 player, calculate precision at that point
4. Average all precision values

**Example:**
- If our top 3 predictions are all correct: precision = (1/1 + 2/2 + 3/3) / 3 = 1.0
- If we find them at positions 2, 5, 7: precision = (1/2 + 2/5 + 3/7) / 3 ≈ 0.48

**Why this metric:**
- Focuses on ranking quality for top candidates
- Rewards finding correct players early
- Penalizes placing actual top-7 players lower in predictions
- Better represents the real goal: identifying MVP frontrunners

---

## 5. Backtesting Framework

### Purpose of Backtesting:
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

**What it does:**
- Tests the model on multiple years (1996-2024)
- For each year, trains on all previous years and predicts that year
- Calculates average precision for each year
- Returns overall mean average precision

**Why skip first 5 years:**
- Need sufficient training data (at least 5 years of history)
- Ensures stable model training

**Benefits:**
- More robust evaluation than single year test
- Shows if model works consistently across different eras
- Reveals temporal patterns (is model better for recent years?)

### Ranking Analysis:
```python
def add_ranking_columns(comparison):
    ranked_comparison = comparison.sort_values("Share", ascending=False)
    ranked_comparison["Rank"] = list(range(1, ranked_comparison.shape[0] + 1))
    
    ranked_comparison = ranked_comparison.sort_values("Predicted", ascending=False)
    ranked_comparison["Predicted Rank"] = list(range(1, ranked_comparison.shape[0] + 1))
    
    ranked_comparison["Difference"] = ranked_comparison["Rank"] - ranked_comparison["Predicted Rank"]
    
    return ranked_comparison
```

**Purpose:** Adds actual rank, predicted rank, and their difference.

**Difference interpretation:**
- **Positive difference:** Model underrated the player (actual rank better than predicted)
- **Negative difference:** Model overrated the player (predicted rank better than actual)
- **Zero:** Perfect prediction for that player's rank

**Why useful:** Helps identify systematic biases (does model favor/disfavor certain player types?)

---

## 6. Feature Engineering

### Year-Normalized Statistics:
```python
normalized_stats = stats.groupby("Year")[["PTS", "AST", "STL", "BLK", "3P"]].apply(
    lambda x: x / x.mean(), include_groups=False
)

stats[["PTS_R", "AST_R", "STL_R", "BLK_R", "3P_R"]] = normalized_stats[["PTS", "AST", "STL", "BLK", "3P"]]
```

**What it does:** Creates ratio features by dividing each stat by the yearly average.

**Why this is critical:**
- **Era adjustment:** NBA game has evolved significantly (1990s vs 2020s)
- **Pace changes:** Teams score more points now due to faster pace
- **3-point revolution:** Modern players shoot many more 3-pointers
- **Normalization:** A player averaging 30 PPG in 1995 was more impressive than in 2024

**Example:**
- 1995: League average = 20 PPG, Player scores 30 → Ratio = 1.5
- 2024: League average = 25 PPG, Player scores 30 → Ratio = 1.2
- The 1995 player is more exceptional relative to peers

**Impact:** These normalized features should improve model accuracy across different eras.

### Categorical Encoding:
```python
stats["Position_Encoded"] = stats["Pos"].astype("category").cat.codes
stats["Team_Encoded"] = stats["Team"].astype("category").cat.codes
```

**What it does:** Converts text categories (positions like "PG", "SF" and team names) into numeric codes.

**Why:** Machine learning models require numeric inputs. This is a simple encoding where each unique category gets a number.

**Note:** These features are created but not added to `predictor_features` in the current version, so they're not being used yet. This could be a future improvement.

---

## 7. Model Comparison

### Random Forest Alternative:
```python
random_forest_model = RandomForestRegressor(n_estimators=50, random_state=1, min_samples_split=5)
```

**What is Random Forest:**
- Ensemble of 50 decision trees
- Each tree learns different patterns
- Final prediction = average of all trees
- `min_samples_split=5`: Need at least 5 samples to split a node

**Why try Random Forest:**
- Can capture non-linear relationships (Ridge is linear)
- Automatically handles feature interactions
- Less sensitive to outliers
- Can model complex MVP voting patterns

**Comparison:**
```python
# Random Forest on recent years (1991 + 28 = 2019 onwards)
mean_avg_precision_rf, _, _ = backtest_model(stats, random_forest_model, years_range[28:], predictor_features)

# Ridge on same period for fair comparison
mean_avg_precision_ridge, _, _ = backtest_model(stats, ridge_model, years_range[28:], predictor_features)
```

**Why test on years[28:] (2019-2024):**
- Random Forest can overfit on small datasets
- Testing on recent years with all historical training data
- Provides fair comparison between models
- Most relevant period for current predictions

---

## Key Insights and Model Strategy

### The MVP Problem:
1. **Narrative matters:** MVP voting isn't purely statistical - media narratives, team record, and "story" matter
2. **Top-heavy distribution:** Only 5-7 players get serious consideration each year
3. **Temporal evolution:** What makes an MVP has changed over decades

### Why This Approach Works:
1. **Temporal validation:** Backtesting ensures model works across eras
2. **Right metric:** Top-7 Average Precision focuses on what matters (finding candidates)
3. **Feature engineering:** Normalized stats account for era differences
4. **Model diversity:** Testing both linear (Ridge) and non-linear (Random Forest) approaches

### Potential Improvements:
1. Add categorical features (position, team) to predictors
2. Create interaction features (e.g., PTS × W/L%)
3. Include previous year's performance (momentum)
4. Add media coverage metrics if available
5. Ensemble both models for final prediction

---

## How to Use This Module

1. **Load data:** Ensure `combined_stats_master.csv` is in `../data cleaning/`
2. **Run sequentially:** Execute cells in order from top to bottom
3. **Interpret results:**
   - Higher `mean_avg_precision` = better model (max = 1.0)
   - Check `Difference` column to see over/underrated players
   - Review feature importance to understand what drives MVP voting

4. **Make predictions:**
   - Train on all available data
   - Use for next season's predictions
   - Monitor early season stats and update predictions

---

## Variable Naming Convention

**Before → After (Purpose):**
- `prediction` → `predictor_features` (list of input features)
- `train` → `train_data` (training dataset)
- `test` → `test_data` (test dataset)
- `reg` → `ridge_model` (Ridge regression model)
- `test_predictions` → `predictions_2024` (2024 predictions)
- `compare` → `comparison_df` (actual vs predicted comparison)
- `aps` → `average_precisions` (list of AP scores)
- `mean_ap` → `mean_avg_precision` (overall mean score)
- `sc` → `scaler` (StandardScaler object)
- `rf` → `random_forest_model` (Random Forest model)
- `ranks_predicted()` → `add_ranking_columns()` (clearer function purpose)
- `find_top_7_accuracy()` → `calculate_top7_average_precision()` (accurate metric name)
- `back_test()` → `backtest_model()` (standard naming)

---

## Dependencies

```python
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
```

---

## Output Files

This module produces in-memory results. To save predictions:
```python
comparison_df.to_csv('mvp_predictions_2024.csv', index=False)
all_predictions.to_csv('historical_predictions.csv', index=False)
```

---

## Questions or Issues?

Review this README alongside the code comments. Each section builds on the previous one, so understanding flows from top to bottom.
