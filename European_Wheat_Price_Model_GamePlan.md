# European Wheat Price Forecasting Model
## Game Plan für die Implementierung

---

## Executive Summary

Basierend auf dem bewährten Coffee Price Model soll ein analoges System für **Euronext (MATIF) Milling Wheat** entwickelt werden. Der Fokus liegt auf europäischen Weizenpreisen mit besonderer Berücksichtigung der Black Sea Region (Russland/Ukraine), die als Preistreiber Nr. 1 fungiert.

**Ziel-Benchmark:** MAPE < 8% (vergleichbar mit Coffee Model)
**Forecast-Horizont:** 3-12 Monate
**Target:** Euronext Milling Wheat (EBM) in EUR/t

---

## Phase 1: Datenakquisition & Infrastruktur
**Geschätzte Dauer: 2-3 Wochen**

### 1.1 Preisdaten (Äquivalent zu Arabica/Robusta)

| Datenquelle | Was | Frequenz | Zugang |
|-------------|-----|----------|--------|
| **LSEG/Reuters** | Euronext Milling Wheat Futures (EBM) | Täglich | LSEG Workspace ✓ |
| **Bloomberg** | MATIF Wheat Historical | Täglich | Bloomberg Terminal ✓ |
| **Barchart** | Euronext Wheat Quotes | Täglich/Kostenlos | API verfügbar |
| **CBOT** | Chicago SRW Wheat (Benchmark) | Täglich | Für Spread-Analyse |

**Mindestanforderung:** 10+ Jahre History (2015-2025)

### 1.2 Fundamentale Daten

#### A) Stocks/Inventories (Äquivalent zu ICE Coffee Stocks)

| Datenquelle | Was | Frequenz | Priorität |
|-------------|-----|----------|-----------|
| **USDA WASDE** | Global Wheat Ending Stocks | Monatlich | ⭐⭐⭐⭐⭐ |
| **USDA PSD** | EU/Russia/Ukraine Stocks | Monatlich | ⭐⭐⭐⭐⭐ |
| **Eurostat** | EU Intervention Stocks | Monatlich | ⭐⭐⭐ |
| **FranceAgriMer** | French Wheat Stocks | Wöchentlich | ⭐⭐⭐⭐ |

**Kritisch:** Stocks-to-Use Ratio ist der BESTE Einzelprediktor (Korrelation -0.85 bei Kaffee)

#### B) Production & Yield Data

| Datenquelle | Was | Frequenz | API |
|-------------|-----|----------|-----|
| **JRC MARS Bulletin** | EU Crop Conditions & Yield Forecasts | Monatlich | Agri4Cast Toolbox |
| **USDA FAS** | Global Production Estimates | Monatlich | PSD Online |
| **Eurostat** | EU Production (apro_cpsh1) | Jährlich | Eurostat API |
| **Strategie Grains** | EU Crop Forecasts | Monatlich | Subscription |

#### C) Trade/Export Data

| Datenquelle | Was | Frequenz | Priorität |
|-------------|-----|----------|-----------|
| **EC Agridata Portal** | EU Wheat Exports/Imports | Monatlich | ⭐⭐⭐⭐⭐ |
| **Eurostat COMEXT** | Detaillierte Handelsströme | Monatlich | ⭐⭐⭐⭐ |
| **USDA Export Sales** | US Weekly Export Sales | Wöchentlich | ⭐⭐⭐ |
| **SovEcon/APK-Inform** | Black Sea Export Estimates | Wöchentlich | ⭐⭐⭐⭐⭐ |

### 1.3 COT Data (Spekulanten-Positionierung)

| Datenquelle | Contract | Frequenz | Zugang |
|-------------|----------|----------|--------|
| **CFTC** | CBOT Wheat Futures | Wöchentlich (Freitag) | Kostenlos |
| **Euronext** | MATIF Milling Wheat | Wöchentlich | Euronext Reports |

**Python-Library:** `cot_reports` (GitHub: NDelventhal/cot_reports)

```python
# Beispiel COT Download
from cot_reports import cot_all
df_cot = cot_all(report_type='legacy_fut')
wheat_cot = df_cot[df_cot['Contract Market Name'].str.contains('WHEAT')]
```

---

## Phase 2: Feature Engineering
**Geschätzte Dauer: 2-3 Wochen**

### 2.1 Kern-Features (Mapping vom Coffee Model)

| Coffee Model Feature | Wheat Äquivalent | Beschreibung |
|---------------------|------------------|--------------|
| ONI/ENSO Index | **NAO Index + European Weather** | North Atlantic Oscillation beeinflusst EU-Wetter |
| ICE Coffee Stocks | **Global Wheat Stocks-to-Use** | USDA WASDE Ending Stocks / Total Use |
| USD/BRL | **EUR/USD + RUB/USD** | Währungseffekte EU vs. Black Sea |
| COT Net Position | **COT Wheat (CBOT + MATIF)** | Spekulanten-Sentiment |
| Brazil Production | **Black Sea Production** | Russia + Ukraine Output |

### 2.2 Weizen-spezifische Features

#### A) Supply-Side Features
```python
# Tier 1 - Must Have
features_supply = [
    'global_stocks_to_use',           # WASDE: Ending Stocks / Total Use
    'eu_production_forecast',          # JRC MARS
    'russia_production',               # USDA PSD
    'ukraine_production',              # USDA PSD
    'black_sea_exports_monthly',       # LSEG Trade Flow
    'france_crop_condition',           # FranceAgriMer
]

# Tier 2 - Nice to Have
features_supply_advanced = [
    'eu_winter_wheat_area',           # Eurostat
    'spring_wheat_area_us',           # USDA Prospective Plantings
    'australia_production',           # Southern Hemisphere Offset
    'india_wheat_exports',            # Policy-driven
]
```

#### B) Demand-Side Features
```python
features_demand = [
    'eu_wheat_exports',               # Eurostat COMEXT
    'egypt_wheat_imports',            # Largest global importer
    'algeria_morocco_imports',        # Key MENA markets
    'china_wheat_imports',            # Variable demand
    'global_wheat_trade_volume',      # USDA
]
```

#### C) Macro & Currency Features
```python
features_macro = [
    'eur_usd',                        # FRED API
    'usd_rub',                        # Russian Ruble
    'usd_uah',                        # Ukrainian Hryvnia
    'brent_crude',                    # Energy/Fertilizer costs
    'natural_gas_ttf',                # EU Gas → Fertilizer
    'urea_price',                     # Fertilizer proxy
]
```

#### D) Weather/Climate Features
```python
features_weather = [
    'nao_index',                      # North Atlantic Oscillation
    'eu_soil_moisture_anomaly',       # JRC MARS / Copernicus
    'france_precipitation_anomaly',   # Open-Meteo API
    'germany_temperature_anomaly',    # Open-Meteo API
    'black_sea_drought_index',        # SPEI/SPI
    'us_winter_wheat_condition',      # USDA Crop Progress
]
```

#### E) Market Sentiment Features
```python
features_sentiment = [
    'cot_net_position_cbot',          # CFTC
    'cot_net_position_matif',         # Euronext
    'cot_momentum_4w',                # 4-week change
    'cot_extreme_long',               # > 90th percentile
    'cot_extreme_short',              # < 10th percentile
    'wheat_corn_spread',              # Substitution indicator
    'matif_cbot_spread',              # Regional premium
]
```

### 2.3 Geopolitische Features (Weizen-spezifisch!)

```python
# CRITICAL: Black Sea Risk Factor
features_geopolitical = [
    'ukraine_war_intensity_proxy',    # News sentiment / Events
    'russia_export_quota_active',     # Binary: 1 wenn aktiv
    'russia_export_tax_rate',         # Aktueller Steuersatz
    'black_sea_shipping_disruption',  # Versicherungs-/Frachtkosten
    'eu_solidarity_lanes_volume',     # Alternative Ukraine Exports
]
```

---

## Phase 3: Modell-Entwicklung
**Geschätzte Dauer: 3-4 Wochen**

### 3.1 Modell-Architektur (basierend auf Coffee Model Learnings)

| Schritt | Modell | Zweck |
|---------|--------|-------|
| 1 | ARIMA/Prophet | Baseline + Trend/Seasonality |
| 2 | XGBoost | Feature Importance + Quick Iteration |
| 3 | LSTM | Temporal Dependencies |
| 4 | **Hybrid XGBoost-LSTM** | Best of Both |
| 5 | Ensemble | Robustheit |

### 3.2 Empfohlener Ansatz

```python
# Schritt-für-Schritt wie bei Coffee Model

# 1. Baseline (ARIMA)
from statsmodels.tsa.arima.model import ARIMA
model_baseline = ARIMA(wheat_prices, order=(1,1,1))

# 2. XGBoost mit allen Features
import xgboost as xgb
model_xgb = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)

# 3. Regime-basiertes Modell (aus Coffee Model)
# - Bullish Regime: Stocks-to-Use < 25%, Black Sea Disruption
# - Bearish Regime: Stocks-to-Use > 30%, Strong Harvest
# - Neutral: Alles dazwischen

# 4. Feature Importance Analyse
importance = model_xgb.feature_importances_
# Erwartete Top-Features:
# 1. global_stocks_to_use
# 2. black_sea_exports
# 3. eur_usd
# 4. cot_net_position
# 5. nao_index
```

### 3.3 Training/Test Split

```
Training:   2015-01-01 bis 2023-12-31 (9 Jahre)
Validation: 2024-01-01 bis 2024-06-30 (6 Monate)
Test:       2024-07-01 bis 2025-12-31 (Walk-Forward)
```

---

## Phase 4: Datenquellen-Integration (Code)
**Geschätzte Dauer: 2-3 Wochen**

### 4.1 Kostenlose APIs

```python
# FRED API (Makro-Daten)
from fredapi import Fred
fred = Fred(api_key='DEIN_FRED_KEY')
eur_usd = fred.get_series('DEXUSEU')

# Open-Meteo (Wetter - KOSTENLOS!)
import openmeteo_requests
# France (Paris Basin - Hauptanbaugebiet)
url = "https://archive-api.open-meteo.com/v1/era5"
params = {
    "latitude": 48.8566,
    "longitude": 2.3522,
    "start_date": "2015-01-01",
    "end_date": "2025-12-31",
    "daily": ["temperature_2m_mean", "precipitation_sum"]
}

# CFTC COT Reports
from cot_reports import cot_all
df_cot = cot_all(report_type='disaggregated_fut')
```

### 4.2 LSEG Workspace Integration

```python
# Du hast bereits LSEG Workspace Zugang!
# Für Euronext Wheat:
# - RIC: EBMc1 (Front Month)
# - RIC: EBMc2, EBMc3 (Forward Curve)

# Beispiel LSEG API Call (falls API Zugang)
import lseg.data as ld
ld.open_session()
wheat_data = ld.get_history(
    universe=['EBMc1'],
    fields=['CLOSE', 'VOLUME', 'OPEN_INT'],
    start='2015-01-01',
    end='2025-12-31'
)
```

### 4.3 Eurostat API

```python
import eurostat

# Wheat Production
wheat_prod = eurostat.get_data_df('apro_cpsh1')
wheat_prod = wheat_prod[wheat_prod['crops'] == 'C1100']  # Common wheat

# EU Trade Data
trade_data = eurostat.get_data_df('DS-016894')
```

---

## Phase 5: Backtesting & Validation
**Geschätzte Dauer: 2 Wochen**

### 5.1 Metriken (wie bei Coffee Model)

| Metrik | Ziel | Beschreibung |
|--------|------|--------------|
| MAPE | < 8% | Mean Absolute Percentage Error |
| RMSE | < 15 EUR/t | Root Mean Square Error |
| Direction Accuracy | > 60% | Trefferquote Richtung |
| Sharpe Ratio | > 1.0 | Risk-adjusted Returns (für Trading) |

### 5.2 Walk-Forward Validation

```python
# Rolling Window Approach
window_train = 252 * 3  # 3 Jahre täglich
window_test = 63        # 3 Monate

for i in range(n_windows):
    train = data[i:i+window_train]
    test = data[i+window_train:i+window_train+window_test]
    model.fit(train)
    predictions = model.predict(test)
    evaluate(predictions, test)
```

### 5.3 Stress-Tests

| Szenario | Zeitraum | Erwartung |
|----------|----------|-----------|
| Ukraine-Invasion 2022 | Feb-Jun 2022 | Modell erkennt Regime-Shift |
| Black Sea Initiative Ende | Jul 2023 | Price Spike korrekt |
| EU Dürre 2022 | Jun-Sep 2022 | Yield Impact erfasst |

---

## Phase 6: Deployment & Monitoring
**Geschätzte Dauer: 1-2 Wochen**

### 6.1 Automatisierung

```python
# Täglicher/Wöchentlicher Update-Prozess
schedule = {
    'daily': ['prices', 'eur_usd', 'cot_positions'],
    'weekly': ['wasde_check', 'export_data', 'model_retrain'],
    'monthly': ['full_model_update', 'feature_review']
}
```

### 6.2 Output-Format

| Output | Frequenz | Format |
|--------|----------|--------|
| Point Forecast | Wöchentlich | EUR/t (3M, 6M, 12M) |
| Confidence Interval | Wöchentlich | 80% & 95% Bands |
| Regime Signal | Wöchentlich | Bullish/Neutral/Bearish |
| Key Driver Report | Monatlich | Top 5 Features |

---

## Datenquellen-Übersicht (Zusammenfassung)

| Kategorie | Quelle | Kosten | API |
|-----------|--------|--------|-----|
| **Preise** | LSEG Workspace | ✓ (bereits vorhanden) | ✓ |
| **Preise** | Bloomberg | ✓ (bereits vorhanden) | ✓ |
| **Fundamentals** | USDA WASDE/PSD | Kostenlos | ✓ |
| **EU Produktion** | JRC MARS | Kostenlos | ✓ |
| **EU Produktion** | Eurostat | Kostenlos | ✓ |
| **Trade Data** | EC Agridata | Kostenlos | ✓ |
| **COT** | CFTC | Kostenlos | ✓ |
| **Wetter** | Open-Meteo | Kostenlos | ✓ |
| **Makro** | FRED | Kostenlos | ✓ |
| **Black Sea Intel** | SovEcon | €€€ | - |

---

## Risiken & Mitigation

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|--------|------------|
| Black Sea Disruption | Hoch | Sehr Hoch | Geopolitik-Features + Regime-Modell |
| Data Lag WASDE | Mittel | Mittel | Alternative Quellen (SovEcon, Strategie Grains) |
| Modell-Overfitting | Mittel | Hoch | Walk-Forward Validation |
| Strukturbruch (Policy) | Niedrig | Hoch | Regime-Detection + Manual Override |

---

## Nächste Schritte (Empfohlene Reihenfolge)

1. **Woche 1-2:** Preisdaten Setup
   - [ ] Euronext Wheat History von LSEG exportieren
   - [ ] CBOT Wheat als Vergleichsbenchmark
   - [ ] Datenqualitätsprüfung

2. **Woche 3-4:** Fundamentale Daten
   - [ ] USDA WASDE Parser bauen
   - [ ] Eurostat API Integration
   - [ ] JRC MARS Bulletin Scraper

3. **Woche 5-6:** Feature Engineering
   - [ ] Stocks-to-Use Berechnung
   - [ ] COT Integration
   - [ ] Wetter-Features

4. **Woche 7-8:** Baseline Modell
   - [ ] ARIMA Baseline
   - [ ] XGBoost erste Version
   - [ ] Feature Importance Analyse

5. **Woche 9-10:** Modell-Optimierung
   - [ ] Regime-Modell (wie bei Coffee)
   - [ ] Hybrid Approach
   - [ ] Hyperparameter Tuning

6. **Woche 11-12:** Backtesting & Launch
   - [ ] Walk-Forward Validation
   - [ ] Stress-Tests
   - [ ] Automatisierung

---

## Fazit

Die Übertragung des Coffee Price Models auf europäischen Weizen ist gut machbar, da:

1. **Ähnliche Treiber:** Stocks-to-Use, Wetter, Spekulanten-Positionen
2. **Bessere Datenlage:** USDA WASDE ist robuster als ICE Coffee Stocks
3. **Deine Tools:** LSEG + Bloomberg bereits vorhanden
4. **Geopolitik-Challenge:** Black Sea ist komplexer als Brasilien-Wetter, aber modellierbar

**Hauptunterschied zu Kaffee:** Der geopolitische Faktor (Russland/Ukraine) ist bei Weizen deutlich dominanter als ENSO bei Kaffee. Das Modell muss Regime-Shifts durch Policy-Entscheidungen (Exportquoten, Sanktionen) erfassen können.

---

*Erstellt: Februar 2026*
*Basierend auf: Coffee Price Forecasting Model v4*
