# European Wheat Price Forecasting Model
## Game Plan für die Implementierung (aktualisiert)

---

## Executive Summary

Ziel ist eine **Quartalsprognose (Quarter-End)** für den europäischen Weizenpreis auf Basis von **Euronext MATIF Milling Wheat Futures**.

**Forecast-Target (verbindlich):**
- **Instrument:** MATIF Milling Wheat Futures
- **Serie:** **Front Month Continuous**, **additiv backadjusted**
- **Target:** **Settlement** am **letzten Handelstag des Quartals** (EUR/t)
- **As-Of (Primär-Forecast-Zeitpunkt):** **letzter Freitag-Close** vor dem Handelstag, der **10 Handelstage vor Quarter-End** liegt
- **Updates:** wöchentlich (Freitag EOD); Refresh bis Quarter-End möglich

**Datenzugang:** LSEG/Reuters, Bloomberg

---

## 0. Data Contract & Definitionen (muss vor Implementierung fixiert sein)

### 0.1 Trading Calendar
- Quarter-End = **letzter Handelstag** im Quartal (Euronext Handelstage)
- As-Of Datum wird ausschließlich über diesen Kalender berechnet

### 0.2 Price Fields
- **Features / As-Of:** Close
- **Target:** Settlement am Quarter-End
- **Fallback-Regel:** falls Settlement am Quarter-End fehlt → Close verwenden und im Output flaggen

### 0.3 Continuous Series (Front Month) – Roll & Backadjust (deterministisch)
**Backadjustment:** additiv (historische Preise werden um Roll-Differenz adjustiert)

**Roll-Kandidat:** nächster Kontrakt nach Expiry (Front → Next)

**Rolltrigger (entscheidet das System):**
1) **Primär:** Volume-Crossover (Next > Current) für **N** aufeinanderfolgende Handelstage  
2) **Fallback:** Open-Interest-Crossover (Next > Current) für **N** Tage, wenn Volume nicht zuverlässig/verfügbar  
3) **Guardrails:**
   - frühestens innerhalb von **G** Handelstagen vor Expiry rollen (verhindert „zu frühe“ Rolls)
   - spätestens **F** Handelstage vor Expiry **force-roll**, wenn Next verfügbar

**Roll-Event Logging:** Roll-Datum, from/to Contract, Close-Diff, cumulative adjustment

> Parameter (Default-Vorschlag): N=2, G=15, F=3 (später per Backtest feinjustieren)

### 0.4 Data Availability („latest available“ ohne Leakage)
- Für alle nicht-täglichen Inputs (Reports, Makro, Fundamentals):
  - jedes Datum hat ein `available_at` (Release Timestamp/Datum)
  - Feature-Wert am As-Of = **zu diesem Zeitpunkt zuletzt verfügbarer Wert**
  - Missingness wird zusätzlich als Flag gespeichert
- Revisions: wenn möglich „first print“/snapshots speichern; ansonsten strikt vendor timestamped „available_at“ verwenden

---

## Phase 1: Datenakquisition & Infrastruktur
**Geschätzte Dauer: 2–3 Wochen**

### 1.1 Preisdaten (MATIF Contracts)
**Minimum Dataset (contract-level, täglich):**
- date, contract_id
- close
- settlement (wenn verfügbar)
- volume
- open_interest

**Quelle:** LSEG/Reuters oder Bloomberg (Vendor-Adapter später austauschbar)

### 1.2 Contract Master / Expiry Calendar
- Contract-Ticker/ID Mapping
- Expiry/Last Trade Date je Kontrakt
- Konsistenzchecks (keine Lücken, monotone Expiries)

### 1.3 Storage & Reproducibility
- Raw Zone (immutable), Curated Zone (bereinigt), Feature Store (as-of)
- Versionierung von:
  - Rollregel-Parametern
  - Continuous-Serie
  - Feature-Definitionen

---

## Phase 2: Continuous-Serie & Kalenderlogik
**Geschätzte Dauer: 1–2 Wochen**

### 2.1 Trading Calendar Utilities
- Bestimme:
  - Quarter-End pro Quartal
  - Handelstag **10 Trading Days vor Quarter-End**
  - As-Of = letzter Freitag vor diesem Handelstag

### 2.2 Continuous Builder
- Implementiere deterministische Rollregel (0.3)
- Erzeuge:
  - continuous_close_badj
  - continuous_settlement_badj (wenn verfügbar)
  - roll_events table
  - active_contract pro Tag

**Quality Gates:**
- keine „sprunghaften“ Artefakte außer Roll-Events
- Rollfrequenz plausibel (z.B. ~4–5 Rolls/Jahr je nach Kontraktmonaten)

---

## Phase 3: Feature Engineering (Tier-1 zuerst)
**Geschätzte Dauer: 2–3 Wochen**

### 3.1 Tier-1 (robust, immer verfügbar)
**Market/Technical (aus Continuous):**
- Returns: 1w / 4w
- Volatilität: 4w
- Momentum: 4w
- Volume/OI Changes: 1w
- Roll- bzw. Expiry-Proxies: `roll_event_recent`, `days_to_expiry` (aus Contract Master)

**Term Structure (optional aber empfehlenswert):**
- c1–c2 Spread / Slope als Regime-Signal

**Macro (lag-safe):**
- EUR/USD
- Energie (z.B. Brent, ggf. EU Gas) als Inputkosten-/Inflationsproxy

### 3.2 Tier-2 (nach MVP, wenn stabil)
- USDA/PSD/WASDE Derived Features (stocks-to-use etc.) strikt `available_at` gemerged
- Wetter-/Dürre-Indizes (stabile, region-aggregierte Anomalien)
- Geopolitik/Sentiment nur wenn dauerhaft reproduzierbar

### 3.3 Missingness & Release-Day Handling
- Forward-fill nur bis zur nächsten Veröffentlichung
- Missing Flags pro Feature

---

## Phase 4: Dataset Construction (Quarter-End Supervised Task)
**Geschätzte Dauer: 1 Woche**

### 4.1 Primär-Row je Quartal
- X = Features am **As-Of Freitag**
- y = **Settlement** am Quarter-End (fallback Close)

### 4.2 Wöchentliche Updates (empfohlen)
- Zusätzliche Rows für weitere Freitage zwischen As-Of und Quarter-End
- Feature `weeks_to_qend` zur Stabilisierung gegen Timing-Shift
- Mindestabstand (z.B. nicht innerhalb der letzten 3 Handelstage)

---

## Phase 5: Modell-Entwicklung
**Geschätzte Dauer: 2–3 Wochen**

### 5.1 Baselines (Pflicht)
- Naiv: letzter verfügbarer Preis/Trend (z.B. last Friday close)
- Linear (Ridge/ElasticNet) als interpretierbare Benchmark

### 5.2 Hauptmodell (empfohlen)
- Gradient Boosted Trees (robust bei mixed features + missingness)
- Optional: Quantile-Modelle (Intervalle 80/95%) für Risikoabschätzung

### 5.3 Modellwahl-Kriterien
- stabile Performance über Quartale
- Stressquartale separat auswerten
- keine „magische“ Backtest-Performance (Leakage-Checks)

---

## Phase 6: Backtesting & Validation (Quarter-aware)
**Geschätzte Dauer: 1–2 Wochen**

### 6.1 Walk-Forward Setup
- Train: alle Quartale bis Q(t-1)
- Test: Q(t) (inkl. wöchentlicher Update-Rows)
- Optional: expanding window vs. fixed window vergleichen

### 6.2 Metriken (Quartalsziel)
- **MAE (EUR/t)** – primär
- RMSE (EUR/t)
- MAPE (%) – sekundär
- Directional Accuracy optional (nur wenn Use-Case relevant)

### 6.3 Stress-/Regime-Reporting
- separater Score für „Stress“-Quartale (z.B. 2022)
- Feature Drift & Data Drift Checks (vor allem Roll-/Expiry-Umfeld)

---

## Phase 7: Deployment & Monitoring
**Geschätzte Dauer: 1–2 Wochen**

### 7.1 Weekly Production Job (Freitag EOD)
- Pull neue Daten
- Rebuild Continuous + Features
- Wenn As-Of oder danach bis Quarter-End:
  - Forecast Quarter-End Settlement
  - optional Intervalle + Treiber-Report

### 7.2 Monitoring & Runbooks
- Data Freshness (Vendor-Feeds)
- Missingness-Spikes
- Roll-Anomalien (zu häufig/zu selten)
- Performance Drift pro Quartal/Horizon
- Runbook: „Report delayed“, „Vendor outage“, „contract master mismatch“

---

## Deliverables (Definition of Done)
- deterministische Continuous-Serie + Roll-Log
- reproduzierbarer Quarter-End Dataset Builder (As-Of Regel)
- leak-safe „latest available“ Feature Join
- Walk-forward Backtest Report (MAE/RMSE/MAPE) inkl. Stressquartale
- Weekly Forecast Output (CSV/JSON) + Monitoring Checks
