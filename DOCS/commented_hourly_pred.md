## 1. `energy_forecast/train_phase.py` DONE

| Lines | What it does | Why comment for “daily only” |
|------|----------------|------------------------------|
| **23–29** | Imports `DAYAHEAD_HOURLY_FEATURES`, `HOURLY_FEATURES`, `RECENT_WINDOW` (partly daily) | Only needed for hourly / day-ahead hourly paths below. |
| **41** | `from energy_forecast.hourly import ...` | Hourly models. |
| **42–46** | `hourly_shape` imports | Shape / hourly intra-day model. |
| **146–163** | Day-ahead hourly model (`train_hourly_model` on `DAYAHEAD_HOURLY_FEATURES`, `val_hourly_fc`) | Pure hourly head + feeds hourly part of validation. |
| **165–174** | `build_shape_feature_frame` / `train_shape_model` try/except | Hourly shape model. |
| **176** | `validation_metrics(val_daily, df, val_hourly_fc)` | Third arg is hourly forecast; you either comment this call and replace with a **daily-only** validation helper, or comment the hourly section **inside** `validation_metrics` in `daily.py` (see §4). |
| **216–235** | Last-7d hourly eval + `train_hourly_model` full + feature importance | Short-term hourly metrics + `xgb_hourly_full`. |
| **255–259** | `TrainPhaseResult` fields: `xgb_hourly_full`, `xgb_da_val`, `shape_full`, `hourly_short_term_metrics`, `hourly_feature_importance_top15` | You’d comment assignments and pass **stubs** (`None` / `{}` / `[]`) so the dataclass still constructs — that implies **small uncommented stub lines** unless you change the dataclass. |

**Keep (for daily):** **81–144** (load → preprocess → `build_features` → `build_daily_agg` → two-stage daily train/tune/val), **184–214** (full daily clf/reg + `rec_full`), and the **`return`** block with stubs for commented-out fields.

---

## 2. `energy_forecast/infer_phase.py` DONE

| Lines | What it does | Why comment |
|------|----------------|---------------|
| **15–17** | `decompose_daily_to_hourly`, `hybrid_forecast_48h`, `predict_future_hourly_recursive` | Intra-day expansion + 48h hybrid. |
| **38–39** | `xgb_hourly_full`, `shape_full` from `tr` | Only for hourly paths. |
| **98–111** | Build `future_hourly_df` from shape or decompose | **Intra-day distribution** after daily future rows. |
| **113–125** | `run_hybrid_48h` | 48h hourly forecast. |
| **129–130** | `InferPhaseResult` includes `future_hourly_df`, `fc_48h` | You’d return **empty DataFrames** (or same columns, zero rows) with a short non-commented stub, or change the dataclass. |

**Keep:** **43–97** (future **daily** loop with `DAILY_FEATURES_V22` — that is your daily prediction horizon).

---

## 3. `energy_forecast/pipeline.py` DONE

| Lines | What | Why comment / adjust |
|------|------|------------------------|
| **65–67** | `future_hourly_df`, `fc_48h` from `inf` | No hourly outputs if infer is daily-only. |
| **99–100** | `RunResult` `future_hourly`, `forecast_48h` | Would become `[]` or omitted if you change the dataclass. |

**Keep:** **98** `future_daily`, **80–101** val table and daily-oriented fields (you may still want **`hourly_rows` / `data_start` / `data_end`** as “history span” labels or comment those fields if you want the API to say “daily only”).

---

## 4. `energy_forecast/daily.py` DONE

| Lines | What | Note |
|------|------|------|
| **316–371** `validation_metrics` | **336–350** merge hourly actual vs `val_hourly_forecast`, hourly MAE/RMSE/MAPE | Comment **only this block** to drop hourly validation metrics while keeping **321–334** and **352–370** (daily + totals). |

**Optional (your “features that affect daily accuracy”):** **`build_daily_agg`** **31** (`daily_zero_hour_ratio` from hour-level zeros) and the chain **52–55, 79–82** (`last_72h_zero_hour_ratio`). Those are **daily-level** inputs in `DAILY_FEATURES_V22` but built from **intra-day** behaviour. Commenting them changes daily accuracy on purpose; you must keep `DAILY_FEATURES_V22` / model inputs in sync (in `constants.py`) if you remove columns.

---

## 5. `energy_forecast/features.py` (`build_features`) DONE

Used for the **hourly** frame before `build_daily_agg`. For **daily** models you only **require** the columns that `build_daily_agg` takes from the hourly frame via `groupby(...).agg(...)`: **`is_holiday`, `is_weekend`, `is_working_day`** (see `daily.py` **26–31**).

Typical **comment candidates** (intra-day / mainly for hourly & day-ahead hourly models, **not** in `DAILY_FEATURES_V22` list):

| Lines | Block |
|------|--------|
| **23–25** | `is_working_hour` |
| **27–31** | `days_to_next_holiday` |
| **33–36** | `hour_of_day`, `dow`, `month`, `day_of_month` on hourly rows (daily regen from `ds` in `build_daily_agg` for calendar parts of daily) |
| **38–43** | `hour_sin/cos`, `dow_sin/cos`, `mon_sin/cos` on hourly |
| **45–54** | `shift`, `shift_sin/cos` |
| **56–63** | `is_zero`, `hours_since_shutdown`, `just_restarted` |
| **65–76** | All `cons_lag_*`, `kwh_lag_*`, rolling means/std/min/max/range |

**Keep at minimum:** **20–22** (holiday/weekend/working day flags used by `build_daily_agg` `.agg(..., "first")`).

---

## 6. `energy_forecast/profiles.py` DONE    

| Lines | What |
|------|------|
| **11–60** entire `build_profiles` | Builds **24h fraction** profiles for DOW / day-type — only needed for **hourly** shape / decompose / hybrid. |

**Call site:** `train_phase.py` **103** `build_profiles(df, _gj)` — comment and replace with **stub** profiles (e.g. uniform 1/24) **if** you remove all infer paths that call `decompose_daily_to_hourly` / shape; otherwise infer will break.

---

## 7. `energy_forecast/model_registry.py`  DONE

| Lines | What |
|------|------|
| **132–135** | `joblib.dump` hourly, day-ahead, shape |
| **141–142** | CSV `future_hourly.csv`, `forecast_48h.csv` |
| **166–167, 170** | Meta: hourly metrics, importance, `has_shape` |
| **200–201, 207–208** | `artifact_paths_ok` hourly/shape file checks |
| **224–227** | `joblib.load` hourly, day-ahead, shape |

You’d **comment** these and either **skip loading** those artifacts in `load_train_phase_for_inference` or load **dummy** models (still needs a few non-commented lines unless inference no longer touches them).

---

## 8. `energy_forecast/server.py`  DONE

No separate hourly route; **`build_forecast_api_payload`** in `model_registry` still serializes hourly keys. Comment/adjust when `InferPhaseResult` / payload drops hourly series (see §10).

---

## 9. `energy_forecast/model_registry.py` — `build_forecast_api_payload`

| Lines (approx.) | Payload keys |
|------------------|--------------|
| End of function | `future_hourly`, `forecast_48h` in the returned dict (see file around **future_daily** / **forecast_48h**) | Comment or set to `[]` if API is daily-only. |

(Read the tail of `build_forecast_api_payload` in your file for exact line numbers.)

---

## 10. `energy_forecast/static/app.js` (and charts)

Grep-driven: any code that renders **`future_hourly`**, **`forecast_48h`**, “48h”, “hourly” charts, or KPIs tied to **`hourly_short_term_metrics` / `hourly_mape`**. Comment those **UI blocks** so the dashboard matches daily-only responses. (Exact line numbers shift with your current `app.js` length; search those strings.)

---

## 11. Files you **do not** need to comment for “no hourly **prediction**”

- **`preprocess.py`** — Still needed to build an **hourly** table so **`build_daily_agg`** can sum to daily **`y`** and compute **`daily_zero_hour_ratio`** unless you rewrite aggregation.
- **`hourly.py`**, **`hybrid.py`**, **`decompose.py`**, **`hourly_shape.py`** (module bodies) — Can stay **unchanged** if all **call sites** above are commented; no requirement to comment the whole file.

---

## 12. `energy_forecast/constants.py` (optional)

| Lines | Content |
|------|--------|
| **48–97** | `HOURLY_FEATURES`, `DAYAHEAD_HOURLY_FEATURES` | Optional: wrap in a comment block “reserved for hourly mode” if nothing imports them after train/infer changes. |

**`DAILY_FEATURES_V22`** (**13–46**): keep unless you remove **`daily_zero_hour_ratio` / `last_72h_zero_hour_ratio`** from `build_daily_agg` and from this list in lockstep.

---

### Summary

- **Mandatory comments** for “daily prediction only, no intra-day forecast”: **`train_phase`** hourly/shape/day-ahead + hourly eval; **`infer_phase`** hourly future + hybrid; **`pipeline`** hourly result fields; **`model_registry`** persist/load/payload for hourly artifacts; **`daily.validation_metrics`** hourly block.
- **Feature engineering comments** (intra-day, mainly for hourly models): **`features.py` ~23–76**, keeping **~20–22** for **`build_daily_agg`**.
- **Deliberate daily-accuracy tradeoff** (optional): **`daily.py`** `daily_zero_hour_ratio` / **`last_72h_zero_hour_ratio`** (+ matching **`constants.DAILY_FEATURES_V22`**).

Because **`TrainPhaseResult` / `InferPhaseResult` / `RunResult`** still expect several hourly-related fields, a real implementation almost always keeps a **few tiny stub assignments** (not “comment-only”) unless you also refactor those dataclasses.