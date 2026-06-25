## What this system does

Think of it like a smart planner for machine energy:

1. **Reads your meter history**
2. **Learns your daily behavior** (working days, weekends, shutdown patterns)
3. **Predicts total energy per day**
4. **Breaks that daily total into hour-by-hour values**
5. **Creates a 48-hour near-term forecast** that is more responsive in first few hours
6. **Checks itself with validation metrics** so you know if it is reliable

---

## End-to-end flow (plain English)

### 1) Data cleaning and preparation
- We take raw energy meter readings from MongoDB.
- Convert them into hourly values.
- Calculate hourly consumption change.
- Add date/time context (hour, weekday, month, holiday, etc.).

Why: this gives the model clean and structured history.

---

### 2) Learn “daily pattern” first
- The system first decides: **Will this be an active day or a shutdown/low-load day?**
- If active, it predicts **how much total energy that day will consume**.

Why: daily total is easier and more stable to predict first.

---

### 3) Improve day decision quality
- It tunes the internal decision threshold so shutdown/active classification is balanced.
- It also adjusts for recency (recent behavior may differ from old behavior).

Why: this avoids over/under-predicting when operations recently changed.

---

### 4) Build hourly shape (v2.4 improvement)
- Instead of only using fixed day-of-week templates, we now use a dedicated **day-ahead hourly model** to generate a more realistic hourly curve.
- It uses recent hourly history patterns (same hour yesterday/last week, rolling behavior).

Why: this significantly improved hourly quality (MAPE dropped from ~33.8% to ~29.3%).

---

### 5) 7-day forecast output
- For each future day, it predicts total daily energy.
- Then it creates hourly values for those days using learned hourly shape logic.
- Final output gives both:
  - daily forecast table
  - hourly forecast curve

---

### 6) 48-hour hybrid forecast
- First few hours: uses short-term hourly behavior (more reactive).
- Later hours: smoothly shifts toward daily-guided shape forecast.

Why: best of both worlds — short-term responsiveness + daily stability.

---

### 7) Validation and trust checks
After every run, it reports:
- Daily accuracy
- Hourly accuracy
- Total monthly error %
- Active-hour MAPE
- Short-term holdout performance

Why: you can quickly judge “Is this model safe to use now?”

---

## In one line

**Your pipeline now does: clean data → predict daily totals → shape them into realistic hourly forecasts → blend short-term + daily logic for 48h → validate and report confidence.**

---