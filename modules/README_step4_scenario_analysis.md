
# Step 4 Scenario Analysis (Minimal Viable Version)

This file upgrades the current Step 4 output from a **single proportional allocation table** into a **small scenario-analysis package**.

It is designed to work **directly on top of the current repo outputs**:

- `data/01-processed/step4_resource_deployment.csv`
- `data/01-processed/step3_hotspots_beats.csv`

So you do **not** need to rebuild Step 3 or rewrite the existing Step 4 logic first.

---

## What this script adds

The current Step 4 output already gives a single allocation table.  
This script adds **multiple deployment scenarios** and **proxy operational metrics** so the project can answer:

- What happens if we allocate by total demand only?
- What happens if we give extra priority to high-risk demand?
- What happens if we protect long-run hotspot beats with a minimum allocation?

Instead of only saying:

> “Here is how many units each beat gets.”

you can now say:

> “We compared several deployment policies and evaluated them with quantitative metrics.”

---

## Scenarios included

### 1) `status_quo`
Allocate units **proportionally to `AVG_CALLS`**.

This is the clean baseline policy:
- no extra risk weighting
- no hotspot floor
- just allocate more units where average demand is higher

---

### 2) `risk_aware_1.5`
Allocate proportionally to:

\[
\text{SCENARIO\_DEMAND} = \text{AVG\_CALLS} \times \left(1 + (1.5 - 1)\times \text{HIGH\_RISK\_RATIO}\right)
\]

This gives more resources to beats/shifts where the share of high-risk calls is larger.

---

### 3) `risk_aware_2.0`
Same as above, but with stronger emphasis on high-risk demand:

\[
\text{SCENARIO\_DEMAND} = \text{AVG\_CALLS} \times \left(1 + (2.0 - 1)\times \text{HIGH\_RISK\_RATIO}\right)
\]

This is a simple sensitivity check.

---

### 4) `hotspot_protect_top5_min2`
Use the Step 3 hotspot table (`step3_hotspots_beats.csv`) to identify the **top 5 hotspot beats**.

For each shift:
1. Give each hotspot beat at least **2 units**
2. Allocate the remaining units proportionally to `AVG_CALLS`

This makes the policy easier to explain operationally:

> “The highest-demand beats always keep a minimum level of coverage.”

---

## Proxy metrics used

We do not have true response time or dispatch completion data, so we use a simple capacity-based proxy.

### Capacity assumption
Assume one unit can handle:

\[
\mu = 3
\]

calls **per hour**.

Since each shift is currently 8 hours:
- Night = 00–07
- Day = 08–15
- Evening = 16–23

the per-shift capacity is:

\[
\text{CAPACITY} = \text{UNITS} \times \mu \times 8
\]

---

### Metrics

#### 1) `total_coverage`
\[
\text{COVERAGE} = \min(\text{AVG\_CALLS}, \text{CAPACITY})
\]

Summed across all beat-shift rows.

#### 2) `coverage_ratio`
\[
\text{coverage\_ratio} = \frac{\sum \text{COVERAGE}}{\sum \text{AVG\_CALLS}}
\]

#### 3) `total_shortfall`
\[
\text{SHORTFALL} = \max(0, \text{AVG\_CALLS} - \text{CAPACITY})
\]

Summed across all beat-shift rows.

#### 4) `peak_shortfall`
Find the top 10% highest-demand beat-shift rows (by `AVG_CALLS`) and sum their shortfall.

This shows how well each policy protects the busiest demand cells.

#### 5) `hotspot_shortfall`
Sum of shortfall restricted to hotspot beats only.

---

## Files produced

The script writes a new folder:

`data/01-processed/step4_scenarios/`

Inside it:

- `step4_scenario_allocations.csv`  
  Allocation + capacity + coverage + shortfall for every beat × shift × scenario

- `step4_scenario_summary.csv`  
  One summary row per scenario

- `scenario_coverage_ratio.png`  
  Bar chart comparing coverage ratio across scenarios

- `scenario_total_shortfall.png`  
  Bar chart comparing total shortfall across scenarios

- `allocation_top_beats_day.png`  
  Grouped bar chart showing how top beats are allocated in the selected shift



---

## How to run

From the repo root:

```bash
python modules/step4_scenario_analysis.py
```

### Example with defaults
```bash
python modules/step4_scenario_analysis.py \
  --base-allocation data/01-processed/step4_resource_deployment.csv \
  --hotspots data/01-processed/step3_hotspots_beats.csv \
  --outdir data/01-processed/step4_scenarios \
  --total-units 50 \
  --mu-per-hour 3.0 \
  --risk-weights 1.5 2.0 \
  --top-k 5 \
  --min-units 2 \
  --plot-shift Day
```

---

