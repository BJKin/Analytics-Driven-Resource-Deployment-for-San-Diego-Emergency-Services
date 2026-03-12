from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SHIFT_ORDER = ["Night", "Day", "Evening"]
SHIFT_HOURS = {"Night": 8, "Day": 8, "Evening": 8}


def _allocate_proportional(demand: pd.Series, total_units: int) -> pd.Series:
    """Allocate integer units proportionally to nonnegative demand."""
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    demand = demand.astype(float).clip(lower=0.0)
    if demand.empty:
        raise ValueError("demand series is empty")

    total_demand = float(demand.sum())
    if total_demand <= 0:
        out = pd.Series(0, index=demand.index, dtype=int)
        out.iloc[0] = total_units
        return out

    raw = demand / total_demand * total_units
    base = np.floor(raw).astype(int)
    remainder = int(total_units - base.sum())

    if remainder > 0:
        frac_order = (raw - base).sort_values(ascending=False).index.tolist()
        for idx in frac_order[:remainder]:
            base.loc[idx] += 1

    out = base.astype(int)
    if int(out.sum()) != total_units:
        raise AssertionError("allocated units do not sum to total_units")
    return out


def _allocate_hotspot_protect(
    demand: pd.Series,
    hotspot_mask: pd.Series,
    total_units: int,
    min_units: int,
) -> pd.Series:
    """
    Allocate units with a hotspot floor:
    1) Give each hotspot at least `min_units`
    2) Allocate remaining units proportionally to demand across all beats
    """
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if min_units < 0:
        raise ValueError("min_units must be nonnegative")

    demand = demand.astype(float).clip(lower=0.0)
    hotspot_mask = hotspot_mask.astype(bool).reindex(demand.index).fillna(False)

    alloc = pd.Series(0, index=demand.index, dtype=int)
    hotspot_idx = hotspot_mask[hotspot_mask].index.tolist()

    reserved = len(hotspot_idx) * min_units
    if reserved > total_units:
        # Graceful fallback: distribute one unit at a time across hotspots.
        units_left = total_units
        i = 0
        while units_left > 0 and len(hotspot_idx) > 0:
            alloc.loc[hotspot_idx[i % len(hotspot_idx)]] += 1
            units_left -= 1
            i += 1
        return alloc

    if hotspot_idx:
        alloc.loc[hotspot_idx] = min_units

    remaining_units = int(total_units - alloc.sum())
    if remaining_units > 0:
        extra = _allocate_proportional(demand, remaining_units)
        alloc = alloc + extra

    if int(alloc.sum()) != total_units:
        raise AssertionError("allocated units do not sum to total_units")
    return alloc


def load_base_step4_table(path: str | Path) -> pd.DataFrame:
    """Load the current step4_resource_deployment.csv output."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Base step4 CSV not found: {path}")

    df = pd.read_csv(path)
    required = {"BEAT", "AVG_CALLS", "HIGH_RISK_RATIO", "SHIFT"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in base step4 CSV: {sorted(missing)}")

    df = df.copy()
    df["BEAT"] = df["BEAT"].astype(str).str.strip()
    df["SHIFT"] = pd.Categorical(df["SHIFT"], categories=SHIFT_ORDER, ordered=True)
    df["AVG_CALLS"] = pd.to_numeric(df["AVG_CALLS"], errors="coerce").fillna(0.0)
    df["HIGH_RISK_RATIO"] = pd.to_numeric(df["HIGH_RISK_RATIO"], errors="coerce").fillna(0.0).clip(0, 1)
    df = df[df["BEAT"] != ""].copy()
    df = df.sort_values(["SHIFT", "BEAT"]).reset_index(drop=True)
    return df


def load_hotspot_set(path: str | Path, top_k: int) -> Set[str]:
    """Load the top-K hotspot beats from step3_hotspots_beats.csv."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Hotspot CSV not found: {path}")

    df = pd.read_csv(path)
    if "BEAT" not in df.columns:
        raise ValueError("Hotspot CSV must contain BEAT")
    if "CALLS" in df.columns:
        df = df.sort_values("CALLS", ascending=False)

    hotspot_set = set(df["BEAT"].astype(str).str.strip().head(top_k).tolist())
    if not hotspot_set:
        raise ValueError("Hotspot set is empty")
    return hotspot_set


def build_scenario_allocations(
    base_df: pd.DataFrame,
    hotspot_set: Set[str],
    total_units: int,
    risk_weights: Sequence[float],
    min_units: int,
) -> pd.DataFrame:
    """Create scenario allocations for all shifts."""
    out_frames: List[pd.DataFrame] = []

    scenario_specs = [("status_quo", None)]
    for w in risk_weights:
        scenario_specs.append((f"risk_aware_{w}", float(w)))
    scenario_specs.append((f"hotspot_protect_top{len(hotspot_set)}_min{min_units}", "hotspot"))

    for shift in SHIFT_ORDER:
        sub = base_df[base_df["SHIFT"] == shift].copy()
        if sub.empty:
            continue

        demand_status_quo = sub.set_index("BEAT")["AVG_CALLS"]
        hotspot_mask = sub.set_index("BEAT").index.to_series().isin(hotspot_set)

        for scenario_name, spec in scenario_specs:
            if scenario_name == "status_quo":
                units = _allocate_proportional(demand_status_quo, total_units)
                scenario_demand = demand_status_quo.copy()

            elif spec == "hotspot":
                units = _allocate_hotspot_protect(
                    demand=demand_status_quo,
                    hotspot_mask=hotspot_mask,
                    total_units=total_units,
                    min_units=min_units,
                )
                scenario_demand = demand_status_quo.copy()

            else:
                w = float(spec)
                scenario_demand = sub.set_index("BEAT")["AVG_CALLS"] * (
                    1.0 + (w - 1.0) * sub.set_index("BEAT")["HIGH_RISK_RATIO"]
                )
                units = _allocate_proportional(scenario_demand, total_units)

            frame = sub.set_index("BEAT")[["AVG_CALLS", "HIGH_RISK_RATIO"]].copy()
            frame["SHIFT"] = shift
            frame["SCENARIO"] = scenario_name
            frame["SCENARIO_DEMAND"] = scenario_demand
            frame["UNITS"] = units
            frame["IS_HOTSPOT"] = frame.index.to_series().isin(hotspot_set).astype(int)
            out_frames.append(frame.reset_index())

    if not out_frames:
        raise ValueError("No scenario allocations were created")
    return pd.concat(out_frames, ignore_index=True)


def add_capacity_metrics(df: pd.DataFrame, mu_per_hour: float) -> pd.DataFrame:
    """
    Add proxy operational metrics.

    AVG_CALLS is interpreted as average calls per beat per shift per day.
    If one unit can process `mu_per_hour` calls per hour, then shift capacity is:

        capacity = units * mu_per_hour * shift_hours
    """
    if mu_per_hour <= 0:
        raise ValueError("mu_per_hour must be positive")

    out = df.copy()
    out["SHIFT_HOURS"] = out["SHIFT"].map(SHIFT_HOURS).astype(float)
    out["CAPACITY"] = out["UNITS"].astype(float) * float(mu_per_hour) * out["SHIFT_HOURS"]
    out["COVERAGE"] = np.minimum(out["AVG_CALLS"].astype(float), out["CAPACITY"])
    out["SHORTFALL"] = np.maximum(0.0, out["AVG_CALLS"].astype(float) - out["CAPACITY"])
    return out


def summarize_scenarios(df: pd.DataFrame, peak_quantile: float = 0.90) -> pd.DataFrame:
    """Create one summary row per scenario."""
    if not (0 < peak_quantile < 1):
        raise ValueError("peak_quantile must be between 0 and 1")

    rows = []
    demand_threshold = float(df["AVG_CALLS"].quantile(peak_quantile))

    for scenario, sub in df.groupby("SCENARIO", sort=False):
        total_demand = float(sub["AVG_CALLS"].sum())
        total_coverage = float(sub["COVERAGE"].sum())
        coverage_ratio = total_coverage / total_demand if total_demand > 0 else np.nan
        total_shortfall = float(sub["SHORTFALL"].sum())
        peak_shortfall = float(sub.loc[sub["AVG_CALLS"] >= demand_threshold, "SHORTFALL"].sum())

        rows.append(
            {
                "scenario": scenario,
                "total_demand": total_demand,
                "total_coverage": total_coverage,
                "coverage_ratio": coverage_ratio,
                "total_shortfall": total_shortfall,
                "peak_shortfall": peak_shortfall,
                "hotspot_shortfall": float(sub.loc[sub["IS_HOTSPOT"] == 1, "SHORTFALL"].sum()),
                "avg_units_per_shift": float(sub.groupby("SHIFT")["UNITS"].sum().mean()),
            }
        )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["coverage_ratio", "total_coverage"], ascending=[False, False]).reset_index(drop=True)
    return summary


def plot_summary_bars(summary_df: pd.DataFrame, outdir: Path) -> None:
    """Save 2 compact comparison charts."""
    outdir.mkdir(parents=True, exist_ok=True)

    # coverage ratio
    plt.figure(figsize=(9, 5))
    plt.bar(summary_df["scenario"], summary_df["coverage_ratio"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Coverage ratio")
    plt.title("Scenario comparison: coverage ratio")
    plt.tight_layout()
    plt.savefig(outdir / "scenario_coverage_ratio.png", dpi=180)
    plt.show()

    # total shortfall
    plt.figure(figsize=(9, 5))
    plt.bar(summary_df["scenario"], summary_df["total_shortfall"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Total shortfall")
    plt.title("Scenario comparison: total shortfall")
    plt.tight_layout()
    plt.savefig(outdir / "scenario_total_shortfall.png", dpi=180)
    plt.show()


def plot_top_beats_allocation(
    alloc_df: pd.DataFrame,
    outdir: Path,
    shift: str = "Day",
    top_n: int = 10,
) -> None:
    """Save a grouped bar chart comparing allocation across scenarios for top-demand beats."""
    sub = alloc_df[alloc_df["SHIFT"] == shift].copy()
    if sub.empty:
        return

    top_beats = (
        sub.groupby("BEAT")["AVG_CALLS"].mean().sort_values(ascending=False).head(top_n).index.tolist()
    )
    wide = (
        sub[sub["BEAT"].isin(top_beats)]
        .pivot_table(index="BEAT", columns="SCENARIO", values="UNITS", aggfunc="first")
        .fillna(0)
    )
    wide = wide.loc[top_beats]

    x = np.arange(len(wide.index))
    n_scenarios = max(1, len(wide.columns))
    width = 0.8 / n_scenarios

    plt.figure(figsize=(12, 5))
    for i, col in enumerate(wide.columns):
        plt.bar(x + i * width, wide[col].values, width=width, label=col)

    plt.xticks(x + width * (n_scenarios - 1) / 2, wide.index.astype(str), rotation=0)
    plt.xlabel("BEAT")
    plt.ylabel("Allocated units")
    plt.title(f"Top-{top_n} beat allocation comparison ({shift} shift)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"allocation_top_beats_{shift.lower()}.png", dpi=180)
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 4 scenario analysis")
    parser.add_argument(
        "--base-allocation",
        default="./data/01-processed/step4_resource_deployment.csv",
        help="Path to existing step4_resource_deployment.csv",
    )
    parser.add_argument(
        "--hotspots",
        default="./data/01-processed/step3_hotspots_beats.csv",
        help="Path to step3_hotspots_beats.csv",
    )
    parser.add_argument(
        "--outdir",
        default="./data/01-processed/step4_scenarios",
        help="Directory for scenario-analysis outputs",
    )
    parser.add_argument(
        "--total-units",
        type=int,
        default=50,
        help="Total units available per shift",
    )
    parser.add_argument(
        "--mu-per-hour",
        type=float,
        default=3.0,
        help="Assumed calls handled per unit per hour",
    )
    parser.add_argument(
        "--risk-weights",
        type=float,
        nargs="+",
        default=[1.5, 2.0],
        help="Risk-aware weights to test",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many hotspot beats to protect",
    )
    parser.add_argument(
        "--min-units",
        type=int,
        default=2,
        help="Minimum units per hotspot beat per shift",
    )
    parser.add_argument(
        "--plot-shift",
        default="Day",
        choices=SHIFT_ORDER,
        help="Shift used in the top-beat allocation comparison plot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base_df = load_base_step4_table(args.base_allocation)
    hotspot_set = load_hotspot_set(args.hotspots, top_k=args.top_k)

    alloc_df = build_scenario_allocations(
        base_df=base_df,
        hotspot_set=hotspot_set,
        total_units=args.total_units,
        risk_weights=args.risk_weights,
        min_units=args.min_units,
    )
    alloc_df = add_capacity_metrics(alloc_df, mu_per_hour=args.mu_per_hour)
    summary_df = summarize_scenarios(alloc_df)

    alloc_path = outdir / "step4_scenario_allocations.csv"
    summary_path = outdir / "step4_scenario_summary.csv"
    alloc_df.to_csv(alloc_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    plot_summary_bars(summary_df, outdir)
    plot_top_beats_allocation(alloc_df, outdir, shift=args.plot_shift, top_n=10)

    print(f"[OK] Wrote allocations to: {alloc_path}")
    print(f"[OK] Wrote summary to: {summary_path}")
    print(f"[OK] Saved plots to: {outdir}")


if __name__ == "__main__":
    main()
