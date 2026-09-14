"""Interactive Streamlit dashboard for the Boat Wash AI System.

Run with:
    streamlit run src/dashboard/app.py

Shows, live, the pieces from the concept design's "Smart monitoring" slide:
targeted spray (CV grid overlay on the hull), the water-quality gate, pump
health alerts, and closed-loop recycling stats.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.orchestrator.decision_engine import Orchestrator
from src.utils.config import DEFAULT_CONFIG
from src.vision.synthetic_data import generate_hull_image

st.set_page_config(page_title="Building R13 — Boat Wash AI System", layout="wide")


@st.cache_resource
def get_orchestrator(seed: int) -> Orchestrator:
    return Orchestrator(DEFAULT_CONFIG, seed=seed)


def init_state() -> None:
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = get_orchestrator(seed=7)
        st.session_state.rng = np.random.default_rng(7)
        st.session_state.tick = 0


def hull_overlay_figure(image: np.ndarray, grid: np.ndarray, threshold: float):
    rows, cols = grid.shape
    h, w = image.shape[:2]
    cell_h, cell_w = h / rows, w / cols

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(image)
    for r in range(rows):
        for c in range(cols):
            score = grid[r, c]
            color = "red" if score >= threshold else ("orange" if score >= 0.12 else "none")
            if color != "none":
                ax.add_patch(
                    plt.Rectangle((c * cell_w, r * cell_h), cell_w, cell_h,
                                  fill=True, alpha=0.30, color=color, linewidth=0)
                )
            ax.add_patch(
                plt.Rectangle((c * cell_w, r * cell_h), cell_w, cell_h,
                              fill=False, edgecolor="white", linewidth=0.5)
            )
    ax.set_title("Targeted spray plan (red = full pressure, orange = light rinse)", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    return fig


def main() -> None:
    init_state()
    orchestrator: Orchestrator = st.session_state.orchestrator

    st.title("🛥️ Building R13 — Boat Wash AI System")
    st.caption(
        "Simulated Monitor & ML layer: computer-vision stain detection, sensor fleet, "
        "ML pump-health monitoring and the rule-based orchestrator, driving a closed-loop "
        "water recycling system."
    )

    with st.sidebar:
        st.header("Controls")
        st.write(f"Vision backend: **{orchestrator.detector.backend}**")
        n_cycles = st.slider("Cycles to run", 1, 20, 1)
        inject_contamination = st.checkbox("Inject contamination this batch")
        inject_pump_fault = st.checkbox("Degrade pump_A this batch")
        run_clicked = st.button("▶ Run wash cycle(s)", type="primary")
        reset_clicked = st.button("↺ Reset simulation")

        if reset_clicked:
            for k in ("orchestrator", "rng", "tick"):
                st.session_state.pop(k, None)
            st.rerun()

    if run_clicked:
        if inject_contamination:
            orchestrator.sensors.inject_contamination(st.session_state.tick + 1)
        if inject_pump_fault:
            orchestrator.sensors.degrade_pump("pump_A", st.session_state.tick + 1)

        for _ in range(n_cycles):
            st.session_state.tick += 1
            image = generate_hull_image(size=DEFAULT_CONFIG.grid.image_size, rng=st.session_state.rng).image
            orchestrator.run_cycle(st.session_state.tick, image)
        st.session_state.last_image = image

    if not orchestrator.log:
        st.info("Click **Run wash cycle(s)** in the sidebar to start the simulation.")
        return

    latest = orchestrator.log[-1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Hull dirtiness (latest)", f"{latest.vision.coverage_pct:.1f}%")
    col2.metric("Water recycled", f"{orchestrator.recycling_loop.recycled_rate_pct:.1f}%")
    col3.metric("Water saved vs. blanket wash", f"{orchestrator.recycling_loop.water_saved_vs_blanket_pct:.1f}%")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Computer vision — targeted spray")
        fig = hull_overlay_figure(st.session_state.last_image, latest.vision.grid, DEFAULT_CONFIG.grid.dirty_threshold)
        st.pyplot(fig)
        st.caption(f"Backend: `{latest.vision.backend}` · {len(latest.vision.dirty_cells)} dirty cells flagged")

    with right:
        st.subheader("Water quality gate")
        gate_ok = latest.water_gate.safe_to_reuse
        st.markdown(f"**Status:** {'🟢 SAFE TO REUSE' if gate_ok else '🔴 BLOCKED — held for disposal'}")
        wq = latest.water_reading
        st.table(pd.DataFrame([{
            "pH": wq.ph, "turbidity (NTU)": wq.turbidity_ntu,
            "chlorine (ppm)": wq.chlorine_ppm, "TDS (ppm)": wq.tds_ppm,
        }]))
        if not gate_ok:
            for reason in latest.water_gate.reasons:
                st.warning(reason)

        st.subheader("Pump health")
        for reading, alert in zip(latest.pump_readings, latest.pump_alerts):
            badge = "🔴 ANOMALY" if alert.is_anomaly else "🟢 normal"
            st.markdown(
                f"**{reading.pump_id}** — {badge}  \n"
                f"pressure={reading.pressure_bar} bar · flow={reading.flow_lpm} L/min · "
                f"energy={reading.energy_kw} kW · score={alert.anomaly_score} (`{alert.method}`)"
            )

    st.subheader("Trend across the run")
    df = pd.DataFrame({
        "tick": [r.tick for r in orchestrator.log],
        "dirtiness_pct": [r.vision.coverage_pct for r in orchestrator.log],
        "tank_level_l": [r.recycling.tank_level_l for r in orchestrator.log],
        "pump_A_pressure": [r.pump_readings[0].pressure_bar for r in orchestrator.log],
        "pump_B_pressure": [r.pump_readings[1].pressure_bar for r in orchestrator.log],
    }).set_index("tick")
    st.line_chart(df)

    st.subheader("Decision log")
    log_rows = []
    for r in orchestrator.log[::-1]:
        log_rows.append({
            "tick": r.tick, "dirt %": r.vision.coverage_pct, "water used (L)": r.recycling.volume_used_l,
            "reused": r.recycling.reused, "alerts": "; ".join(r.alerts) if r.alerts else "-",
        })
    st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
