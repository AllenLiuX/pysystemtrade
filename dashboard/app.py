"""
Risk Parity Dashboard — Streamlit App

Sidebar navigation with views for:
- Overview (KPIs, equity curve, comparison)
- Weights (rolling weights, positions)
- Volatility (portfolio + instrument vol)
- Instruments (per-instrument performance)
- System Health (connection, cache status)
"""

import streamlit as st
import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Risk Parity Dashboard",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def get_supabase_status():
    """Check Supabase connection once per session."""
    from dashboard.data.supabase import check_connection, get_last_update
    ok = check_connection()
    last_update = get_last_update() if ok else None
    return ok, last_update


def main():
    st.title("Risk Parity Dashboard")

    # Initialize refresh state
    if "dashboard_refresh" not in st.session_state:
        st.session_state["dashboard_refresh"] = False

    # Get metrics (with refresh support)
    refresh = st.session_state.pop("dashboard_refresh", False)
    with st.spinner("Loading metrics..."):
        from dashboard.data.metrics import get_metrics, get_cache_age
        from dashboard.data.supabase import check_connection, get_last_update

        metrics = get_metrics(refresh=refresh)
        cache_age = get_cache_age()

    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        view = st.radio(
            "Go to",
            ["Overview", "Weights", "Volatility", "Instruments", "System Health"],
            index=0,
        )

        st.divider()
        st.header("System Status")
        supabase_ok, last_update = get_supabase_status()
        if supabase_ok:
            st.caption("🟢 Supabase connected")
        else:
            st.caption("🔴 Supabase disconnected")

        if cache_age:
            age_hours = (datetime.now(timezone.utc) - cache_age).total_seconds() / 3600
            st.caption(f"Cache: {age_hours:.1f}h old")

        if st.button("Refresh", type="primary", use_container_width=True):
            st.session_state["dashboard_refresh"] = True
            st.rerun()

    # Route to view
    if view == "Overview":
        from dashboard.views.overview import render as render_overview
        render_overview(metrics)
    elif view == "Weights":
        from dashboard.views.weights import render as render_weights
        render_weights(metrics)
    elif view == "Volatility":
        from dashboard.views.volatility import render as render_volatility
        render_volatility(metrics)
    elif view == "Instruments":
        from dashboard.views.instruments import render as render_instruments
        render_instruments(metrics)
    elif view == "System Health":
        from dashboard.views.system_health import render as render_health
        render_health(cache_age, supabase_ok, last_update)


if __name__ == "__main__":
    main()
