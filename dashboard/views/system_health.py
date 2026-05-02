"""
System Health view — data freshness, connection status, cache info.
"""

import streamlit as st
from datetime import datetime, timezone


def render(cache_age, supabase_ok: bool, last_update):
    st.header("System Health")

    st.subheader("Data Connection")
    if supabase_ok:
        st.success("Supabase: Connected")
    else:
        st.error("Supabase: Connection failed")

    st.subheader("Data Freshness")
    if last_update:
        st.info(f"Last data update: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.warning("Unable to determine last data update")

    st.subheader("Cache Status")
    if cache_age:
        age_hours = (datetime.now(timezone.utc) - cache_age.replace(tzinfo=timezone.utc) if cache_age.tzinfo is None else cache_age).total_seconds() / 3600
        st.info(f"Cache age: {age_hours:.1f} hours")
        if age_hours > 24:
            st.warning("Cache is older than 24 hours — consider refreshing")
    else:
        st.warning("No cache found — first run will compute backtest")

    if st.button("Refresh Data", type="primary"):
        st.session_state["dashboard_refresh"] = True
        st.rerun()
