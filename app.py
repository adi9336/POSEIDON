import os
import sys
import threading
import time
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.orchestrator.main import PoseidonOrchestrator
from src.state.schemas import OrchestratorRequest


def initialize_orchestrator() -> None:
    """Initialize orchestrator in session state."""
    if "orchestrator" not in st.session_state:
        try:
            with st.spinner("Initializing orchestrator..."):
                st.session_state.orchestrator = PoseidonOrchestrator()
            st.success("Orchestrator initialized successfully")
        except Exception as exc:
            st.error(f"Failed to initialize orchestrator: {exc}")
            st.info("Please make sure OPENAI_API_KEY is set in your .env file")
            st.stop()


def render_diagnostics(result: Dict[str, Any], response_confidence: float) -> None:
    validation = (result or {}).get("validation", {}) or {}
    issues = validation.get("issues", []) or []
    metrics = validation.get("metrics", {}) or {}
    passed = bool(validation.get("passed", False))
    v_conf = float(validation.get("confidence", response_confidence))

    with st.container(border=True):
        st.markdown("### Diagnostics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Confidence", f"{v_conf:.2f}")
        col2.metric("Validation", "PASS" if passed else "WARN")
        col3.metric("Rows", int((result or {}).get("row_count", 0)))

        if issues:
            st.warning("Validation Issues")
            for issue in issues:
                st.markdown(f"- {issue}")
        else:
            st.success("No validation issues detected")

        if metrics:
            st.caption("Validation Metrics")
            st.json(metrics)


def render_visualizations(result: Dict[str, Any]) -> None:
    rows = (result or {}).get("data", []) or []
    if not rows:
        return
    try:
        df = pd.DataFrame(rows)
        st.markdown("### Visualizations")
        plot_df = df.copy()
        if "time" in plot_df.columns:
            plot_df["time"] = pd.to_datetime(plot_df["time"], errors="coerce")
        numeric_cols = [c for c in ["temp", "psal", "nitrate", "pres"] if c in plot_df.columns]

        tab_trend, tab_depth, tab_geo, tab_corr, tab_insights = st.tabs(
            ["Trends", "Depth Profile", "Geo View", "Correlations", "Insights"]
        )

        with tab_trend:
            if "time" in plot_df.columns and numeric_cols:
                trend_df = plot_df.dropna(subset=["time"]).copy()
                if not trend_df.empty:
                    long_df = trend_df[["time"] + numeric_cols].melt(
                        id_vars=["time"], var_name="variable", value_name="value"
                    ).dropna()
                    fig = px.line(
                        long_df,
                        x="time",
                        y="value",
                        color="variable",
                        markers=True,
                        title="Time-Series Trends",
                        template="plotly_dark",
                    )
                    fig.update_layout(height=380, legend_title_text="Variable")
                    st.plotly_chart(fig, use_container_width=True, key="trend_plot")
                else:
                    st.info("No valid timestamps available for trend view.")
            else:
                st.info("Trend view needs `time` + numeric variables.")

        with tab_depth:
            if "pres" in plot_df.columns:
                y_var = st.selectbox(
                    "Choose variable vs depth",
                    options=[c for c in ["temp", "psal", "nitrate"] if c in plot_df.columns],
                    key="depth_var_select",
                )
                depth_df = plot_df.dropna(subset=["pres", y_var]).copy()
                if not depth_df.empty:
                    hover_cols = [c for c in ["latitude", "longitude", "time"] if c in depth_df.columns]
                    fig = px.scatter(
                        depth_df,
                        x=y_var,
                        y="pres",
                        color=y_var,
                        title=f"{y_var.upper()} vs Depth",
                        hover_data=hover_cols,
                        template="plotly_dark",
                    )
                    fig.update_yaxes(autorange="reversed", title="Pressure/Depth")
                    fig.update_layout(height=380)
                    st.plotly_chart(fig, use_container_width=True, key="depth_plot")
                else:
                    st.info("Not enough depth data for this variable.")
            else:
                st.info("Depth profile requires `pres` column.")

        with tab_geo:
            if "latitude" in plot_df.columns and "longitude" in plot_df.columns:
                geo_df = plot_df.dropna(subset=["latitude", "longitude"]).copy()
                if not geo_df.empty:
                    geo_color = "temp" if "temp" in geo_df.columns else None
                    hover_cols = [c for c in ["temp", "psal", "nitrate", "pres", "time"] if c in geo_df.columns]
                    fig = px.scatter_geo(
                        geo_df,
                        lat="latitude",
                        lon="longitude",
                        color=geo_color,
                        hover_data=hover_cols,
                        title="Geospatial Distribution",
                        projection="natural earth",
                        template="plotly_dark",
                    )
                    fig.update_layout(height=420, margin=dict(l=0, r=0, t=45, b=0))
                    st.plotly_chart(fig, use_container_width=True, key="geo_plot")
                else:
                    st.info("No geo points available.")
            else:
                st.info("Geo view requires `latitude` and `longitude`.")

        with tab_corr:
            corr_cols = [c for c in ["temp", "psal", "nitrate", "pres"] if c in plot_df.columns]
            corr_df = plot_df[corr_cols].dropna() if corr_cols else pd.DataFrame()
            if len(corr_cols) >= 2 and not corr_df.empty:
                corr_matrix = corr_df.corr(numeric_only=True)
                fig = go.Figure(
                    data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns.tolist(),
                        y=corr_matrix.index.tolist(),
                        zmin=-1,
                        zmax=1,
                        colorscale="RdBu",
                        colorbar={"title": "corr"},
                        text=[[f"{v:.2f}" for v in row] for row in corr_matrix.values],
                        texttemplate="%{text}",
                    )
                )
                fig.update_layout(
                    title="Correlation Heatmap",
                    template="plotly_dark",
                    height=380,
                )
                st.plotly_chart(fig, use_container_width=True, key="corr_plot")
            else:
                st.info("Need at least two numeric variables with data for correlation view.")

        with tab_insights:
            variable_insights = (result or {}).get("variable_insights", {}) or {}
            if variable_insights:
                st.caption("Parallel Variable Insights")
                st.json(variable_insights)
            else:
                st.info("No variable insights available.")
    except Exception as exc:
        st.caption(f"Visualization skipped: {exc}")


def _progress_from_events(events: list[Dict[str, Any]]) -> float:
    if not events:
        return 0.05
    progress = 0.05
    for event in events:
        message = str(event.get("message", "")).lower()
        event_type = str(event.get("event_type", "")).lower()
        if event_type == "started":
            progress = max(progress, 0.1)
        if "understanding" in message:
            progress = max(progress, 0.25)
        if "retrieval" in message or "fetch" in message:
            progress = max(progress, 0.5)
        if "analysis" in message:
            progress = max(progress, 0.75)
        if "validation" in message:
            progress = max(progress, 0.9)
        if event_type in {"completed", "failed"}:
            progress = 1.0
    return min(progress, 1.0)


def execute_with_live_progress(
    orchestrator: PoseidonOrchestrator, req: OrchestratorRequest
) -> tuple[Any, list[Dict[str, Any]], Any]:
    result_box: Dict[str, Any] = {"response": None, "error": None}
    collected_events: list[Dict[str, Any]] = []

    def _worker():
        try:
            result_box["response"] = orchestrator.execute(req)
        except Exception as exc:  # pragma: no cover - runtime safety
            result_box["error"] = exc

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    events_placeholder = st.empty()

    status_placeholder.info("Started orchestration...")
    progress_placeholder.progress(0.05)

    while worker.is_alive():
        batch = orchestrator.pop_events(req.conversation_id)
        if batch:
            for e in batch:
                item = {"event_type": str(e.event_type), "message": e.message}
                collected_events.append(item)
            latest = collected_events[-1]["message"]
            status_placeholder.info(latest)
            progress_placeholder.progress(_progress_from_events(collected_events))
            events_markdown = "\n".join(
                [f"- `{ev['event_type']}`: {ev['message']}" for ev in collected_events[-8:]]
            )
            events_placeholder.markdown(f"**Live Events**\n{events_markdown}")
        time.sleep(0.2)

    # Drain remaining events after worker completion.
    batch = orchestrator.pop_events(req.conversation_id)
    for e in batch:
        collected_events.append({"event_type": str(e.event_type), "message": e.message})

    progress_placeholder.progress(_progress_from_events(collected_events))
    if collected_events:
        events_markdown = "\n".join(
            [f"- `{ev['event_type']}`: {ev['message']}" for ev in collected_events]
        )
        events_placeholder.markdown(f"**Live Events**\n{events_markdown}")

    return result_box.get("response"), collected_events, result_box.get("error")


def main() -> None:
    st.set_page_config(
        page_title="Oceanographic Data Analysis Agent",
        page_icon="??",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    initialize_orchestrator()

    if "history" not in st.session_state:
        st.session_state.history = []
    if "message_count" not in st.session_state:
        st.session_state.message_count = 0

    with st.sidebar:
        st.title("Controls")

        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.history = []
            st.session_state.message_count = 0
            st.rerun()

        st.markdown("---")
        st.markdown("### Statistics")
        st.metric("Total Messages", st.session_state.message_count)
        st.metric(
            "Conversations",
            len([m for m in st.session_state.history if m.get("role") == "user"]),
        )

        st.markdown("---")
        with st.expander("Example Queries"):
            st.markdown(
                """
- What's the temperature near Mumbai in January 2025?
- Show salinity trend at 500m depth near Goa for last 30 days
- Compare temperature at 200m near Chennai in March 2024 vs April 2024
"""
            )

    st.title("Oceanographic Data Analysis Agent")
    st.markdown("Ask questions and review validation diagnostics.")

    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("role") == "assistant" and message.get("diagnostics"):
                diag = message["diagnostics"]
                result = diag.get("result", {})
                render_diagnostics(result, float(diag.get("confidence", 0.0)))
                render_visualizations(result)
                events = diag.get("events", [])
                if events:
                    with st.expander("Execution Events"):
                        for event in events:
                            st.markdown(f"- `{event.get('event_type')}`: {event.get('message')}")

    if prompt := st.chat_input("Ask me about oceanographic data..."):
        st.session_state.history.append({"role": "user", "content": prompt})
        st.session_state.message_count += 1

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                req = OrchestratorRequest(
                    query=prompt,
                    mode="multi",
                )
                response, events, run_error = execute_with_live_progress(
                    st.session_state.orchestrator, req
                )
                if run_error is not None:
                    raise run_error
                if response is None:
                    raise RuntimeError("No response returned from orchestrator")

                result = response.result or {}
                summary = (
                    result.get("summary")
                    or result.get("message")
                    or "No summary available."
                )

                st.markdown(summary)
                render_diagnostics(result, float(response.confidence))
                render_visualizations(result)

                st.session_state.history.append(
                    {
                        "role": "assistant",
                        "content": summary,
                        "diagnostics": {
                            "result": result,
                            "confidence": float(response.confidence),
                            "events": events,
                        },
                    }
                )
                st.session_state.message_count += 1
            except Exception as exc:
                error_msg = f"Error: {exc}"
                st.error(error_msg)
                st.session_state.history.append(
                    {"role": "assistant", "content": error_msg}
                )
                st.session_state.message_count += 1

        st.rerun()


if __name__ == "__main__":
    main()
