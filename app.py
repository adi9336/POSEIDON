import os
import sys
import threading
import time
from typing import Any, Dict

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.paths import ensure_runtime_dirs
from src.orchestrator.main import PoseidonOrchestrator
from src.state.schemas import OrchestratorRequest
from src.supervisor.agent import SupervisorAgent
from src.supervisor.state import SupervisorConversationState, SupervisorPhase


# ──────────────────────────────────────────────
# Initialization
# ──────────────────────────────────────────────

def initialize() -> None:
    """Initialize orchestrator, supervisor, and session state."""
    ensure_runtime_dirs()

    if "orchestrator" not in st.session_state:
        try:
            with st.spinner("🔧 Initializing POSEIDON..."):
                st.session_state.orchestrator = PoseidonOrchestrator()
                st.session_state.supervisor = SupervisorAgent()
        except Exception as exc:
            st.error(f"Failed to initialize: {exc}")
            st.info("Please make sure OPENAI_API_KEY is set in your .env file")
            st.stop()

    if "supervisor_session" not in st.session_state:
        st.session_state.supervisor_session = SupervisorConversationState()
    if "history" not in st.session_state:
        st.session_state.history = []
    if "message_count" not in st.session_state:
        st.session_state.message_count = 0
    if "pending_action" not in st.session_state:
        st.session_state.pending_action = None  # clarification | map | approval

    # Start campaign system only when explicitly enabled.
    enable_campaigns = os.getenv("POSEIDON_ENABLE_CAMPAIGNS", "false").lower() == "true"
    if enable_campaigns and "campaign_manager" not in st.session_state:
        try:
            from src.campaigns.manager import CampaignManager
            st.session_state.campaign_manager = CampaignManager()
            st.session_state.campaign_manager.start()
        except Exception as exc:
            pass  # campaigns are optional, don't block the app


# ──────────────────────────────────────────────
# Rendering helpers
# ──────────────────────────────────────────────

def render_diagnostics(result: Dict[str, Any], response_confidence: float) -> None:
    validation = (result or {}).get("validation", {}) or {}
    issues = validation.get("issues", []) or []
    metrics = validation.get("metrics", {}) or {}
    passed = bool(validation.get("passed", False))
    v_conf = float(validation.get("confidence", response_confidence))

    with st.container(border=True):
        st.markdown("### 📊 Diagnostics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Confidence", f"{v_conf:.2f}")
        col2.metric("Validation", "✅ PASS" if passed else "⚠️ WARN")
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


def render_visualizations(result: Dict[str, Any], key_suffix: str = "") -> None:
    rows = (result or {}).get("data", []) or []
    if not rows:
        return
    try:
        df = pd.DataFrame(rows)
        st.markdown("### 📈 Visualizations")
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
                    fig = px.line(long_df, x="time", y="value", color="variable",
                                 markers=True, title="Time-Series Trends", template="plotly_dark")
                    fig.update_layout(height=380, legend_title_text="Variable")
                    st.plotly_chart(fig, key=f"trend_plot{key_suffix}")
                else:
                    st.info("No valid timestamps available for trend view.")
            else:
                st.info("Trend view needs `time` + numeric variables.")

        with tab_depth:
            if "pres" in plot_df.columns:
                y_var = st.selectbox(
                    "Choose variable vs depth",
                    options=[c for c in ["temp", "psal", "nitrate"] if c in plot_df.columns],
                    key=f"depth_var_select{key_suffix}",
                )
                depth_df = plot_df.dropna(subset=["pres", y_var]).copy()
                if not depth_df.empty:
                    hover_cols = [c for c in ["latitude", "longitude", "time"] if c in depth_df.columns]
                    fig = px.scatter(depth_df, x=y_var, y="pres", color=y_var,
                                    title=f"{y_var.upper()} vs Depth", hover_data=hover_cols,
                                    template="plotly_dark")
                    fig.update_yaxes(autorange="reversed", title="Pressure/Depth")
                    fig.update_layout(height=380)
                    st.plotly_chart(fig, key=f"depth_plot{key_suffix}")
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
                    fig = px.scatter_geo(geo_df, lat="latitude", lon="longitude",
                                        color=geo_color, hover_data=hover_cols,
                                        title="Geospatial Distribution",
                                        projection="natural earth", template="plotly_dark")
                    fig.update_layout(height=420, margin=dict(l=0, r=0, t=45, b=0))
                    st.plotly_chart(fig, key=f"geo_plot{key_suffix}")
                else:
                    st.info("No geo points available.")
            else:
                st.info("Geo view requires `latitude` and `longitude`.")

        with tab_corr:
            corr_cols = [c for c in ["temp", "psal", "nitrate", "pres"] if c in plot_df.columns]
            corr_df = plot_df[corr_cols].dropna() if corr_cols else pd.DataFrame()
            if len(corr_cols) >= 2 and not corr_df.empty:
                corr_matrix = corr_df.corr(numeric_only=True)
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values, x=corr_matrix.columns.tolist(),
                    y=corr_matrix.index.tolist(), zmin=-1, zmax=1,
                    colorscale="RdBu", colorbar={"title": "corr"},
                    text=[[f"{v:.2f}" for v in row] for row in corr_matrix.values],
                    texttemplate="%{text}",
                ))
                fig.update_layout(title="Correlation Heatmap", template="plotly_dark", height=380)
                st.plotly_chart(fig, key=f"corr_plot{key_suffix}")
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


# ──────────────────────────────────────────────
# Supervisor interactive renderers
# ──────────────────────────────────────────────

def render_clarification_dialog(response) -> None:
    """Render clarification questions as interactive selectboxes."""
    st.markdown("---")
    st.markdown("**🤔 I need a bit more information:**")

    answers = {}
    for q in response.clarification_questions:
        if q.options:
            choice = st.selectbox(
                q.text,
                options=q.options,
                index=q.options.index(q.default) if q.default and q.default in q.options else 0,
                key=f"clarify_{q.id}",
            )
            answers[q.id] = choice
        else:
            text_input = st.text_input(q.text, key=f"clarify_{q.id}")
            answers[q.id] = text_input

    if st.button("✅ Submit Answers", key="submit_clarifications", type="primary"):
        session = st.session_state.supervisor_session
        clarification_engine = st.session_state.supervisor.clarification_engine

        # Parse and merge answers
        for q_id, answer in answers.items():
            parsed = clarification_engine.parse_clarification_answer(q_id, answer)
            session.merge_clarifications(parsed)

        st.session_state.pending_action = None
        # Re-process with supervisor
        supervisor = st.session_state.supervisor
        next_response = supervisor.process(session, "User provided clarification answers")

        _handle_supervisor_response(next_response)
        st.rerun()


def render_map_confirmation(response) -> None:
    """Render a Folium map for location confirmation with drawing tools."""
    from folium.plugins import Draw
    import math

    map_data = response.map_data or {}
    center_lat = map_data.get("center_lat", 20.0)
    center_lon = map_data.get("center_lon", 73.0)
    radius_km = map_data.get("radius_km", 100.0)
    float_count = map_data.get("argo_float_count", 0)

    st.markdown("---")
    st.markdown(f"**🗺️ Select Your Search Area** — {float_count} ARGO floats detected nearby")
    st.caption("Use the **draw tools** on the map to draw a circle or rectangle for your search area, "
               "or use the slider below to set a radius around the center point.")

    # Create Folium map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="CartoDB dark_matter")

    # Add drawing controls for circle and rectangle
    Draw(
        export=False,
        draw_options={
            "polyline": False,
            "polygon": False,
            "marker": True,
            "circle": True,
            "rectangle": True,
            "circlemarker": False,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(m)

    # Show suggested center + radius
    folium.Marker(
        [center_lat, center_lon],
        popup=f"Suggested center: {center_lat:.2f}°, {center_lon:.2f}°",
        icon=folium.Icon(color="red", icon="crosshairs", prefix="fa"),
    ).add_to(m)
    folium.Circle(
        [center_lat, center_lon],
        radius=radius_km * 1000,
        color="#00b4d8",
        fill=True,
        fill_opacity=0.12,
        popup=f"Suggested radius: {radius_km}km",
        dash_array="10",
    ).add_to(m)

    # Add nearby float markers
    map_interface = st.session_state.supervisor.map_interface
    floats = map_interface.get_nearby_floats(center_lat, center_lon, radius_km)
    for f in floats:
        folium.CircleMarker(
            [f["latitude"], f["longitude"]],
            radius=4,
            color="#00ff88",
            fill=True,
            popup=f"Float {f['platform_number']}\nLast seen: {f.get('last_seen', 'N/A')}",
        ).add_to(m)

    map_output = st_folium(m, width=700, height=450, key="location_map")

    # Extract drawn shapes from map output
    drawn_lat, drawn_lon, drawn_radius_km = center_lat, center_lon, radius_km
    has_drawing = False

    if map_output and map_output.get("all_drawings"):
        drawings = map_output["all_drawings"]
        if drawings:
            last_drawing = drawings[-1]
            geom = last_drawing.get("geometry", {})
            props = last_drawing.get("properties", {})

            if geom.get("type") == "Point" and "radius" in props:
                # User drew a circle
                coords = geom.get("coordinates", [])
                if len(coords) >= 2:
                    drawn_lon, drawn_lat = coords[0], coords[1]
                    drawn_radius_km = props["radius"] / 1000.0  # meters → km
                    has_drawing = True
                    st.success(f"✅ Circle drawn: center ({drawn_lat:.2f}°, {drawn_lon:.2f}°), radius {drawn_radius_km:.0f}km")

            elif geom.get("type") == "Polygon":
                # User drew a rectangle — compute center and approximate radius
                coords = geom.get("coordinates", [[]])[0]
                if len(coords) >= 4:
                    lats = [c[1] for c in coords]
                    lons = [c[0] for c in coords]
                    drawn_lat = sum(lats) / len(lats)
                    drawn_lon = sum(lons) / len(lons)
                    # Approximate radius from bounding box
                    dlat = (max(lats) - min(lats)) * 111.0 / 2.0
                    dlon = (max(lons) - min(lons)) * 111.0 * math.cos(math.radians(drawn_lat)) / 2.0
                    drawn_radius_km = max(dlat, dlon)
                    has_drawing = True
                    st.success(f"✅ Rectangle drawn: center ({drawn_lat:.2f}°, {drawn_lon:.2f}°), ~{drawn_radius_km:.0f}km radius")

    # Fallback: check for simple click
    if not has_drawing and map_output and map_output.get("last_clicked"):
        drawn_lat = map_output["last_clicked"]["lat"]
        drawn_lon = map_output["last_clicked"]["lng"]
        st.info(f"📍 You clicked: ({drawn_lat:.2f}°, {drawn_lon:.2f}°) — use slider for radius")

    col1, col2, col3 = st.columns(3)
    if not has_drawing:
        new_radius = col1.slider("Search radius (km)", 50, 500, int(radius_km), step=50, key="map_radius")
    else:
        new_radius = drawn_radius_km
        col1.metric("Drawn radius", f"{drawn_radius_km:.0f} km")

    with col2:
        if st.button("✅ Confirm Area", key="confirm_location", type="primary"):
            session = st.session_state.supervisor_session
            session.confirmed_coordinates = {
                "lat": drawn_lat,
                "lon": drawn_lon,
                "radius_km": float(new_radius),
            }
            st.session_state.pending_action = None

            supervisor = st.session_state.supervisor
            next_response = supervisor.process(session, "Location confirmed")
            _handle_supervisor_response(next_response)
            st.rerun()

    with col3:
        if st.button("❌ Cancel", key="cancel_map"):
            session = st.session_state.supervisor_session
            session.advance_phase(SupervisorPhase.GREETING)
            st.session_state.pending_action = None
            st.rerun()


def render_approval_card(response) -> None:
    """Render workflow approval card."""
    plan = response.workflow_plan
    if not plan:
        return

    st.markdown("---")
    with st.container(border=True):
        st.markdown(f"### 📋 Execution Plan: {plan.name}")
        for step in plan.steps:
            st.markdown(f"  {step}")
        st.markdown("")

        col1, col2, col3 = st.columns(3)
        col1.metric("⏱️ Est. Time", plan.estimated_time)
        col2.metric("💰 Est. Cost", plan.estimated_cost)
        col3.metric("📊 Data", plan.data_volume)

        st.markdown("")
        btn_col1, btn_col2, btn_col3 = st.columns(3)

        with btn_col1:
            if st.button("✅ Proceed", key="approve_workflow", type="primary"):
                session = st.session_state.supervisor_session
                st.session_state.pending_action = None
                supervisor = st.session_state.supervisor
                next_response = supervisor.process(session, "yes")
                _handle_supervisor_response(next_response)
                st.rerun()

        with btn_col2:
            if st.button("✏️ Modify", key="modify_workflow"):
                session = st.session_state.supervisor_session
                session.advance_phase(SupervisorPhase.CLARIFYING)
                st.session_state.pending_action = None
                st.rerun()

        with btn_col3:
            if st.button("❌ Cancel", key="cancel_workflow"):
                session = st.session_state.supervisor_session
                session.advance_phase(SupervisorPhase.GREETING)
                st.session_state.pending_action = None
                st.session_state.history.append({
                    "role": "assistant",
                    "content": "No problem! Feel free to ask another question."
                })
                st.rerun()


# ──────────────────────────────────────────────
# Supervisor response handler
# ──────────────────────────────────────────────

def _handle_supervisor_response(response) -> None:
    """Process a SupervisorResponse and update session state."""
    session = st.session_state.supervisor_session

    # Add supervisor message to chat history
    if response.message:
        st.session_state.history.append({
            "role": "assistant",
            "content": response.message,
        })
        st.session_state.message_count += 1

    # Set pending action for the UI to render
    if response.response_type == "clarification_needed":
        st.session_state.pending_action = ("clarification", response)
    elif response.response_type == "map_confirmation_needed":
        st.session_state.pending_action = ("map", response)
    elif response.response_type == "approval_needed":
        st.session_state.pending_action = ("approval", response)
    elif response.response_type == "ready_to_execute":
        st.session_state.pending_action = ("execute", response)


# ──────────────────────────────────────────────
# Execution with live progress
# ──────────────────────────────────────────────

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
        except Exception as exc:
            result_box["error"] = exc

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    events_placeholder = st.empty()

    status_placeholder.info("⚙️ Executing workflow...")
    progress_placeholder.progress(0.05)

    while worker.is_alive():
        batch = orchestrator.pop_events(req.conversation_id)
        if batch:
            for e in batch:
                collected_events.append({"event_type": str(e.event_type), "message": e.message})
            latest = collected_events[-1]["message"]
            status_placeholder.info(f"⚙️ {latest}")
            progress_placeholder.progress(_progress_from_events(collected_events))
            events_md = "\n".join(
                [f"- `{ev['event_type']}`: {ev['message']}" for ev in collected_events[-8:]]
            )
            events_placeholder.markdown(f"**Live Events**\n{events_md}")
        time.sleep(0.2)

    batch = orchestrator.pop_events(req.conversation_id)
    for e in batch:
        collected_events.append({"event_type": str(e.event_type), "message": e.message})
    progress_placeholder.progress(_progress_from_events(collected_events))

    return result_box.get("response"), collected_events, result_box.get("error")


# ──────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="POSEIDON — Oceanographic Data Analysis",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    initialize()

    # ── Sidebar ──
    with st.sidebar:
        st.title("🌊 POSEIDON")
        st.caption("Multi-Agent Oceanographic Analysis")
        st.markdown("---")

        if st.button("🔄 New Conversation", use_container_width=True):
            st.session_state.history = []
            st.session_state.message_count = 0
            st.session_state.supervisor_session = SupervisorConversationState()
            st.session_state.pending_action = None
            st.rerun()

        st.markdown("### Statistics")
        st.metric("Messages", st.session_state.message_count)
        phase = st.session_state.supervisor_session.phase.value
        st.metric("Supervisor Phase", phase.replace("_", " ").title())

        st.markdown("---")
        with st.expander("💡 Example Queries"):
            st.markdown(
                """
- What's the temperature near Mumbai?
- Show salinity trend at 500m depth near Hawaii
- Compare ocean conditions in the Arabian Sea
- Show me ocean data (triggers clarification)
"""
            )

        # Show session memory
        session_mem = st.session_state.supervisor_session.session_memory
        if session_mem:
            st.markdown("---")
            with st.expander("🧠 Session Memory", expanded=False):
                for i, entry in enumerate(session_mem, 1):
                    q = entry.get("query", "")[:60]
                    ts = entry.get("timestamp", "")[:19]
                    st.markdown(f"**{i}.** {q}...")
                    st.caption(ts)

    # ── Main area ──
    st.title("🌊 POSEIDON — Oceanographic Data Analysis")
    st.markdown("*Powered by a Supervisor Agent that guides your analysis*")

    # Render chat history
    for idx, message in enumerate(st.session_state.history):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("role") == "assistant" and message.get("diagnostics"):
                diag = message["diagnostics"]
                result = diag.get("result", {})
                render_diagnostics(result, float(diag.get("confidence", 0.0)))
                render_visualizations(result, key_suffix=f"_hist_{idx}")

    # Render pending interactive component (if any)
    pending = st.session_state.pending_action
    if pending:
        action_type, response = pending
        if action_type == "clarification":
            render_clarification_dialog(response)
        elif action_type == "map":
            render_map_confirmation(response)
        elif action_type == "approval":
            render_approval_card(response)
        elif action_type == "execute":
            _run_execution()

    # ── Chat input ──
    if prompt := st.chat_input("Ask me about oceanographic data..."):
        st.session_state.history.append({"role": "user", "content": prompt})
        st.session_state.message_count += 1

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                session = st.session_state.supervisor_session
                supervisor = st.session_state.supervisor

                # Route through Supervisor
                response = supervisor.process(session, prompt)

                # Display message
                if response.message:
                    st.markdown(response.message)

                # Handle response type
                _handle_supervisor_response(response)

            except Exception as exc:
                error_msg = f"Error: {exc}"
                st.error(error_msg)
                st.session_state.history.append({"role": "assistant", "content": error_msg})
                st.session_state.message_count += 1

        st.rerun()


def _run_execution() -> None:
    """Execute the workflow via the orchestrator (triggered after supervisor approval)."""
    session = st.session_state.supervisor_session
    query = session.user_query

    # Build intent overrides from supervisor-collected data
    intent = session.confirmed_intent or {}
    intent.update(session.clarifications_collected)
    if session.confirmed_coordinates:
        intent["lat"] = session.confirmed_coordinates.get("lat")
        intent["lon"] = session.confirmed_coordinates.get("lon")
        intent["radius_km"] = session.confirmed_coordinates.get("radius_km", 100.0)

    with st.chat_message("assistant"):
        try:
            req = OrchestratorRequest(
                query=query,
                mode="multi",
                intent_override=intent if intent else None,
            )
            response, events, run_error = execute_with_live_progress(
                st.session_state.orchestrator, req
            )
            if run_error is not None:
                raise run_error
            if response is None:
                raise RuntimeError("No response returned from orchestrator")

            result = response.result or {}
            summary = result.get("summary") or result.get("message") or "Analysis complete."

            ec = st.session_state.supervisor_session.execution_counter
            st.markdown(f"✅ **Analysis Complete!**\n\n{summary}")
            render_diagnostics(result, float(response.confidence))
            render_visualizations(result, key_suffix=f"_exec_{ec}")

            # Suggestions
            if session.confirmed_intent:
                var = session.confirmed_intent.get("variable", "")
                loc = session.confirmed_intent.get("location", "")
                st.markdown("---")
                st.markdown("**💡 What else can I help with?**")
                suggestions = []
                if var == "temp":
                    suggestions.append(f"• Show salinity data at the same location")
                if loc:
                    suggestions.append(f"• Show temperature trends over the past 30 days near {loc}")
                suggestions.append("• Explore a different region")
                for s in suggestions:
                    st.markdown(s)

            # Save to session memory before resetting
            session.save_to_memory(result_summary=summary)
            session.advance_phase(SupervisorPhase.COMPLETED)
            st.session_state.pending_action = None

            st.session_state.history.append({
                "role": "assistant",
                "content": summary,
                "diagnostics": {
                    "result": result,
                    "confidence": float(response.confidence),
                    "events": events,
                },
            })
            st.session_state.message_count += 1

        except Exception as exc:
            error_msg = f"❌ Execution error: {exc}"
            st.error(error_msg)
            session.advance_phase(SupervisorPhase.ERROR)
            st.session_state.pending_action = None

            # Offer recovery suggestions
            st.markdown("**🔄 Would you like to:**")
            st.markdown("- Try with a broader search area")
            st.markdown("- Adjust your query parameters")
            st.markdown("- Start a new query")

            st.session_state.history.append({"role": "assistant", "content": error_msg})
            st.session_state.message_count += 1


if __name__ == "__main__":
    main()
