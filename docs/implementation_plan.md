# POSEIDON Supervisor Agent Architecture Redesign

A production-grade Supervisor Agent that sits between the user and the existing 5-agent orchestrator, providing intelligent clarification, geographic confirmation, workflow approval, and progress communication.

## User Review Required

> [!IMPORTANT]
> **Scope Decision**: This plan implements the Supervisor Agent **within the existing Streamlit + FastAPI stack** using **OpenAI GPT-4o** (already configured in [.env](file:///c:/Users/adity/POSEIDON/.env)). It does NOT add Claude/Anthropic models since those API keys aren't set up. If you want to use Claude instead, please let me know and I'll adjust.

> [!WARNING]
> **Map library**: I'll use **Folium** (renders natively in Streamlit via `st_folium`) for interactive maps. This avoids JavaScript complexity. Mapbox/Leaflet.js would need a separate frontend app. Let me know if you prefer a different approach.

> [!IMPORTANT]
> **Phased delivery**: This is a large redesign (~15 new/modified files). I propose implementing **Phase 1** (core Supervisor + Streamlit UI) first and verifying it works before proceeding to Phases 2-3 (API v2, testing). Confirm if this is acceptable.

---

## Proposed Changes

### Supervisor Core Package

New package `src/supervisor/` with the Supervisor Agent logic, conversation state machine, and sub-engines.

#### [NEW] [\_\_init\_\_.py](file:///c:/Users/adity/POSEIDON/src/supervisor/__init__.py)
Empty init to make `src/supervisor/` a package.

#### [NEW] [state.py](file:///c:/Users/adity/POSEIDON/src/supervisor/state.py)
`SupervisorConversationState` — Pydantic model tracking multi-turn conversation. Fields:
- `session_id`, `phase` (enum: `greeting | clarifying | confirming_location | awaiting_approval | executing | completed | error`)
- `user_query`, `clarifications_collected: Dict`, `confirmed_coordinates: Optional[dict]`
- `workflow_plan: Optional[dict]`, `execution_result: Optional[dict]`
- `chat_history: List[dict]` — full user/supervisor message log
- Helper methods: `advance_phase()`, [add_message()](file:///c:/Users/adity/POSEIDON/src/state/models.py#89-102), `is_ready_to_execute()`

#### [NEW] [agent.py](file:///c:/Users/adity/POSEIDON/src/supervisor/agent.py)
`SupervisorAgent` — the core reasoning engine (~200 lines). Uses OpenAI function-calling to:
1. **Analyze** user query → classify intent (direct_retrieval / exploratory / analytical / ambiguous)
2. **Identify missing params** (location, depth, time_range, variable)
3. **Generate clarification questions** as structured JSON (with `options` and `defaults`)
4. **Decide** if map confirmation is needed (when lat/lon are from geocoding, not explicit)
5. **Build workflow plan** with estimated steps, time, data volume
6. **Produce a supervisor response** message for the user

Key method: `async process(session: SupervisorConversationState, user_input: str) -> SupervisorResponse` — returns one of:
- `clarification_needed` with questions list
- `map_confirmation_needed` with suggested coordinates + radius
- `approval_needed` with workflow plan
- `ready_to_execute` — triggers orchestrator
- [message](file:///c:/Users/adity/POSEIDON/src/state/models.py#89-102) — general conversational response

LLM system prompt embedded as a constant (`SUPERVISOR_SYSTEM_PROMPT`), focused on oceanographic domain expertise and reasoning.

#### [NEW] [clarification.py](file:///c:/Users/adity/POSEIDON/src/supervisor/clarification.py)
`ClarificationEngine` — generates structured clarification questions. Uses the existing [ScientificIntent](file:///c:/Users/adity/POSEIDON/src/state/models.py#12-43) model to detect which fields are `None` and generates user-friendly multi-choice questions. No LLM call needed — rule-based with fallback to LLM for ambiguous cases.

#### [NEW] [map_interface.py](file:///c:/Users/adity/POSEIDON/src/supervisor/map_interface.py)
`MapInterface` — generates Folium maps for the Streamlit UI:
- `generate_confirmation_map(lat, lon, radius_km)` → returns a `folium.Map` object
- `check_argo_coverage(lat, lon, radius_km)` → queries the SQLite DB for nearby float data, returns count
- Uses the existing geosolver for geocoding

#### [NEW] [planner.py](file:///c:/Users/adity/POSEIDON/src/supervisor/planner.py)
`WorkflowPlanner` — produces an approval-ready execution plan:
- Input: complete [ScientificIntent](file:///c:/Users/adity/POSEIDON/src/state/models.py#12-43) + confirmed coordinates
- Output: `WorkflowPlan` (steps list, estimated time, estimated API cost, data volume estimate)
- Cost estimation based on number of ARGO floats × analysis complexity

---

### State Model Updates

#### [MODIFY] [models.py](file:///c:/Users/adity/POSEIDON/src/state/models.py)
Add `SupervisorPhase` enum and `SupervisorState` Pydantic model (imported from `src/supervisor/state.py` to avoid circular deps — or define inline). No breaking changes to existing models.

---

### Streamlit UI Redesign

#### [MODIFY] [app.py](file:///c:/Users/adity/POSEIDON/app.py)
Major changes to support multi-turn Supervisor interaction:

1. **Session state additions**: `supervisor_state: SupervisorConversationState`, `pending_clarification`, `pending_map`, `pending_approval`
2. **Chat flow redesign**: Instead of sending every user message directly to the orchestrator, route through `SupervisorAgent.process()` first:
   - If Supervisor returns `clarification_needed` → render clarification widget (selectboxes/buttons)
   - If Supervisor returns `map_confirmation_needed` → render Folium map with `st_folium`
   - If Supervisor returns `approval_needed` → render approval card with Yes/No/Modify buttons
   - If Supervisor returns `ready_to_execute` → run existing [execute_with_live_progress()](file:///c:/Users/adity/POSEIDON/app.py#207-257)
3. **New rendering functions**:
   - `render_clarification_dialog(questions)` — selectboxes + submit button
   - `render_map_confirmation(lat, lon, radius)` — Folium map + confirm/adjust buttons
   - `render_approval_card(plan)` — workflow plan display + approve/modify/cancel buttons
   - `render_supervisor_message(msg)` — styled supervisor responses

---

### API v2 Endpoints (Phase 2)

#### [MODIFY] [server.py](file:///c:/Users/adity/POSEIDON/src/api/server.py)
Add new v2 endpoints for external API consumers (separate from Streamlit):
- `POST /api/v2/supervisor/query` — initial query, returns next step
- `POST /api/v2/supervisor/clarify` — submit clarification answers
- `POST /api/v2/supervisor/confirm-location` — confirm map coordinates
- `POST /api/v2/supervisor/approve-workflow` — approve execution
- `GET /api/v2/supervisor/status/{execution_id}` — poll execution status

---

### Dependencies

#### [MODIFY] [requirements.txt](file:///c:/Users/adity/POSEIDON/requirements.txt)
Add:
- `folium>=0.18.0` — interactive map rendering
- `streamlit-folium>=0.23.0` — Streamlit-Folium bridge

---

## Verification Plan

### Existing Tests (will not be broken)

The existing integration test at [test_orchestrator_smoke.py](file:///c:/Users/adity/POSEIDON/tests/integration/test_orchestrator_smoke.py) uses monkeypatching to test the orchestrator pipeline end-to-end. This test will continue to pass since we're adding a new layer **above** the orchestrator, not modifying it.

Run with:
```
cd c:\Users\adity\POSEIDON
python -m pytest tests/integration/test_orchestrator_smoke.py -v
```

### New Automated Tests

#### [NEW] `tests/unit/test_supervisor_state.py`
Test `SupervisorConversationState` phase transitions:
- `greeting → clarifying` when missing params detected
- `clarifying → confirming_location` when location needs confirmation
- `confirming_location → awaiting_approval` for complex queries
- `awaiting_approval → executing → completed`

Run with:
```
python -m pytest tests/unit/test_supervisor_state.py -v
```

#### [NEW] `tests/unit/test_clarification_engine.py`
Test `ClarificationEngine`:
- Query with missing depth → generates depth question
- Query with missing time_range → generates time question
- Query with all params → returns no questions

Run with:
```
python -m pytest tests/unit/test_clarification_engine.py -v
```

### Manual Verification (Streamlit)

After implementation, run the Streamlit app and test these flows:

1. **Ambiguous query flow**: Type "Show me ocean data" → Supervisor should ask clarifying questions → answer them → should proceed
2. **Location confirmation flow**: Type "Temperature near Hawaii at 500m" → Supervisor should show a Folium map → click confirm → should proceed to execution
3. **Direct execution flow**: Type "What is temperature at 19.0N 72.8E at 500m depth for last 7 days?" → Supervisor should skip clarification and execute directly

```
cd c:\Users\adity\POSEIDON
streamlit run app.py
```
