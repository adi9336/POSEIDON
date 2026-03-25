"""
Supervisor Agent — the central intelligence layer for POSEIDON.
Routes user queries through clarification → map confirmation → approval → execution.
Uses OpenAI for query reasoning and intent analysis.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langsmith import traceable

from src.supervisor.clarification import ClarificationEngine
from src.supervisor.map_interface import MapInterface
from src.supervisor.planner import WorkflowPlanner
from src.supervisor.state import (
    SupervisorConversationState,
    SupervisorPhase,
    SupervisorResponse,
    WorkflowPlan,
)
from src.tools.intent_extractor import extract_intent_with_llm, fallback_intent_extraction

load_dotenv()
logger = logging.getLogger(__name__)

SUPERVISOR_SYSTEM_PROMPT = """You are the Supervisor Agent for POSEIDON, an oceanographic data analysis system
that uses ARGO float data (temperature, salinity, pressure, depth, nitrate).

You are an expert at GEOGRAPHIC REASONING. You understand coastlines, borders,
ocean regions, and can convert region names into precise coordinate ranges.

## CHAIN OF THOUGHT — Follow these steps for EVERY query:

### STEP 1: GEOGRAPHIC DECOMPOSITION
Think carefully about what geographic area the user means:
- "Russian borders" → Russia's maritime borders span ~41°N to 82°N, coastline along
  Arctic Ocean (66-82°N, 30°E-170°W), Pacific coast (42-62°N, 130-170°E),
  Black Sea (42-47°N, 36-42°E), Baltic Sea (54-60°N, 20-30°E)
- "Arabian Sea" → approximately 0°N-25°N, 44°E-77°E
- "Bay of Bengal" → approximately 5°N-22°N, 80°E-95°E
- "near Mumbai" → center at 19.08°N, 72.88°E, radius ~200km
- "Pacific Ocean" → massive area, ask user to narrow down
- "Mediterranean" → approximately 30°N-46°N, 6°W-36°E
- For a COUNTRY'S OCEAN DATA: identify the country's COASTLINE, not its center.
  Russia's center is landlocked. Its COASTAL regions are what matter for ocean data.
- For BORDERS: identify the maritime boundaries along the coast, not land borders.

### STEP 2: PARAMETER ANALYSIS
Identify: variable (temp/psal/nitrate), depth, time_range.
What's missing? What's ambiguous?

### STEP 3: ARGO COVERAGE ASSESSMENT
ARGO floats are ocean-based. Consider:
- Does the region have ocean access?
- Is it a coastal area, open ocean, or landlocked?
- Large regions may need to be narrowed down

### STEP 4: DECISION LOGIC
Based on the above, decide:
- needs_clarification: if key params are missing OR the region is too large/ambiguous
- needs_map_confirmation: if we resolved coordinates but user should verify
- needs_approval: if the query is complex or resource-heavy
- ready_to_execute: if everything is clear and reasonable

### STEP 5: RESPONSE SYNTHESIS
Generate a helpful message explaining your reasoning to the user.

## RESPONSE FORMAT (valid JSON):
{
  "reasoning": "Your detailed step-by-step geographic + parameter reasoning",
  "intent_type": "direct_retrieval | exploratory | analytical | ambiguous",
  "needs_clarification": true/false,
  "needs_map_confirmation": true/false,
  "needs_approval": true/false,
  "message": "Your conversational response to the user",
  "suggestions": ["suggestion 1", "suggestion 2"],
  "geographic_analysis": {
    "region_type": "point | coastal | ocean_region | country_coast | too_large",
    "center_lat": number_or_null,
    "center_lon": number_or_null,
    "lat_range": [min_lat, max_lat] or null,
    "lon_range": [min_lon, max_lon] or null,
    "coastline_description": "brief description of the relevant coastline"
  }
}

## CRITICAL RULES:
- NEVER geocode a country to its capital for ocean data. Use its COASTLINE.
- For large countries (Russia, USA, Canada, Australia), ASK which coast.
- Think about what OCEAN is near that location.
- Be conversational and explain your geographic reasoning to the user.
- Provide smart defaults the user can quickly accept.
- When uncertain: ASK, don't assume.
"""


class SupervisorAgent:
    """
    Core Supervisor Agent that manages the conversation flow.

    Lifecycle:
        1. User sends query → process()
        2. Supervisor analyzes → returns SupervisorResponse
        3. UI renders response (clarification / map / approval / results)
        4. User responds → process() again with updated state
        5. Repeat until ready_to_execute or completed
    """

    def __init__(self) -> None:
        self.clarification_engine = ClarificationEngine()
        self.map_interface = MapInterface()
        self.workflow_planner = WorkflowPlanner()
        self._llm = None

    def _get_llm(self):
        """Lazy-init OpenAI client."""
        if self._llm is None:
            try:
                from openai import OpenAI
                self._llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except Exception as exc:
                logger.error(f"Failed to initialize OpenAI: {exc}")
        return self._llm

    @traceable(name="supervisor_process")
    def process(
        self,
        session: SupervisorConversationState,
        user_input: str,
    ) -> SupervisorResponse:
        """
        Main entry point. Process user input and return the appropriate response.
        """
        session.add_message("user", user_input)

        # If session completed/errored, reset for a new query (preserving memory)
        if session.phase in (SupervisorPhase.COMPLETED, SupervisorPhase.ERROR):
            session.reset_for_new_query()

        # Route based on current phase
        if session.phase == SupervisorPhase.GREETING:
            return self._handle_initial_query(session, user_input)
        elif session.phase == SupervisorPhase.CLARIFYING:
            return self._handle_clarification_response(session, user_input)
        elif session.phase == SupervisorPhase.CONFIRMING_LOCATION:
            return self._handle_location_confirmation(session, user_input)
        elif session.phase == SupervisorPhase.AWAITING_APPROVAL:
            return self._handle_approval_response(session, user_input)
        else:
            return self._handle_initial_query(session, user_input)

    @traceable(name="supervisor_initial_query")
    def _handle_initial_query(
        self,
        session: SupervisorConversationState,
        query: str,
    ) -> SupervisorResponse:
        """Handle the first user query — analyze intent and decide next step."""
        session.user_query = query

        # Step 1: Parse intent using existing intent extractor
        intent_dict = self._extract_intent(query)
        session.confirmed_intent = intent_dict

        # Step 2: Use LLM for higher-level Chain of Thought reasoning
        llm_analysis = self._reason_about_query(query, intent_dict, session)

        # Step 3: Merge geographic analysis from LLM into intent
        # The LLM's geographic reasoning is SMARTER than naive geocoding
        geo_analysis = llm_analysis.get("geographic_analysis", {})
        if geo_analysis:
            # If the LLM figured out a center point, use it
            if geo_analysis.get("center_lat") is not None and geo_analysis.get("center_lon") is not None:
                intent_dict["lat"] = geo_analysis["center_lat"]
                intent_dict["lon"] = geo_analysis["center_lon"]
                session.confirmed_intent = intent_dict
            # Store lat/lon ranges if provided (for bounding box queries)
            if geo_analysis.get("lat_range"):
                intent_dict["lat_range"] = geo_analysis["lat_range"]
            if geo_analysis.get("lon_range"):
                intent_dict["lon_range"] = geo_analysis["lon_range"]
            if geo_analysis.get("coastline_description"):
                intent_dict["coastline_description"] = geo_analysis["coastline_description"]
            session.confirmed_intent = intent_dict

        # Step 4: Check what's missing (rule-based)
        missing_params = self.clarification_engine.identify_missing_params(
            intent_dict, session.clarifications_collected
        )

        # Step 5: Decision logic — prefer LLM's reasoning over rule-based
        intent_type = llm_analysis.get("intent_type", "exploratory")
        suggestions = llm_analysis.get("suggestions", [])

        # If the LLM explicitly says clarification is needed, trust it
        # (e.g., "Russia is too large, which coast?")
        llm_needs_clarification = llm_analysis.get("needs_clarification", False)
        if llm_needs_clarification:
            return self._ask_for_clarification(session, missing_params, llm_analysis, suggestions)

        # If critical params are still missing → clarify
        critical_missing = [p for p in missing_params if p in ("variable", "location")]
        if critical_missing:
            return self._ask_for_clarification(session, missing_params, llm_analysis, suggestions)

        # If location was resolved → confirm on map
        needs_map = llm_analysis.get("needs_map_confirmation", False)
        has_coords = intent_dict.get("lat") is not None and intent_dict.get("lon") is not None
        if (needs_map or has_coords) and intent_dict.get("lat") is not None:
            return self._ask_for_map_confirmation(session, intent_dict, suggestions)

        # If query is complex → ask for approval
        needs_approval = llm_analysis.get("needs_approval", False)
        if needs_approval or intent_type == "analytical":
            return self._ask_for_approval(session, intent_dict, suggestions)

        # Simple & complete → ready to execute
        return self._ready_to_execute(session, intent_dict, llm_analysis, suggestions)

    def _handle_clarification_response(
        self,
        session: SupervisorConversationState,
        user_input: str,
    ) -> SupervisorResponse:
        """User provided clarification answers. Merge and re-evaluate."""
        # This is typically called by the UI after collecting selectbox answers,
        # but user might also type a free-text response.
        # The UI layer handles structured answers via merge_clarifications().
        # Here we handle free-text fallback.

        # Re-analyze with the collected clarifications merged in
        intent = session.confirmed_intent or {}
        intent.update(session.clarifications_collected)

        # Check if we still need more info
        missing = self.clarification_engine.identify_missing_params(
            intent, session.clarifications_collected
        )
        critical_missing = [p for p in missing if p in ("variable", "location")]

        if critical_missing and session.clarification_turn_count < session.max_clarification_turns:
            questions = self.clarification_engine.generate_questions(
                critical_missing,
                already_asked=list(session.clarifications_collected.keys()),
            )
            if questions:
                msg = "Thanks! I still need a bit more information:"
                session.add_message("supervisor", msg)
                return SupervisorResponse(
                    response_type="clarification_needed",
                    message=msg,
                    clarification_questions=questions,
                )

        # Check if map confirmation is needed
        wants_map = session.clarifications_collected.get("wants_map")
        if wants_map:
            session.advance_phase(SupervisorPhase.CONFIRMING_LOCATION)
            msg = "Please select your area of interest on the map below."
            session.add_message("supervisor", msg)
            return SupervisorResponse(
                response_type="map_confirmation_needed",
                message=msg,
                map_data={"center_lat": 20.0, "center_lon": 73.0, "radius_km": 200.0},
            )

        # If location was set via clarification, try to geocode
        loc_name = session.clarifications_collected.get("location") or intent.get("location")
        if loc_name and intent.get("lat") is None:
            geocoded = self._try_geocode(loc_name)
            if geocoded:
                intent.update(geocoded)
                session.confirmed_intent = intent
                return self._ask_for_map_confirmation(session, intent, [])

        # We have enough → proceed to approval or execution
        session.confirmed_intent = intent
        return self._ask_for_approval(session, intent, [])

    def _handle_location_confirmation(
        self,
        session: SupervisorConversationState,
        user_input: str,
    ) -> SupervisorResponse:
        """User confirmed (or adjusted) location on map. Proceed to approval."""
        # The UI layer sets session.confirmed_coordinates before calling this.
        intent = session.confirmed_intent or {}
        if session.confirmed_coordinates:
            intent["lat"] = session.confirmed_coordinates.get("lat")
            intent["lon"] = session.confirmed_coordinates.get("lon")
            session.confirmed_intent = intent

        return self._ask_for_approval(session, intent, [])

    def _handle_approval_response(
        self,
        session: SupervisorConversationState,
        user_input: str,
    ) -> SupervisorResponse:
        """User approved or rejected the workflow plan."""
        lower = user_input.strip().lower()
        if lower in ("yes", "y", "approve", "go", "proceed", "ok"):
            return self._ready_to_execute(session, session.confirmed_intent or {}, {}, [])
        elif lower in ("no", "n", "cancel"):
            session.advance_phase(SupervisorPhase.GREETING)
            msg = "No problem! Feel free to ask a different question or adjust your parameters."
            session.add_message("supervisor", msg)
            return SupervisorResponse(response_type="message", message=msg)
        else:
            msg = "I didn't catch that. Would you like to proceed with the analysis? (Yes/No)"
            session.add_message("supervisor", msg)
            return SupervisorResponse(response_type="message", message=msg)

    # ── Helper builders ──

    def _ask_for_clarification(
        self,
        session: SupervisorConversationState,
        missing_params: List[str],
        llm_analysis: Dict[str, Any],
        suggestions: List[str],
    ) -> SupervisorResponse:
        session.advance_phase(SupervisorPhase.CLARIFYING)
        questions = self.clarification_engine.generate_questions(
            missing_params,
            already_asked=list(session.clarifications_collected.keys()),
        )
        msg = llm_analysis.get(
            "message",
            "I'd like to help you explore ocean data! Could you clarify a few things?"
        )
        session.add_message("supervisor", msg)
        return SupervisorResponse(
            response_type="clarification_needed",
            message=msg,
            clarification_questions=questions,
            suggestions=suggestions,
        )

    def _ask_for_map_confirmation(
        self,
        session: SupervisorConversationState,
        intent: Dict[str, Any],
        suggestions: List[str],
    ) -> SupervisorResponse:
        session.advance_phase(SupervisorPhase.CONFIRMING_LOCATION)
        lat = intent.get("lat", 0.0)
        lon = intent.get("lon", 0.0)
        map_data = self.map_interface.generate_map_data(lat, lon)
        location_name = intent.get("location", "the selected region")
        float_count = map_data.get("argo_float_count", 0)

        msg = (
            f"I've identified **{location_name}** ({lat:.1f}°, {lon:.1f}°). "
        )
        coastline_desc = intent.get("coastline_description", "")
        if coastline_desc:
            msg += f"\n📍 *{coastline_desc}*\n\n"
        msg += (
            f"There are **{float_count} ARGO floats** in this area.\n\n"
            f"Please confirm the search area on the map below, or use the draw tools to define your own."
        )
        session.add_message("supervisor", msg)
        return SupervisorResponse(
            response_type="map_confirmation_needed",
            message=msg,
            map_data=map_data,
            suggestions=suggestions,
        )

    def _ask_for_approval(
        self,
        session: SupervisorConversationState,
        intent: Dict[str, Any],
        suggestions: List[str],
    ) -> SupervisorResponse:
        session.advance_phase(SupervisorPhase.AWAITING_APPROVAL)
        coords = session.confirmed_coordinates
        plan = self.workflow_planner.build_plan(intent, coords)
        session.workflow_plan = plan.model_dump()

        msg = (
            f"Here's the execution plan for your query:\n\n"
            f"**{plan.name}**\n\n"
            + "\n".join(plan.steps)
            + f"\n\n⏱️ Estimated time: {plan.estimated_time}\n"
            f"💰 Estimated cost: {plan.estimated_cost}\n"
            f"📊 Data volume: {plan.data_volume}\n\n"
            f"Would you like to proceed? (Yes / No / Modify)"
        )
        session.add_message("supervisor", msg)
        return SupervisorResponse(
            response_type="approval_needed",
            message=msg,
            workflow_plan=plan,
            suggestions=suggestions,
        )

    def _ready_to_execute(
        self,
        session: SupervisorConversationState,
        intent: Dict[str, Any],
        llm_analysis: Dict[str, Any],
        suggestions: List[str],
    ) -> SupervisorResponse:
        session.advance_phase(SupervisorPhase.EXECUTING)
        variable = intent.get("variable", "temperature")
        location = intent.get("location", "the specified region")
        msg = f"Starting analysis of **{variable}** data near **{location}**..."
        session.add_message("supervisor", msg)
        return SupervisorResponse(
            response_type="ready_to_execute",
            message=msg,
            suggestions=suggestions,
        )

    # ── Intent extraction ──

    def _extract_intent(self, query: str) -> Dict[str, Any]:
        """Use existing intent extractor to parse the query."""
        try:
            intent = extract_intent_with_llm(query)
            return intent.model_dump() if hasattr(intent, "model_dump") else dict(intent)
        except Exception as exc:
            logger.warning(f"LLM intent extraction failed: {exc}, using fallback")
            try:
                intent = fallback_intent_extraction(query)
                return intent.model_dump() if hasattr(intent, "model_dump") else dict(intent)
            except Exception:
                return {}

    # ── LLM reasoning ──

    @traceable(name="supervisor_llm_reasoning", run_type="llm")
    def _reason_about_query(
        self,
        query: str,
        intent: Dict[str, Any],
        session: Optional[SupervisorConversationState] = None,
    ) -> Dict[str, Any]:
        """Use OpenAI to reason about the query at a higher level."""
        client = self._get_llm()
        if not client:
            return self._fallback_reasoning(intent)

        try:
            # Build context from session memory + recent chat
            context_parts = []
            if session:
                memory_summary = session.get_memory_summary()
                if memory_summary:
                    context_parts.append(f"SESSION MEMORY:\n{memory_summary}")
                # Include last 6 messages for immediate context
                recent_chat = session.chat_history[-6:]
                if recent_chat:
                    chat_lines = [f"  {m['role']}: {m['content'][:200]}" for m in recent_chat]
                    context_parts.append(f"RECENT CHAT:\n" + "\n".join(chat_lines))

            context_str = "\n\n".join(context_parts)
            user_msg = (
                f"User query: \"{query}\"\n\n"
                f"Parsed intent: {json.dumps(intent, default=str)}\n\n"
            )
            if context_str:
                user_msg += f"CONVERSATION CONTEXT:\n{context_str}\n\n"
            user_msg += "Analyze this query and respond in JSON format."

            response = client.chat.completions.create(
                model=os.getenv("SUPERVISOR_MODEL", "gpt-4o"),
                messages=[
                    {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            return json.loads(content)
        except Exception as exc:
            logger.warning(f"Supervisor LLM reasoning failed: {exc}")
            return self._fallback_reasoning(intent)

    def _fallback_reasoning(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based fallback when LLM is unavailable."""
        has_var = bool(intent.get("variable"))
        has_loc = bool(intent.get("location") or intent.get("lat"))
        has_depth = intent.get("depth") is not None

        if has_var and has_loc and has_depth:
            return {
                "intent_type": "direct_retrieval",
                "needs_clarification": False,
                "needs_map_confirmation": bool(intent.get("location") and intent.get("lat")),
                "needs_approval": False,
                "message": "I have all the parameters needed. Let me execute this query.",
                "suggestions": [],
            }
        elif has_var and has_loc:
            return {
                "intent_type": "exploratory",
                "needs_clarification": False,
                "needs_map_confirmation": bool(intent.get("lat")),
                "needs_approval": True,
                "message": "I'll prepare an analysis plan for your review.",
                "suggestions": [
                    "Would you also like salinity data?",
                    "I can show trends over 30 days",
                ],
            }
        else:
            return {
                "intent_type": "ambiguous",
                "needs_clarification": True,
                "needs_map_confirmation": False,
                "needs_approval": False,
                "message": "I'd love to help! Could you clarify a few things?",
                "suggestions": [],
            }

    def _try_geocode(self, location_name: str) -> Optional[Dict[str, float]]:
        """Attempt geocoding using the marine polygon classifier."""
        try:
            from src.tools.geosolver import resolve_location_fast
            result = resolve_location_fast(location_name)
            if result:
                return {"lat": result[0], "lon": result[1]}
        except Exception as exc:
            logger.warning(f"Geocoding failed for '{location_name}': {exc}")
        return None
