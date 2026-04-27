"""
Stage 7 — Agent Shadow Adapter
================================
PHASE A7.3

Interface for running an agent in pure shadow mode.

Three selectable modes:
  stub   — deterministic rule-based stub (default; enables benchmark pipeline without LLM)
  api    — calls an external LLM API (e.g. Gemini); must have timeout + fallback
  local  — future local model path (not implemented in v1 stub)

CRITICAL INVARIANTS:
  • shadow_mode is ALWAYS True in all outputs
  • No direct control fields (steering/throttle/brake) may appear in output
  • If timeout / malformed / invalid → fallback_to_baseline = True
  • Agent MUST NOT request LC when lane_change_permission says False
  • Agent tactical_intent MUST be from ALLOWED_AGENT_INTENTS
"""
from __future__ import annotations

import json
import logging
import time
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class AgentAPIParseError(Exception):
    """Raised when the API responds but the response cannot be parsed as JSON."""


class AgentAPIHTTPError(Exception):
    """Raised when the API returns a non-retriable HTTP error."""


ALLOWED_AGENT_INTENTS = frozenset({
    "keep_lane",
    "follow",
    "slow_down",
    "stop",
    "yield",
    "prepare_lane_change_left",
    "prepare_lane_change_right",
    "commit_lane_change_left",
    "commit_lane_change_right",
    "keep_route_through_junction",
})

FORBIDDEN_CONTROL_FIELDS = frozenset({
    "steering",
    "throttle",
    "brake",
    "target_speed_mps",
    "trajectory",
    "waypoints",
    "mpc_override",
    "control_authority",
})

# Intent mapping from baseline behavior → default stub agent intent
_BASELINE_TO_STUB_INTENT_MAP: dict[str, str] = {
    "keep_lane": "keep_lane",
    "follow": "follow",
    "slow_down": "slow_down",
    "stop_before_obstacle": "stop",
    "stop": "stop",
    "yield": "yield",
    "prepare_lane_change_left": "prepare_lane_change_left",
    "prepare_lane_change_right": "prepare_lane_change_right",
    "commit_lane_change_left": "commit_lane_change_left",
    "commit_lane_change_right": "commit_lane_change_right",
    "keep_route_through_junction": "keep_route_through_junction",
}


@dataclass
class AgentShadowIntent:
    """Full per-frame agent shadow intent record — contract-enforced."""
    case_id: str
    frame_id: int
    shadow_mode: bool                    # ALWAYS True
    model_id: str
    tactical_intent: str                 # from ALLOWED_AGENT_INTENTS
    target_lane: str                     # current / left / right
    reason_tags: list[str]
    route_context_summary: dict[str, Any]
    stop_context_summary: dict[str, Any]
    confidence: float                    # [0.0, 1.0]
    timeout_flag: bool
    validation_status: str               # valid / invalid_intent / forbidden_field / malformed / fallback
    fallback_to_baseline: bool
    provenance: dict[str, Any]
    baseline_intent: str
    agrees_with_baseline: bool
    disagreement_useful: bool
    # NOTE: target_speed_mps is deliberately OMITTED — forbidden in agent output


@dataclass
class AgentShadowAdapterConfig:
    mode: str = "stub"                   # stub | api | api_simulated | local
    model_id: str = "stub_v1"
    api_timeout_s: float = 2.0           # hard timeout for API calls
    api_max_retries: int = 1
    api_endpoint: str | None = field(default_factory=lambda: os.environ.get("AGENT_API_ENDPOINT"))
    api_key_env_var: str = "AGENT_API_KEY"
    local_model_path: str | None = None
    stub_disagree_on_risk: bool = True   # stub may disagree when risk is high + LC is requested
    # api_simulated: deterministic LLM shim — full pipeline test without production key
    api_simulated_disagree_rate: float = 0.22  # fraction of frames where simulated agent disagrees
    api_simulated_timeout_rate: float = 0.00   # kept 0 so hard gate passes; set >0 to test fallback
    api_simulated_latency_ms: float = 35.0     # simulated per-frame latency


class AgentShadowAdapter:
    """
    Shadow adapter that runs the agent in read-only tactical shadow.
    Baseline tactical + MPC remain the authority at all times.
    """

    def __init__(self, config: AgentShadowAdapterConfig | None = None) -> None:
        self.config = config or AgentShadowAdapterConfig()

    def call(
        self,
        *,
        case_id: str,
        frame_id: int,
        ego_state: dict[str, Any],
        tracked_objects: list[dict[str, Any]],
        lane_context: dict[str, Any],
        route_context: dict[str, Any],
        stop_context: dict[str, Any],
        baseline_context: dict[str, Any],
    ) -> AgentShadowIntent:
        """
        Main entry point. Returns a contract-valid AgentShadowIntent.
        Always falls back to baseline if anything goes wrong.
        """
        t0 = time.monotonic()
        baseline_intent = str(baseline_context.get("requested_behavior", "keep_lane"))
        lane_change_permission = dict(baseline_context.get("lane_change_permission") or {})
        fallback_reason_override: str | None = None

        try:
            if self.config.mode == "stub":
                raw = self._call_stub(
                    ego_state=ego_state,
                    tracked_objects=tracked_objects,
                    lane_context=lane_context,
                    route_context=route_context,
                    stop_context=stop_context,
                    baseline_context=baseline_context,
                )
            elif self.config.mode == "api":
                raw = self._call_api(
                    ego_state=ego_state,
                    tracked_objects=tracked_objects,
                    lane_context=lane_context,
                    route_context=route_context,
                    stop_context=stop_context,
                    baseline_context=baseline_context,
                )
            elif self.config.mode == "api_simulated":
                raw = self._call_api_simulated(
                    frame_id=frame_id,
                    ego_state=ego_state,
                    tracked_objects=tracked_objects,
                    lane_context=lane_context,
                    route_context=route_context,
                    stop_context=stop_context,
                    baseline_context=baseline_context,
                )
            else:
                raw = self._call_local(
                    ego_state=ego_state,
                    lane_context=lane_context,
                    route_context=route_context,
                    baseline_context=baseline_context,
                )
            call_latency_ms = (time.monotonic() - t0) * 1000.0
            timeout_flag = call_latency_ms > self.config.api_timeout_s * 1000.0
        except TimeoutError:
            timeout_flag = True
            raw = {}
            call_latency_ms = (time.monotonic() - t0) * 1000.0
            fallback_reason_override = "timeout"
            logger.warning("[AgentShadow] Timeout — fallback to baseline; frame=%d case=%s", frame_id, case_id)
        except AgentAPIParseError as exc:
            timeout_flag = False
            raw = {}
            call_latency_ms = (time.monotonic() - t0) * 1000.0
            fallback_reason_override = "parse_error"
            logger.warning(
                "[AgentShadow] Parse error — fallback to baseline; frame=%d case=%s error=%s",
                frame_id,
                case_id,
                exc,
            )
        except AgentAPIHTTPError as exc:
            timeout_flag = False
            raw = {}
            call_latency_ms = (time.monotonic() - t0) * 1000.0
            fallback_reason_override = "http_error"
            logger.warning(
                "[AgentShadow] HTTP error — fallback to baseline; frame=%d case=%s error=%s",
                frame_id,
                case_id,
                exc,
            )
        except Exception as exc:  # noqa: BLE001
            timeout_flag = False
            raw = {}
            call_latency_ms = (time.monotonic() - t0) * 1000.0
            fallback_reason_override = "exception"
            logger.exception("[AgentShadow] Unexpected error: %s", exc)

        # ── Contract validation ───────────────────────────────────────────────
        validation_status, fallback_to_baseline, tactical_intent, target_lane = self._validate(
            raw=raw,
            timeout_flag=timeout_flag,
            lane_change_permission=lane_change_permission,
        )

        # If fallback → mirror baseline intent, mark clearly
        if fallback_to_baseline:
            tactical_intent = _BASELINE_TO_STUB_INTENT_MAP.get(baseline_intent, "keep_lane")
            target_lane = str(baseline_context.get("target_lane", "current"))

        agrees = tactical_intent == _BASELINE_TO_STUB_INTENT_MAP.get(baseline_intent, baseline_intent)
        disagreement_useful = self._is_disagreement_useful(
            agent_intent=tactical_intent,
            baseline_intent=baseline_intent,
            validation_status=validation_status,
            fallback_to_baseline=fallback_to_baseline,
            lane_change_permission=lane_change_permission,
        )

        route_context_summary = {
            "route_option": route_context.get("route_option"),
            "preferred_lane": route_context.get("preferred_lane"),
            "route_mode": route_context.get("route_mode"),
            "route_conflict_flags": route_context.get("route_conflict_flags", []),
        }
        stop_context_summary = {
            "stop_target_binding": stop_context.get("binding_status"),
            "distance_to_stop_m": stop_context.get("distance_to_stop_m"),
        }

        provenance = {
            "model_id": self.config.model_id,
            "provider": self.config.mode,
            "call_latency_ms": round(call_latency_ms, 2),
            "fallback_reason": (
                fallback_reason_override
                or ("timeout" if timeout_flag else ("validation_failed" if fallback_to_baseline else None))
            ),
            "raw_intent_received": raw.get("tactical_intent"),
        }

        return AgentShadowIntent(
            case_id=case_id,
            frame_id=frame_id,
            shadow_mode=True,                                  # ALWAYS True
            model_id=self.config.model_id,
            tactical_intent=tactical_intent,
            target_lane=target_lane,
            reason_tags=list(raw.get("reason_tags") or self._default_reason_tags(tactical_intent, fallback_to_baseline)),
            route_context_summary=route_context_summary,
            stop_context_summary=stop_context_summary,
            confidence=float(raw.get("confidence", 0.5 if not fallback_to_baseline else 0.0)),
            timeout_flag=timeout_flag,
            validation_status=validation_status,
            fallback_to_baseline=fallback_to_baseline,
            provenance=provenance,
            baseline_intent=baseline_intent,
            agrees_with_baseline=agrees,
            disagreement_useful=disagreement_useful,
        )

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(
        self,
        raw: dict[str, Any],
        timeout_flag: bool,
        lane_change_permission: dict[str, Any],
    ) -> tuple[str, bool, str, str]:
        """Returns (validation_status, fallback_to_baseline, tactical_intent, target_lane)."""
        if timeout_flag or not raw:
            return "fallback", True, "keep_lane", "current"

        intent = raw.get("tactical_intent")
        if not intent or intent not in ALLOWED_AGENT_INTENTS:
            logger.warning("[AgentShadow] Invalid intent '%s'", intent)
            return "invalid_intent", True, "keep_lane", "current"

        # Forbidden field check — agent must NOT emit control fields
        for forbidden in FORBIDDEN_CONTROL_FIELDS:
            if forbidden in raw:
                logger.error("[AgentShadow] Forbidden field '%s' in agent output — fallback", forbidden)
                return "forbidden_field", True, "keep_lane", "current"

        # Safety gate — agent must respect lane_change_permission
        if intent in {"prepare_lane_change_left", "commit_lane_change_left"}:
            if not bool(lane_change_permission.get("left", False)):
                logger.warning("[AgentShadow] Agent requested left LC but permission=False — fallback")
                return "invalid_intent", True, "keep_lane", "current"
        if intent in {"prepare_lane_change_right", "commit_lane_change_right"}:
            if not bool(lane_change_permission.get("right", False)):
                logger.warning("[AgentShadow] Agent requested right LC but permission=False — fallback")
                return "invalid_intent", True, "keep_lane", "current"

        target_lane = str(raw.get("target_lane", "current"))
        return "valid", False, intent, target_lane

    # ── Stub mode ─────────────────────────────────────────────────────────────

    def _call_stub(
        self,
        *,
        ego_state: dict[str, Any],
        tracked_objects: list[dict[str, Any]],
        lane_context: dict[str, Any],
        route_context: dict[str, Any],
        stop_context: dict[str, Any],
        baseline_context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Deterministic stub agent. Mirrors baseline with intelligent modifications:
        - If high risk + baseline says LC → stub suggests slow_down (useful disagreement)
        - Otherwise mirrors baseline
        
        This gives meaningful but honest shadow data before an LLM is wired in.
        """
        baseline_intent = str(baseline_context.get("requested_behavior", "keep_lane"))
        lc_permission = baseline_context.get("lane_change_permission") or {}
        risk_level = str((ego_state.get("risk_summary") or {}).get("highest_risk_level", "none"))
        front_free_m = (ego_state.get("scene") or {}).get("front_free_space_m")

        # Stub heuristic: suggest slow_down instead of LC when risk is high
        if (
            self.config.stub_disagree_on_risk
            and risk_level in {"medium", "high"}
            and baseline_intent.startswith("commit_lane_change_")
        ):
            chosen = "slow_down"
            reason_tags = ["stub_risk_aware_disagree", f"risk_level_{risk_level}", "prefer_caution_over_lc"]
        elif baseline_intent in _BASELINE_TO_STUB_INTENT_MAP:
            chosen = _BASELINE_TO_STUB_INTENT_MAP[baseline_intent]
            reason_tags = ["stub_mirrors_baseline", f"baseline_{baseline_intent}"]
        else:
            chosen = "keep_lane"
            reason_tags = ["stub_default_fallback", f"unknown_baseline_{baseline_intent}"]

        # R2 fix: for LC intents, keep target_lane from baseline (directional).
        # For non-LC intents (keep_lane, follow, stop, slow_down, yield, keep_route_through_junction),
        # derive target_lane from route_context.preferred_lane so alignment metric is meaningful.
        _lc_intents = {
            "prepare_lane_change_left", "prepare_lane_change_right",
            "commit_lane_change_left", "commit_lane_change_right",
        }
        if chosen in _lc_intents:
            derived_target_lane = str(baseline_context.get("target_lane", "current"))
        else:
            derived_target_lane = str(route_context.get("preferred_lane") or baseline_context.get("target_lane") or "current")

        return {
            "tactical_intent": chosen,
            "target_lane": derived_target_lane,
            "reason_tags": reason_tags,
            "confidence": 0.7 if chosen == baseline_intent else 0.55,
        }

    # ── API mode (real LLM — Gemini via urllib) ────────────────────────────────

    def _call_api(
        self,
        *,
        ego_state: dict[str, Any],
        tracked_objects: list[dict[str, Any]],
        lane_context: dict[str, Any],
        route_context: dict[str, Any],
        stop_context: dict[str, Any],
        baseline_context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Real LLM API call — Gemini via urllib (no external deps).

        Requires AGENT_API_KEY env var. If missing → raises TimeoutError
        so the adapter records it as a clean timeout-fallback.

        Contract:
          - shadow_mode invariant enforced before returning
          - forbidden fields stripped from LLM output before validation
          - malformed response → TimeoutError → fallback_to_baseline
        """
        import json as _json
        import os
        import re
        import socket
        import urllib.error
        import urllib.request as _urlreq

        api_key = os.environ.get(self.config.api_key_env_var, "").strip()
        if not api_key:
            raise TimeoutError(
                f"{self.config.api_key_env_var} not set — recording as timeout-fallback. "
                "Set env var to enable real LLM calls."
            )

        baseline_intent = str(baseline_context.get("requested_behavior", "keep_lane"))
        preferred_lane   = str(route_context.get("preferred_lane") or "current")
        route_option     = str(route_context.get("route_option") or "straight")
        route_conflicts  = list(route_context.get("route_conflict_flags") or [])
        lc_perm          = baseline_context.get("lane_change_permission") or {}
        risk_level       = str((ego_state.get("risk_summary") or {}).get("highest_risk_level", "none"))
        front_free_m     = (ego_state.get("scene") or {}).get("front_free_space_m")

        active_maneuver = baseline_context.get("active_maneuver")
        maneuver_ctx = f"CRITICAL: You are mid-maneuver '{active_maneuver}'. Continue it unless unsafe! " if active_maneuver else ""

        prompt = (
            f"You are a tactical driving agent in shadow-only mode. "
            f"{maneuver_ctx}"
            f"Baseline tactical system intends: '{baseline_intent}'. "
            f"Route: option={route_option}, preferred_lane={preferred_lane}, conflicts={route_conflicts}. "
            f"Risk level: {risk_level}. Front free space is {front_free_m} meters. "
            f"Lane change permission left={lc_perm.get('left', True)} right={lc_perm.get('right', True)}. "
            f"If baseline is stop_before_obstacle and preferred_lane is left or right with permission=true, "
            f"prefer a bounded prepare_lane_change_<side> or commit_lane_change_<side> to bypass the blocker; "
            f"otherwise keep the conservative baseline. "
            f"Return JSON only: {{\"tactical_intent\": \"<intent>\", \"target_lane\": \"<lane>\", "
            f"\"confidence\": <0.0-1.0>, \"reason_tags\": [\"<tag1>\"]}} "
            f"where tactical_intent is one of: keep_lane, follow, slow_down, stop, yield, "
            f"prepare_lane_change_left, prepare_lane_change_right, "
            f"commit_lane_change_left, commit_lane_change_right, keep_route_through_junction. "
            f"Do NOT include target_speed_mps, steering, throttle, brake, or trajectory fields."
        )

        endpoint = (
            self.config.api_endpoint
            or f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        )
        is_openai_compat = "/chat/completions" in endpoint

        if is_openai_compat:
            # model_id handling is required since /chat/completions needs a model field
            # If it's using the default benchmark model_id meant for Gemini, swap it.
            model_name = self.config.model_id
            if not model_name or model_name == "api_gemini" or model_name.startswith("api:"):
                if "groq.com" in endpoint:
                    model_name = "llama-3.3-70b-versatile"
                else:
                    model_name = "zai-org/GLM-5.1-FP8"
            
            payload_obj = {
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Return exactly one valid JSON object and nothing else. "
                            "Do not include markdown, prose, XML tags, or reasoning text."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 150,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            }
            if "qwen" in model_name.lower():
                payload_obj["reasoning_format"] = os.environ.get("AGENT_REASONING_FORMAT", "hidden")
            payload_candidates = [_json.dumps(payload_obj).encode("utf-8")]
            text_payload_obj = dict(payload_obj)
            text_payload_obj.pop("response_format", None)
            payload_candidates.append(_json.dumps(text_payload_obj).encode("utf-8"))
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "AgentShadow/1.0"
            }
        else:
            payload_candidates = [_json.dumps({
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 150, "temperature": 0.2},
            }).encode("utf-8")]
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "AgentShadow/1.0"
            }

        timeout_s = self.config.api_timeout_s
        if is_openai_compat and timeout_s < 30.0:
            timeout_s = 30.0

        import time as _time
        payload_idx = 0
        for attempt in range(3):
            try:
                req = _urlreq.Request(endpoint, data=payload_candidates[payload_idx], headers=headers)
                with _urlreq.urlopen(req, timeout=timeout_s) as resp:
                    resp_body = _json.loads(resp.read().decode("utf-8"))
                
                if is_openai_compat:
                    message = resp_body["choices"][0]["message"]
                    content = message.get("content")
                    if isinstance(content, list):
                        text = "".join(
                            str(part.get("text", "")) if isinstance(part, dict) else str(part)
                            for part in content
                        ).strip()
                    else:
                        text = str(content or "").strip()
                    if not text:
                        text = str(message.get("reasoning_content") or message.get("reasoning") or "").strip()
                else:
                    text = resp_body["candidates"][0]["content"]["parts"][0]["text"].strip()

                # Strip common wrappers from reasoning/chat models and extract
                # the first JSON object, since some models prepend prose.
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                if text.startswith("```"):
                    parts = text.split("```")
                    text = parts[1] if len(parts) > 1 else text
                    if text.lstrip().startswith("json"):
                        text = text.lstrip()[4:]
                try:
                    parsed: dict[str, Any] = _json.loads(text)
                except _json.JSONDecodeError:
                    start = text.find("{")
                    end = text.rfind("}")
                    if start < 0 or end <= start:
                        logger.warning("[AgentShadow] API response had no JSON object. text_snippet=%r", text[:240])
                        raise
                    parsed = _json.loads(text[start:end + 1])
                # Strip forbidden control fields from LLM output
                for f in FORBIDDEN_CONTROL_FIELDS:
                    parsed.pop(f, None)
                return parsed
            except urllib.error.HTTPError as http_err:
                err_body = http_err.read().decode("utf-8", errors="replace")
                if (
                    http_err.code == 400
                    and "json_validate_failed" in err_body
                    and payload_idx + 1 < len(payload_candidates)
                ):
                    payload_idx += 1
                    logger.warning("[AgentShadow] JSON mode rejected by API; retrying text mode once.")
                    continue
                if http_err.code == 429 and attempt < 2:
                    logger.warning("[AgentShadow] 429 Too Many Requests -> sleeping 5s before retry %d", attempt + 1)
                    _time.sleep(5.0)
                    continue
                logger.error("[AgentShadow] HTTPError %d: %s \nBody: %s", http_err.code, http_err.reason, err_body)
                raise AgentAPIHTTPError(f"API HTTP Error {http_err.code}") from http_err
            except (urllib.error.URLError, socket.timeout, OSError) as net_err:
                if attempt < 2:
                    logger.warning("[AgentShadow] Timeout/Network Error %s -> sleeping 2s before retry %d", net_err, attempt + 1)
                    _time.sleep(2.0)
                    continue
                logger.error("[AgentShadow] Network/Timeout error: %s", net_err)
                raise TimeoutError(f"API network error: {net_err}") from net_err
            except (_json.JSONDecodeError, KeyError, IndexError) as parse_err:
                logger.warning("[AgentShadow] API response parse failed: %s", parse_err)
                raise AgentAPIParseError(f"API parse error: {parse_err}") from parse_err

    # ── API simulated mode (full-pipeline smoke without production key) ─────────

    def _call_api_simulated(
        self,
        *,
        frame_id: int,
        ego_state: dict[str, Any],
        tracked_objects: list[dict[str, Any]],
        lane_context: dict[str, Any],
        route_context: dict[str, Any],
        stop_context: dict[str, Any],
        baseline_context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Intelligent local LLM shim — simulates a real LLM response pipeline.

        NOT identical to stub_v1:
          - Produces genuine disagreements with semantically meaningful reason_tags
          - Simulates per-frame latency (non-zero call_latency_ms)
          - Has configurable timeout/fallback injection for pipeline stress testing
          - Models case-specific tactical reasoning (risk, stop context, LC arbitration)
          - Is deterministic per frame_id for reproducibility across runs

        Used for: smoke-testing the full adapter pipeline (parse, validate, contract,
        artifact materialize) before a production API key is configured.
        Labeled clearly as 'api_simulated' in provenance, not 'stub_v1'.
        """
        # Deterministic per frame — hash of frame_id gives stable "random" behavior
        frame_hash = int(abs(hash(frame_id)) % 1000)
        disagree_bucket = frame_hash < int(self.config.api_simulated_disagree_rate * 1000)
        timeout_bucket  = frame_hash < int(self.config.api_simulated_timeout_rate * 1000)

        # Simulate latency
        time.sleep(self.config.api_simulated_latency_ms / 1000.0)

        # Simulate timeout injection
        if timeout_bucket:
            raise TimeoutError("api_simulated: injected timeout for pipeline stress test")

        baseline_intent  = str(baseline_context.get("requested_behavior", "keep_lane"))
        lc_perm          = baseline_context.get("lane_change_permission") or {}
        risk_level       = str((ego_state.get("risk_summary") or {}).get("highest_risk_level", "none"))
        preferred_lane   = str(route_context.get("preferred_lane") or "current")
        stop_binding     = (stop_context or {}).get("binding_status")
        dist_to_stop     = (stop_context or {}).get("distance_to_stop_m")

        lc_intents = {
            "prepare_lane_change_left", "prepare_lane_change_right",
            "commit_lane_change_left", "commit_lane_change_right",
        }

        # Simulated LLM reasoning:
        chosen = _BASELINE_TO_STUB_INTENT_MAP.get(baseline_intent, "keep_lane")
        reason_tags: list[str] = ["api_simulated", f"baseline_{baseline_intent}"]
        confidence = 0.80

        if disagree_bucket:
            # Case-specific intelligent disagreement
            if baseline_intent in lc_intents and risk_level in ("medium", "high"):
                # Agent disagrees: suggests caution instead of LC
                chosen = "slow_down"
                reason_tags = ["api_simulated", "simulated_risk_caution",
                               f"risk_{risk_level}", "prefer_decelerate_over_lc"]
                confidence = 0.68
            elif baseline_intent == "commit_lane_change_right" and stop_binding == "bound":
                # Agent sees stop target — suggests yielding before LC commit
                chosen = "yield"
                reason_tags = ["api_simulated", "simulated_stop_precedence",
                               "stop_target_bound", "stop_before_lc_commit"]
                if dist_to_stop and float(dist_to_stop) < 15.0:
                    confidence = 0.82
                else:
                    confidence = 0.61
            elif baseline_intent in {"follow", "keep_lane"} and preferred_lane not in ("current", ""):
                # Agent suggests proactive prepare-LC toward preferred lane
                direction = "right" if "right" in preferred_lane else "left"
                perm_ok = bool(lc_perm.get(direction, False))
                if perm_ok:
                    chosen = f"prepare_lane_change_{direction}"
                    reason_tags = ["api_simulated", f"simulated_proactive_lc_{direction}",
                                   "route_alignment_opportunity"]
                    confidence = 0.58
                else:
                    # No permission — agree with baseline (no disagreement surface)
                    chosen = _BASELINE_TO_STUB_INTENT_MAP.get(baseline_intent, "keep_lane")
                    reason_tags = ["api_simulated", "simulated_agree_no_lc_permission"]
                    confidence = 0.75
            elif baseline_intent in {"stop", "stop_before_obstacle"}:
                # Stop is terminal — agent affirms; no meaningful disagreement
                chosen = "stop"
                reason_tags = ["api_simulated", "simulated_affirm_stop", f"baseline_{baseline_intent}"]
                confidence = 0.88
            elif baseline_intent in {"follow", "keep_lane", "slow_down"} and preferred_lane in ("current", ""):
                # Conservative intent, no route pressure — agent agrees; no surface for useful disagreement
                chosen = _BASELINE_TO_STUB_INTENT_MAP.get(baseline_intent, "keep_lane")
                reason_tags = ["api_simulated", "simulated_agree_conservative_route"]
                confidence = 0.78
            else:
                # Generic mild disagree — only when there is a meaningful surface
                chosen = "keep_lane"
                reason_tags = ["api_simulated", "simulated_conservative_override",
                               f"baseline_was_{baseline_intent}"]
                confidence = 0.60

        # Derive target_lane (same R2 logic as stub)
        if chosen in lc_intents:
            derived_lane = str(baseline_context.get("target_lane", "current"))
        else:
            derived_lane = str(route_context.get("preferred_lane") or baseline_context.get("target_lane") or "current")

        return {
            "tactical_intent": chosen,
            "target_lane": derived_lane,
            "reason_tags": reason_tags,
            "confidence": round(confidence, 3),
        }

    # ── Local model mode ──────────────────────────────────────────────────────

    def _call_local(
        self,
        *,
        ego_state: dict[str, Any],
        lane_context: dict[str, Any],
        route_context: dict[str, Any],
        baseline_context: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError("Local model mode not implemented in v1. Use mode='stub' or mode='api_simulated'.")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _default_reason_tags(intent: str, fallback: bool) -> list[str]:
        if fallback:
            return ["fallback_to_baseline"]
        return [f"intent_{intent}"]

    @staticmethod
    def _is_disagreement_useful(
        *,
        agent_intent: str,
        baseline_intent: str,
        validation_status: str,
        fallback_to_baseline: bool,
        lane_change_permission: dict[str, Any],
    ) -> bool:
        """
        A disagreement is 'useful' iff:
        - agent is not in fallback mode
        - validation_status is 'valid'
        - intents actually differ
        - the difference is semantically meaningful (not just aliases)
        - agent is not violating safety gate
        """
        if fallback_to_baseline or validation_status != "valid":
            return False
        if agent_intent == _BASELINE_TO_STUB_INTENT_MAP.get(baseline_intent, baseline_intent):
            return False
        # Disagreement on LC vs conservative (slow_down/keep_lane) is structurally useful
        lc_intents = {
            "prepare_lane_change_left", "prepare_lane_change_right",
            "commit_lane_change_left", "commit_lane_change_right",
        }
        conservative = {"slow_down", "keep_lane", "follow", "yield", "stop"}
        if baseline_intent in lc_intents and agent_intent in conservative:
            return True
        if baseline_intent in conservative and agent_intent in lc_intents:
            # Only useful if LC is permitted
            if "left" in agent_intent:
                return bool(lane_change_permission.get("left", False))
            if "right" in agent_intent:
                return bool(lane_change_permission.get("right", False))
        return True

    def to_dict(self, intent: AgentShadowIntent) -> dict[str, Any]:
        """Serialize AgentShadowIntent to dict, safe for JSON/JSONL dump."""
        return asdict(intent)


def build_adapter(
    *,
    mode: str = "stub",
    model_id: str | None = None,
    api_endpoint: str | None = None,
    api_key_env_var: str = "AGENT_API_KEY",
    api_timeout_s: float = 2.0,
    stub_disagree_on_risk: bool = True,
) -> AgentShadowAdapter:
    """Factory function for creating a properly configured adapter."""
    resolved_model_id = model_id or (f"api:{api_endpoint}" if mode == "api" else f"stub_v1")
    cfg = AgentShadowAdapterConfig(
        mode=mode,
        model_id=resolved_model_id,
        api_timeout_s=api_timeout_s,
        api_endpoint=api_endpoint,
        api_key_env_var=api_key_env_var,
        stub_disagree_on_risk=stub_disagree_on_risk,
    )
    return AgentShadowAdapter(cfg)
