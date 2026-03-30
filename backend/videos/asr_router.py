"""
Quality-aware hybrid ASR router:
- English: Groq Whisper first
- Deepgram-supported multilingual languages: Deepgram first
- Malayalam / unsupported languages: explicit local fallback path
- Cloud and local engines can be retried on quality failure, not only exceptions
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import threading
from typing import Dict, Optional, List, Tuple

import ffmpeg
from django.conf import settings

from .audio_preprocessor import ChunkMetadata
from .deepgram_client import transcribe_prerecorded_audio, DeepgramError
from .language import (
    normalize_language_code,
    detect_script_type,
    candidate_languages_for_script,
)

logger = logging.getLogger(__name__)
_GROQ_FALLBACK_LOCK = threading.Lock()
_GROQ_FALLBACK_BLOCK_UNTIL = 0.0
_ASR_PROVIDER_PRIOR_CACHE: Dict[tuple[str, str], Dict[str, float]] = {}


def _render_demo_safe_asr_mode() -> bool:
    return bool(getattr(settings, "RENDER_DEMO_SAFE_ASR_MODE", False))


def _asr_max_retries() -> int:
    if _render_demo_safe_asr_mode():
        return max(1, int(getattr(settings, "RENDER_DEMO_SAFE_ASR_MAX_RETRIES", 1)))
    return max(1, int(getattr(settings, "ASR_MAX_RETRIES", 3)))


def _asr_reject_on_garble() -> bool:
    return bool(getattr(settings, "ASR_REJECT_ON_GARBLE", True))


def _deepgram_supported_languages() -> set:
    """
    Configurable Deepgram language allow-list.
    Malayalam (`ml`) is intentionally excluded as a hard policy.
    """
    raw = getattr(
        settings,
        "DEEPGRAM_SUPPORTED_LANGUAGES",
        "en,hi,ta,te,kn,es,fr,de,pt,it,nl,ru,ja,ko,zh,ar,tr,id,sv,no,pl,uk",
    )
    vals = [normalize_language_code(x, default="", allow_auto=False) for x in str(raw).split(",")]
    supported = {v for v in vals if v}
    supported.discard("ml")
    return supported


def _deepgram_malayalam_status(dg_supported: set) -> Dict[str, object]:
    enabled = bool(getattr(settings, "ASR_ENABLE_DEEPGRAM_MALAYALAM_EXPERIMENT", False))
    available = _deepgram_available()
    supported = "ml" in dg_supported
    if supported:
        reason = "feature_flag_enabled" if enabled else "configured_allow_list"
    elif enabled:
        reason = "policy_disabled_for_malayalam"
    elif not available:
        reason = "deepgram_unavailable"
    else:
        reason = "policy_disabled_for_malayalam"
    return {
        "enabled": enabled,
        "available": available,
        "supported": supported,
        "reason": reason,
    }


def _is_low_content_for_duration(text: str, duration_seconds: float) -> bool:
    """
    Flag suspiciously short transcripts relative to audio duration.
    """
    words = len((text or "").strip().split())
    if duration_seconds <= 0:
        return words < 10
    minutes = max(duration_seconds / 60.0, 0.1)
    wpm = words / minutes
    min_wpm = float(getattr(settings, "ASR_LOW_CONTENT_WPM_MIN", 20.0))
    long_min_words = int(getattr(settings, "ASR_LOW_CONTENT_MIN_WORDS_LONG", 80))
    if duration_seconds >= 120 and words < long_min_words:
        return True
    if duration_seconds >= 180 and wpm < min_wpm:
        return True
    return False


def _malayalam_asr_strategy() -> str:
    strategy = str(getattr(settings, "ASR_MALAYALAM_STRATEGY", "quality_first") or "quality_first").strip().lower()
    if strategy not in {"current_default", "fast_first", "quality_first", "hybrid_retry"}:
        return "quality_first"
    return strategy


def _assert_malayalam_provider_is_local_only(selected_provider: str) -> str:
    provider = str(selected_provider or "").strip().lower()
    assert provider not in {"groq", "groq_whisper", "deepgram"}, (
        "[ML_POLICY_VIOLATION] Malayalam transcript routing must stay local-only. "
        "Malayalam requires local Faster-Whisper for transcript fidelity. "
        "If you intend to change this, update _select_malayalam_strategy() explicitly "
        "and update the corresponding tests in tests_multilingual.py."
    )
    return selected_provider


def _assert_malayalam_provider_is_not_groq(selected_provider: str) -> str:
    """Backward-compatible alias for older tests/imports."""
    return _assert_malayalam_provider_is_local_only(selected_provider)


def _malayalam_model_override_enabled() -> bool:
    return bool(getattr(settings, "ASR_MALAYALAM_MODEL_OVERRIDE_ENABLED", False))


def _malayalam_model_override_name() -> str:
    return str(getattr(settings, "ASR_MALAYALAM_MODEL_OVERRIDE", "") or "").strip()


def _malayalam_second_pass_enabled() -> bool:
    return bool(getattr(settings, "ASR_MALAYALAM_SECOND_PASS_ENABLED", True))


def _get_malayalam_model() -> str:
    """Single source of truth for Malayalam model selection."""
    return str(getattr(settings, "ASR_MALAYALAM_PRIMARY_MODEL", "") or "").strip() or "large-v3"


def _get_malayalam_retry_model() -> str:
    """Retry and specialist paths use the same model as primary."""
    return _get_malayalam_model()


def _malayalam_second_pass_model() -> str:
    configured = str(getattr(settings, "ASR_MALAYALAM_SECOND_PASS_MODEL", "") or "").strip()
    if configured:
        return configured
    return _get_malayalam_retry_model()


def _malayalam_specialist_recovery_model() -> str:
    configured = str(getattr(settings, "ASR_MALAYALAM_SPECIALIST_MODEL", "") or "").strip()
    if configured:
        return configured
    return _malayalam_second_pass_model()


def _describe_audio_preprocess(language_hint: str, duration_seconds: float = 0.0) -> Dict[str, object]:
    normalized_lang = normalize_language_code(language_hint, default="", allow_auto=False)
    preprocess_steps = ["pcm_s16le_16k_mono"]
    chunk_policy_used = "standard_single_stream"
    if normalized_lang == "ml":
        if bool(getattr(settings, "ASR_MALAYALAM_ENABLE_SILENCE_TRIM", False)):
            preprocess_steps.append("trim_leading_silence")
        chunk_policy_used = (
            "malayalam_bounded_longform_single_stream"
            if float(duration_seconds or 0.0) >= float(getattr(settings, "ASR_MALAYALAM_LONGFORM_SECONDS", 480) or 480.0)
            else "malayalam_speech_focused_single_stream"
        )
    return {
        "audio_preprocess_applied": len(preprocess_steps) > 1,
        "preprocess_steps": preprocess_steps,
        "chunk_policy_used": chunk_policy_used,
    }


def _preprocess_audio_for_asr(audio_path: str, language_hint: str = "", already_preprocessed: bool = False) -> str:
    """
    Preprocess to PCM WAV 16k mono before ASR.
    """
    normalized_lang = normalize_language_code(language_hint, default="", allow_auto=False)
    if already_preprocessed:
        return audio_path
    steps = _describe_audio_preprocess(normalized_lang)
    filters: List[str] = []
    if normalized_lang == "ml":
        if "trim_leading_silence" in steps["preprocess_steps"]:
            filters.append("silenceremove=start_periods=1:start_silence=0.3:start_threshold=-42dB")
        logger.info(
            "[ML_AUDIO_PREPROCESS] language=%s applied=%s steps=%s",
            normalized_lang or "unknown",
            bool(steps["audio_preprocess_applied"]),
            ",".join(steps["preprocess_steps"]) or "none",
        )
        logger.info(
            "[ML_AUDIO_CHUNK_POLICY] language=%s chunk_policy=%s",
            normalized_lang or "unknown",
            steps["chunk_policy_used"],
        )
    out_path = tempfile.mktemp(suffix=".wav")
    stream = ffmpeg.input(audio_path)
    output_kwargs = {
        "ar": 16000,
        "ac": 1,
        "acodec": "pcm_s16le",
    }
    if filters:
        output_kwargs["af"] = ",".join(filters)
    stream.output(out_path, **output_kwargs).overwrite_output().run(quiet=True, capture_stdout=True, capture_stderr=True)
    return out_path


def _audio_duration_seconds(audio_path: str) -> float:
    try:
        probe = ffmpeg.probe(audio_path)
        return float(probe["format"]["duration"])
    except Exception:
        return 0.0


def _audio_file_size_bytes(audio_path: str) -> int:
    try:
        return int(os.path.getsize(audio_path))
    except Exception:
        return 0


def _quality_score(payload: Dict) -> float:
    text = (payload.get("text") or "").strip()
    conf = float(payload.get("confidence", payload.get("language_probability", 0.0)) or 0.0)
    words = len(text.split()) if text else 0
    score = min(1.0, max(0.0, conf + (0.15 if words >= 40 else 0.05 if words >= 12 else 0.0)))
    return round(score, 4)


def _with_engine_metadata(
    payload: Dict,
    engine: str,
    req_lang: str,
    duration: float,
    elapsed: float,
    route_reason: str = "",
    fallback_triggered: bool = False,
    fallback_reason: str = "",
    quality_gate_passed: bool = True,
    route_decision: Optional[Dict[str, object]] = None,
) -> Dict:
    meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    meta["asr_provider_used"] = engine
    meta["asr_engine_used"] = engine
    meta["provider_name"] = _provider_name_for_engine(engine)
    meta["requested_transcription_language"] = req_lang
    meta["asr_route_reason"] = route_reason
    meta["fallback_triggered"] = bool(fallback_triggered)
    meta["fallback_reason"] = fallback_reason or ""
    meta["rtf"] = (elapsed / duration) if duration > 0 else 0.0
    meta["latency_seconds"] = round(float(elapsed or 0.0), 3)
    meta["transcript_quality_gate_passed"] = bool(quality_gate_passed)
    if route_decision:
        meta["selected_model"] = route_decision.get("model", "")
        meta["fallback_chain"] = route_decision.get("fallback_chain", [])
        meta["latency_budget_seconds"] = float(route_decision.get("latency_budget_seconds", 0.0) or 0.0)
        meta["route_score"] = float(route_decision.get("score", 0.0) or 0.0)
        meta["detection_confidence"] = float(route_decision.get("detection_confidence", 0.0) or 0.0)
        meta["file_size_bytes"] = int(route_decision.get("file_size_bytes", 0) or 0)
    payload["metadata"] = meta
    payload["asr_engine_used"] = engine
    payload["transcript_quality_score"] = _quality_score(payload)
    return payload


def _provider_name_for_engine(engine: str) -> str:
    return {
        "groq_whisper": "groq",
        "deepgram": "deepgram",
        "whisper_local": "faster_whisper_local",
    }.get(engine, engine)


def _transcribe_with_local_whisper(audio_path: str, source_type: str, lang: str) -> Dict:
    # Lazy import to avoid circular import at module import time.
    from . import utils as videos_utils
    return videos_utils._transcribe_with_faster_whisper(  # pylint: disable=protected-access
        audio_path=audio_path,
        source_type=source_type,
        transcription_language=lang
    )


def _transcribe_with_groq_whisper(audio_path: str, source_type: str, lang: str) -> Dict:
    # Lazy import to avoid circular import at module import time.
    from . import utils as videos_utils
    return videos_utils._transcribe_with_groq(  # pylint: disable=protected-access
        audio_path=audio_path,
        source_type=source_type,
        transcription_language=lang
    )


def _transcribe_with_malayalam_local(audio_path: str, source_type: str) -> Dict:
    # Lazy import to avoid circular import at module import time.
    from . import utils as videos_utils
    return videos_utils._transcribe_with_faster_whisper_model(  # pylint: disable=protected-access
        audio_path=audio_path,
        source_type=source_type,
        transcription_language="ml",
        model_size=_get_malayalam_model(),
        allow_force_large_v3=False,
    )


def _asr_use_groq_fallback() -> bool:
    if not (
        bool(getattr(settings, "ASR_USE_GROQ_FALLBACK", True))
        and bool(getattr(settings, "USE_GROQ_WHISPER", True))
        and bool(getattr(settings, "GROQ_API_KEY", ""))
    ):
        return False
    now = time.time()
    with _GROQ_FALLBACK_LOCK:
        return now >= _GROQ_FALLBACK_BLOCK_UNTIL


def _is_rate_limit_error(err: Exception) -> bool:
    msg = str(err).lower()
    return (
        "429" in msg
        or "rate limit" in msg
        or "rate_limit_exceeded" in msg
        or "tokens per day" in msg
        or "tokens per minute" in msg
        or "too many requests" in msg
    )


def _is_cuda_oom_error(err: Exception) -> bool:
    msg = str(err or "").lower()
    return "cuda failed with error out of memory" in msg or "cuda out of memory" in msg


def _block_groq_fallback_due_to_rate_limit() -> None:
    cooldown_sec = int(getattr(settings, "ASR_GROQ_COOLDOWN_SEC", 900))
    with _GROQ_FALLBACK_LOCK:
        global _GROQ_FALLBACK_BLOCK_UNTIL
        _GROQ_FALLBACK_BLOCK_UNTIL = max(_GROQ_FALLBACK_BLOCK_UNTIL, time.time() + cooldown_sec)
    logger.warning(
        "Groq ASR fallback temporarily disabled for %ss due to rate-limit response.",
        cooldown_sec,
    )


def _is_garbled(payload: Dict) -> bool:
    from . import utils as videos_utils
    return videos_utils._looks_garbled_multiscript(payload.get("text", ""))  # pylint: disable=protected-access


def _detect_lang_for_router(audio_path: str) -> str:
    from . import utils as videos_utils
    return videos_utils._detect_audio_language(audio_path)  # pylint: disable=protected-access


def _detect_lang_for_router_with_confidence(audio_path: str) -> tuple[str, float]:
    from . import utils as videos_utils
    return videos_utils._detect_audio_language_with_confidence(audio_path)  # pylint: disable=protected-access


def _selected_model_for_engine(engine: str, language: str) -> str:
    if engine == "groq_whisper":
        if _is_malayalam(language):
            return "whisper-large-v3"
        return str(getattr(settings, "GROQ_WHISPER_MODEL", "whisper-large-v3-turbo"))
    if engine == "deepgram":
        return str(getattr(settings, "DEEPGRAM_MODEL", "nova-2"))
    if _is_malayalam(language):
        return "large-v3"
    return str(getattr(settings, "WHISPER_MODEL_SIZE", "large-v3"))


def _provider_prior_score(language: str, engine: str) -> float:
    cached = _ASR_PROVIDER_PRIOR_CACHE.get((language, engine)) or {}
    if "quality_score" in cached:
        return float(cached.get("quality_score", 0.0) or 0.0)
    return float(getattr(settings, "ASR_PROVIDER_PRIOR_DEFAULT_QUALITY", 0.72))


def _candidate_score(
    *,
    engine: str,
    language: str,
    duration_seconds: float,
    file_size_bytes: int,
    detection_confidence: float,
    latency_budget_seconds: float,
) -> float:
    quality_prior = _provider_prior_score(language, engine)
    score = quality_prior
    if engine == "groq_whisper":
        score += 0.18
    elif engine == "deepgram":
        score += 0.14
    else:
        score += 0.06

    if duration_seconds > latency_budget_seconds and engine == "whisper_local" and not _is_malayalam(language):
        score -= 0.22
    if duration_seconds > max(latency_budget_seconds, 600) and engine == "groq_whisper":
        score -= 0.06
    if file_size_bytes > int(getattr(settings, "GROQ_WHISPER_MAX_SINGLE_FILE_MB", 23) * 1024 * 1024) and engine == "groq_whisper":
        score -= 0.08
    if detection_confidence < 0.5 and engine != "whisper_local":
        score -= 0.04
    return round(score, 4)


def _call_deepgram_with_retries(
    audio_path: str,
    language: Optional[str],
    detect_language: bool,
) -> Dict:
    max_attempts = _asr_max_retries()
    for attempt in range(1, max_attempts + 1):
        try:
            return transcribe_prerecorded_audio(
                audio_path,
                language=language,
                detect_language=detect_language,
            )
        except DeepgramError:
            if attempt >= max_attempts:
                raise
            delay = min(1.5 * (2 ** (attempt - 1)), 8.0)
            logger.warning(
                "Deepgram ASR retry in %.1fs (attempt %d/%d) "
                "[language=%s detect_language=%s]",
                delay,
                attempt,
                max_attempts,
                language or "auto",
                detect_language,
            )
            time.sleep(delay)

    raise RuntimeError("Deepgram ASR retries exhausted without returning payload")


def _longform_threshold_seconds() -> int:
    return max(60, int(getattr(settings, "ASR_LONGFORM_LOCAL_THRESHOLD_SECONDS", 900)))


def _use_deepgram_for_non_english() -> bool:
    return bool(getattr(settings, "ASR_USE_DEEPGRAM_FOR_NON_ENGLISH", False))


def _deepgram_available() -> bool:
    return _use_deepgram_for_non_english() and bool(getattr(settings, "DEEPGRAM_API_KEY", ""))


def _is_malayalam(lang: str) -> bool:
    return normalize_language_code(lang, default="", allow_auto=False) == "ml"


def _render_demo_safe_transcription_error(language: str, reason: str = "") -> RuntimeError:
    normalized_lang = normalize_language_code(language, default="auto", allow_auto=True)
    if normalized_lang == "ml":
        detail = "Malayalam transcription is disabled on the live demo because it requires heavier local ASR."
    elif normalized_lang not in {"auto", "en"} and normalized_lang not in _deepgram_supported_languages():
        detail = (
            f"Language '{normalized_lang}' is not supported on the current live demo host. "
            "Please try English or a shorter supported clip."
        )
    else:
        detail = (
            "Live demo transcription is temporarily limited on this host. "
            "Please try a shorter video or use the local/full deployment for heavier transcription."
        )
    if reason:
        detail = f"{detail} ({reason})"
    return RuntimeError(detail)


def _choose_demo_safe_engine(
    requested_lang: str,
    chosen_lang: str,
    duration_seconds: float,
    file_size_bytes: int,
    detection_confidence: float,
) -> Dict[str, object]:
    lang = normalize_language_code(chosen_lang, default="auto", allow_auto=True)
    if requested_lang != "auto":
        lang = normalize_language_code(requested_lang, default=lang, allow_auto=False)

    has_groq = bool(getattr(settings, "USE_GROQ_WHISPER", True)) and bool(getattr(settings, "GROQ_API_KEY", ""))
    has_deepgram = _deepgram_available()
    deepgram_supported = _deepgram_supported_languages()
    latency_budget_seconds = float(getattr(settings, "ASR_LATENCY_BUDGET_SECONDS", 480))

    if not has_groq and not has_deepgram:
        raise _render_demo_safe_transcription_error(lang, reason="no_remote_asr_provider_configured")

    if lang == "ml":
        raise _render_demo_safe_transcription_error(lang, reason="render_demo_safe_remote_only")

    if lang == "auto":
        if has_deepgram:
            engine = "deepgram"
            reason = "render_demo_safe_auto_remote"
        else:
            engine = "groq_whisper"
            reason = "render_demo_safe_auto_groq"
    elif lang == "en":
        if has_groq:
            engine = "groq_whisper"
            reason = "render_demo_safe_english_groq"
        elif has_deepgram:
            engine = "deepgram"
            reason = "render_demo_safe_english_deepgram"
        else:
            raise _render_demo_safe_transcription_error(lang)
    elif lang in deepgram_supported and has_deepgram:
        engine = "deepgram"
        reason = "render_demo_safe_supported_deepgram"
    else:
        raise _render_demo_safe_transcription_error(lang, reason="unsupported_language_remote_only")

    fallback_chain: List[str] = []
    if engine == "deepgram" and has_groq:
        fallback_chain.append("groq_whisper")
    elif engine == "groq_whisper" and has_deepgram:
        fallback_chain.append("deepgram")

    scoring_lang = lang if lang != "auto" else "en"
    return {
        "engine": engine,
        "model": _selected_model_for_engine(engine, scoring_lang),
        "reason": reason,
        "fallback_chain": fallback_chain,
        "latency_budget_seconds": latency_budget_seconds,
        "detection_confidence": round(float(detection_confidence or 0.0), 4),
        "file_size_bytes": int(file_size_bytes or 0),
        "score": round(
            _candidate_score(
                engine=engine,
                language=scoring_lang,
                duration_seconds=duration_seconds,
                file_size_bytes=file_size_bytes,
                detection_confidence=detection_confidence,
                latency_budget_seconds=latency_budget_seconds,
            ),
            4,
        ),
        "render_demo_safe_remote_only": True,
    }


def _run_malayalam_local_model(audio_path: str, source_type: str, model_name: str) -> Dict:
    from . import utils as videos_utils
    payload = videos_utils._transcribe_with_faster_whisper_model(  # pylint: disable=protected-access
        audio_path=audio_path,
        source_type=source_type,
        transcription_language="ml",
        model_size=model_name,
        allow_force_large_v3=False,
    )
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    metadata["actual_local_model_name"] = metadata.get("actual_local_model_name") or model_name
    payload["metadata"] = metadata
    return payload


def _select_malayalam_strategy(
    *,
    duration_seconds: float,
    file_size_bytes: int,
    detection_confidence: float,
) -> Dict[str, object]:
    strategy = _malayalam_asr_strategy()
    override_enabled = _malayalam_model_override_enabled()
    override_model = _malayalam_model_override_name()
    fast_primary_model = str(getattr(settings, "ASR_MALAYALAM_FAST_PRIMARY_MODEL", "") or "").strip()
    longform_threshold_seconds = float(getattr(settings, "ASR_MALAYALAM_LONGFORM_SECONDS", 480) or 480.0)
    second_pass_model = _malayalam_second_pass_model()
    primary_model = override_model or _get_malayalam_model()
    strategy_reason = "forced_local_quality_path"
    speed_optimized_for_longform = False
    if duration_seconds >= longform_threshold_seconds and fast_primary_model:
        primary_model = fast_primary_model
        strategy_reason = "prefer_local_longform_fast_path"
        speed_optimized_for_longform = True
    route = {
        "strategy": strategy,
        "strategy_reason": strategy_reason,
        "primary_engine": "whisper_local",
        "primary_model": primary_model,
        "fallback_chain": [],
        "retry_model": second_pass_model,
        "malayalam_primary_model_override": "",
        "second_pass_enabled": _malayalam_second_pass_enabled(),
        "speed_optimized_for_longform": speed_optimized_for_longform,
        "latency_budget_seconds": float(getattr(settings, "ASR_LATENCY_BUDGET_SECONDS", 480)),
        "detection_confidence": round(float(detection_confidence or 0.0), 4),
        "file_size_bytes": int(file_size_bytes or 0),
        "score": _candidate_score(
            engine="whisper_local",
            language="ml",
            duration_seconds=duration_seconds,
            file_size_bytes=file_size_bytes,
            detection_confidence=detection_confidence,
            latency_budget_seconds=float(getattr(settings, "ASR_LATENCY_BUDGET_SECONDS", 480)),
        ),
    }
    route["primary_engine"] = _assert_malayalam_provider_is_local_only(route["primary_engine"])
    if speed_optimized_for_longform:
        route["malayalam_primary_model_override"] = primary_model
    elif strategy == "fast_first":
        route["strategy_reason"] = "forced_local_quality_path_fast_first_ignored_for_malayalam"
    elif strategy == "quality_first":
        route["malayalam_primary_model_override"] = override_model if override_enabled and override_model else ""
        route["strategy_reason"] = "prefer_local_quality_path"
    elif strategy == "hybrid_retry":
        route["strategy_reason"] = "bounded_hybrid_retry_enabled_local_primary"
    else:
        route["strategy_reason"] = "forced_local_quality_path"
    logger.info(
        "[ML_ASR_STRATEGY] strategy=%s primary_engine=%s primary_model=%s retry_model=%s reason=%s",
        route["strategy"],
        route["primary_engine"],
        route["primary_model"],
        route["retry_model"],
        route["strategy_reason"],
    )
    return route


def _choose_primary_engine(
    requested_lang: str,
    chosen_lang: str,
    duration_seconds: float,
    deepgram_supported: set,
    file_size_bytes: int = 0,
    detection_confidence: float = 0.0,
) -> Dict[str, object]:
    """Return a quality-aware route decision."""
    lang = normalize_language_code(chosen_lang, default="auto", allow_auto=True)
    if requested_lang != "auto":
        lang = normalize_language_code(requested_lang, default=lang, allow_auto=False)

    latency_budget_seconds = float(getattr(settings, "ASR_LATENCY_BUDGET_SECONDS", 480))
    candidates: List[tuple[str, str]] = []
    fallback_chain: List[str] = []

    if lang == "en":
        if bool(getattr(settings, "USE_GROQ_WHISPER", True)) and bool(getattr(settings, "GROQ_API_KEY", "")):
            candidates.append(("groq_whisper", "english_fast_cloud"))
            fallback_chain.append("whisper_local")
        else:
            candidates.append(("whisper_local", "english_local_fallback"))
        if not candidates:
            candidates.append(("whisper_local", "english_local_fallback"))
    elif _is_malayalam(lang):
        ml_route = _select_malayalam_strategy(
            duration_seconds=duration_seconds,
            file_size_bytes=file_size_bytes,
            detection_confidence=detection_confidence,
        )
        return {
            "engine": str(ml_route.get("primary_engine", "groq_whisper")),
            "model": str(ml_route.get("primary_model", "whisper-large-v3")),
            "reason": "malayalam_groq_primary" if str(ml_route.get("primary_engine")) == "groq_whisper" else "malayalam_quality_primary",
            "fallback_chain": list(ml_route.get("fallback_chain", []) or ["whisper_local"]),
            "latency_budget_seconds": float(ml_route.get("latency_budget_seconds", latency_budget_seconds) or latency_budget_seconds),
            "detection_confidence": round(float(detection_confidence or 0.0), 4),
            "file_size_bytes": int(file_size_bytes or 0),
            "score": round(float(ml_route.get("score", 0.0) or 0.0), 4),
            "malayalam_asr_strategy": str(ml_route.get("strategy", "current_default")),
            "asr_strategy_reason": str(ml_route.get("strategy_reason", "") or ""),
            "retry_model": str(ml_route.get("retry_model", "") or ""),
            "malayalam_primary_model_override": str(ml_route.get("malayalam_primary_model_override", "") or ""),
            "second_pass_enabled": bool(ml_route.get("second_pass_enabled", False)),
            "speed_optimized_for_longform": bool(ml_route.get("speed_optimized_for_longform", False)),
        }
    elif _deepgram_available() and lang in deepgram_supported:
        candidates.append(("deepgram", "deepgram_supported_language"))
        fallback_chain.extend(["whisper_local", "groq_whisper"])
    else:
        reason = "unsupported_language_local_only"
        if duration_seconds > _longform_threshold_seconds():
            reason = "duration_long_local"
        candidates.append(("whisper_local", reason))

    best_engine, best_reason = candidates[0]
    best_score = -1.0
    for engine, reason in candidates:
        score = _candidate_score(
            engine=engine,
            language=lang,
            duration_seconds=duration_seconds,
            file_size_bytes=file_size_bytes,
            detection_confidence=detection_confidence,
            latency_budget_seconds=latency_budget_seconds,
        )
        if score > best_score:
            best_score = score
            best_engine, best_reason = engine, reason

    return {
        "engine": best_engine,
        "model": _selected_model_for_engine(best_engine, lang),
        "reason": best_reason,
        "fallback_chain": fallback_chain,
        "latency_budget_seconds": latency_budget_seconds,
        "detection_confidence": round(float(detection_confidence or 0.0), 4),
        "file_size_bytes": int(file_size_bytes or 0),
        "score": round(best_score, 4),
    }


def _analyze_malayalam_asr_payload(payload: Dict) -> Dict[str, object]:
    from . import utils as videos_utils

    segments = list(payload.get("segments", []) or [])
    text = str(payload.get("text", "") or "")
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    metrics = videos_utils._extract_asr_metrics(text, payload.get("word_timestamps", []) or [])  # pylint: disable=protected-access
    trust = videos_utils.build_malayalam_transcript_trust(segments)  # pylint: disable=protected-access
    first_pass = videos_utils._should_accept_usable_malayalam_first_pass(payload)  # pylint: disable=protected-access

    trusted_visible_word_count = 0
    trusted_display_unit_count = 0
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        classified = videos_utils.classify_malayalam_segment_type(str(seg.get("text", "") or ""))  # pylint: disable=protected-access
        score = classified.get("score", {}) if isinstance(classified.get("score"), dict) else {}
        seg_type = str(classified.get("type", "") or "")
        if seg_type == "clean_malayalam" or (
            seg_type == "mixed_malayalam_english"
            and float(score.get("score", 0.0) or 0.0) >= 0.34
            and float(score.get("wrong_script_ratio", 0.0) or 0.0) < 0.18
        ):
            trusted_display_unit_count += 1
            trusted_visible_word_count += len(str(seg.get("text", "") or "").split())

    total_segments = max(int(trust.get("total_segments", 0) or 0), 1)
    wrong_script_burden = round(float(trust.get("wrong_script_segments", 0) or 0.0) / float(total_segments), 4)
    contamination_burden = round(
        max(
            float(metrics.get("other_indic_ratio", 0.0) or 0.0),
            1.0 - float(trust.get("malayalam_token_coverage", 0.0) or 0.0),
        ),
        4,
    )
    lexical_trust = float(trust.get("lexical_trust_score", 0.0) or 0.0)
    readability = float(trust.get("overall_readability", 0.0) or 0.0)
    garble = float(first_pass.get("garbled_detector_score", 0.0) or 0.0)
    dominant_script_final = str(trust.get("dominant_script_final", "") or "")
    detected_language = normalize_language_code(
        payload.get("language", metadata.get("detected_language", "")),
        default="",
        allow_auto=False,
    )
    detected_language_confidence = float(
        payload.get("language_probability", payload.get("confidence", metadata.get("detected_language_confidence", 0.0))) or 0.0
    )
    word_count = len((text or "").split())
    fidelity = videos_utils.analyze_malayalam_source_fidelity(
        segments,
        detected_language=detected_language,
        detected_language_confidence=detected_language_confidence,
        dominant_script_final=dominant_script_final,
        script_detector_result=dominant_script_final,
        lexical_trust_score=lexical_trust,
        overall_readability=readability,
        trusted_visible_word_count=trusted_visible_word_count,
        trusted_display_unit_count=trusted_display_unit_count,
    )
    visible_malayalam_char_ratio = float(fidelity.get("visible_malayalam_char_ratio", 0.0) or 0.0)
    script_confusion_candidate = (
        detected_language == "ml"
        and detected_language_confidence >= float(getattr(settings, "ASR_MALAYALAM_CONFUSION_RETRY_MIN_CONFIDENCE", 0.72) or 0.72)
        and dominant_script_final == "other"
        and word_count >= int(getattr(settings, "ASR_MALAYALAM_CONFUSION_RETRY_MIN_WORDS", 3) or 3)
        and max(lexical_trust, readability) >= float(getattr(settings, "ASR_MALAYALAM_CONFUSION_RETRY_MIN_SIGNAL", 0.05) or 0.05)
        and wrong_script_burden >= float(getattr(settings, "ASR_MALAYALAM_CONFUSION_RETRY_MIN_WRONG_SCRIPT", 0.22) or 0.22)
        and contamination_burden >= float(getattr(settings, "ASR_MALAYALAM_CONFUSION_RETRY_MIN_CONTAMINATION", 0.40) or 0.40)
        and garble < 0.95
    )
    high_garble_confusion_burden = (
        contamination_burden >= float(getattr(settings, "ASR_MALAYALAM_HIGH_GARBLE_RETRY_MIN_CONTAMINATION", 0.35) or 0.35)
        or wrong_script_burden >= float(getattr(settings, "ASR_MALAYALAM_HIGH_GARBLE_RETRY_MIN_WRONG_SCRIPT", 0.14) or 0.14)
        or (
            trusted_visible_word_count == 0
            and trusted_display_unit_count == 0
            and word_count >= int(getattr(settings, "ASR_MALAYALAM_HIGH_GARBLE_RETRY_MIN_WORDS", 4) or 4)
        )
    )
    high_garble_confusion_candidate = (
        detected_language == "ml"
        and detected_language_confidence >= float(getattr(settings, "ASR_MALAYALAM_HIGH_GARBLE_RETRY_MIN_CONFIDENCE", 0.88) or 0.88)
        and dominant_script_final == "other"
        and garble >= float(getattr(settings, "ASR_MALAYALAM_HIGH_GARBLE_RETRY_MIN_GARBLE", 0.48) or 0.48)
        and garble < 0.95
        and word_count >= int(getattr(settings, "ASR_MALAYALAM_HIGH_GARBLE_RETRY_MIN_WORDS", 4) or 4)
        and max(lexical_trust, readability) >= float(getattr(settings, "ASR_MALAYALAM_HIGH_GARBLE_RETRY_MIN_SIGNAL", 0.04) or 0.04)
        and high_garble_confusion_burden
    )
    zero_trusted_evidence_other_script_candidate = (
        detected_language == "ml"
        and detected_language_confidence >= float(getattr(settings, "ASR_MALAYALAM_OTHER_SCRIPT_RETRY_MIN_CONFIDENCE", 0.90) or 0.90)
        and dominant_script_final == "other"
        and trusted_visible_word_count == 0
        and trusted_display_unit_count == 0
        and word_count >= int(getattr(settings, "ASR_MALAYALAM_OTHER_SCRIPT_RETRY_MIN_WORDS", 4) or 4)
        and lexical_trust >= float(getattr(settings, "ASR_MALAYALAM_OTHER_SCRIPT_RETRY_MIN_LEXICAL_TRUST", 0.04) or 0.04)
        and readability >= float(getattr(settings, "ASR_MALAYALAM_OTHER_SCRIPT_RETRY_MIN_READABILITY", 0.10) or 0.10)
        and garble >= float(getattr(settings, "ASR_MALAYALAM_OTHER_SCRIPT_RETRY_MIN_GARBLE", 0.08) or 0.08)
        and garble < 0.82
    )
    full_clip_fidelity_retry_candidate = (
        detected_language == "ml"
        and detected_language_confidence >= float(getattr(settings, "ASR_MALAYALAM_FULL_CLIP_RETRY_MIN_CONFIDENCE", 0.90) or 0.90)
        and bool(fidelity.get("source_language_fidelity_failed", False))
        and word_count >= int(getattr(settings, "ASR_MALAYALAM_FULL_CLIP_RETRY_MIN_WORDS", 4) or 4)
        and max(lexical_trust, readability) >= float(getattr(settings, "ASR_MALAYALAM_FULL_CLIP_RETRY_MIN_SIGNAL", 0.04) or 0.04)
        and garble < 0.82
    )
    confusion_candidate = (
        script_confusion_candidate
        or high_garble_confusion_candidate
        or zero_trusted_evidence_other_script_candidate
        or full_clip_fidelity_retry_candidate
    )
    quality_class = "recoverable_but_weak"
    reason = "bounded_second_pass_candidate"
    low_trust_malayalam_script_candidate = (
        detected_language == "ml"
        and dominant_script_final == "malayalam"
        and visible_malayalam_char_ratio >= float(
            getattr(settings, "ASR_MALAYALAM_LOW_TRUST_SCRIPT_RETRY_MIN_RATIO", 0.52) or 0.52
        )
        and word_count >= int(getattr(settings, "ASR_MALAYALAM_LOW_TRUST_SCRIPT_RETRY_MIN_WORDS", 4) or 4)
        and garble < float(getattr(settings, "ASR_MALAYALAM_LOW_TRUST_SCRIPT_RETRY_MAX_GARBLE", 0.45) or 0.45)
        and trusted_visible_word_count == 0
        and trusted_display_unit_count == 0
        and max(lexical_trust, readability) >= float(
            getattr(settings, "ASR_MALAYALAM_LOW_TRUST_SCRIPT_RETRY_MIN_SIGNAL", 0.04) or 0.04
        )
    )
    if bool(first_pass.get("first_pass_accepted", False)) or (
        trusted_visible_word_count >= 10 and readability >= 0.34 and lexical_trust >= 0.40 and wrong_script_burden <= 0.18
    ):
        quality_class = "clearly_good"
        reason = "first_pass_acceptable"
    elif (
        wrong_script_burden >= 0.45
        or garble >= 0.82
        or (
            trusted_visible_word_count == 0
            and trusted_display_unit_count == 0
            and lexical_trust <= 0.16
            and readability <= 0.18
        )
    ):
        if confusion_candidate or low_trust_malayalam_script_candidate:
            quality_class = "recoverable_but_weak"
            if high_garble_confusion_candidate:
                reason = "high_garble_malayalam_confusion_retry_candidate"
            elif script_confusion_candidate:
                reason = "malayalam_mixed_script_confusion_retry_candidate"
            elif full_clip_fidelity_retry_candidate:
                reason = "malayalam_full_clip_fidelity_retry_candidate"
            elif low_trust_malayalam_script_candidate:
                reason = "low_trust_malayalam_script_retry_candidate"
            else:
                reason = "dominant_script_other_zero_trusted_evidence_retry_candidate"
        else:
            quality_class = "clearly_hopeless"
            reason = "hopeless_quality_profile"
    return {
        "quality_class": quality_class,
        "reason": reason,
        "lexical_trust": lexical_trust,
        "readability": readability,
        "wrong_script_burden": wrong_script_burden,
        "contamination_burden": contamination_burden,
        "trusted_visible_word_count": trusted_visible_word_count,
        "trusted_display_unit_count": trusted_display_unit_count,
        "first_pass_accepted": bool(first_pass.get("first_pass_accepted", False)),
        "first_pass_accept_reason": str(first_pass.get("first_pass_accept_reason", "") or ""),
        "quality_score": float(first_pass.get("quality_score", 0.0) or 0.0),
        "dominant_script_final": dominant_script_final,
        "detected_language": detected_language,
        "detected_language_confidence": detected_language_confidence,
        "confusion_candidate": confusion_candidate,
        "script_confusion_candidate": script_confusion_candidate,
        "high_garble_confusion_candidate": high_garble_confusion_candidate,
        "zero_trusted_evidence_other_script_candidate": zero_trusted_evidence_other_script_candidate,
        "full_clip_fidelity_retry_candidate": full_clip_fidelity_retry_candidate,
        "source_language_fidelity_failed": bool(fidelity.get("source_language_fidelity_failed", False)),
        "transcript_fidelity_state": str(fidelity.get("transcript_fidelity_state", "") or ""),
        "catastrophic_wrong_script_failure": bool(fidelity.get("catastrophic_wrong_script_failure", False)),
        "recoverable_malayalam_fidelity_gap": bool(fidelity.get("recoverable_malayalam_fidelity_gap", False)),
        "suspicious_substitution_burden": float(fidelity.get("suspicious_substitution_burden", 0.0) or 0.0),
        "dominant_non_malayalam_visible_ratio": float(fidelity.get("dominant_non_malayalam_visible_ratio", 0.0) or 0.0),
        "visible_malayalam_char_ratio": float(fidelity.get("visible_malayalam_char_ratio", 0.0) or 0.0),
        "nonempty_signal": bool(fidelity.get("nonempty_signal", False)),
        "word_count": word_count,
        "garble_score": garble,
        "visible_malayalam_char_ratio": visible_malayalam_char_ratio,
        "low_trust_malayalam_script_candidate": low_trust_malayalam_script_candidate,
    }


def _should_attempt_malayalam_second_pass(payload: Dict, route_decision: Dict[str, object]) -> Dict[str, object]:
    analysis = _analyze_malayalam_asr_payload(payload)
    strategy = str(route_decision.get("malayalam_asr_strategy", "current_default") or "current_default")
    second_pass_enabled = bool(route_decision.get("second_pass_enabled", False))
    decision = {
        "attempt_second_pass": False,
        "reason": "",
        "blocked_reason": "",
        "analysis": analysis,
    }
    if strategy not in {"current_default", "hybrid_retry", "quality_first"}:
        decision["blocked_reason"] = "strategy_not_second_pass_enabled"
        return decision
    if not second_pass_enabled:
        decision["blocked_reason"] = "second_pass_disabled"
        return decision
    if analysis["quality_class"] == "clearly_good":
        decision["blocked_reason"] = "primary_result_already_good"
        return decision
    if analysis["quality_class"] == "clearly_hopeless":
        decision["blocked_reason"] = "hopeless_malayalam_skips_second_pass"
        return decision
    if (
        float(analysis.get("visible_malayalam_char_ratio", 0.0) or 0.0) <= 0.0
        and str(analysis.get("dominant_script_final", "") or "") == "latin"
        and int(analysis.get("trusted_visible_word_count", 0) or 0) == 0
    ):
        decision["blocked_reason"] = "zero_malayalam_evidence_skips_second_pass"
        return decision
    decision["attempt_second_pass"] = True
    decision["reason"] = str(analysis.get("reason", "") or "bounded_second_pass_candidate")
    return decision


def _is_malayalam_second_pass_better(primary_analysis: Dict[str, object], retry_analysis: Dict[str, object]) -> bool:
    primary_fidelity_failed = bool(primary_analysis.get("source_language_fidelity_failed", False))
    retry_fidelity_failed = bool(retry_analysis.get("source_language_fidelity_failed", False))
    primary_state = str(primary_analysis.get("transcript_fidelity_state", "") or "")
    retry_state = str(retry_analysis.get("transcript_fidelity_state", "") or "")
    primary_visible = int(primary_analysis.get("trusted_visible_word_count", 0) or 0)
    retry_visible = int(retry_analysis.get("trusted_visible_word_count", 0) or 0)
    primary_display = int(primary_analysis.get("trusted_display_unit_count", 0) or 0)
    retry_display = int(retry_analysis.get("trusted_display_unit_count", 0) or 0)
    primary_wrong_script = float(primary_analysis.get("wrong_script_burden", 0.0) or 0.0)
    retry_wrong_script = float(retry_analysis.get("wrong_script_burden", 0.0) or 0.0)
    primary_substitution = float(primary_analysis.get("suspicious_substitution_burden", primary_analysis.get("contamination_burden", 0.0)) or 0.0)
    retry_substitution = float(retry_analysis.get("suspicious_substitution_burden", retry_analysis.get("contamination_burden", 0.0)) or 0.0)
    primary_visible_ratio = float(primary_analysis.get("visible_malayalam_char_ratio", 0.0) or 0.0)
    retry_visible_ratio = float(retry_analysis.get("visible_malayalam_char_ratio", 0.0) or 0.0)

    if primary_fidelity_failed and not retry_fidelity_failed:
        return True
    if retry_fidelity_failed and not primary_fidelity_failed:
        return False
    if primary_state == "catastrophic_wrong_script_failure" and retry_state != "catastrophic_wrong_script_failure":
        return True
    if retry_state == "catastrophic_wrong_script_failure" and primary_state != "catastrophic_wrong_script_failure":
        return False
    if retry_display <= 0 and retry_visible <= 1 and retry_fidelity_failed:
        return False
    if retry_display > primary_display and retry_visible >= primary_visible and retry_wrong_script <= primary_wrong_script + 0.01:
        return True
    if retry_visible >= primary_visible + 4 and retry_wrong_script <= primary_wrong_script and retry_substitution <= primary_substitution + 0.02:
        return True
    if retry_visible_ratio >= primary_visible_ratio + 0.12 and retry_display >= primary_display and retry_wrong_script <= primary_wrong_script:
        return True
    if retry_wrong_script <= primary_wrong_script - 0.12 and retry_display >= primary_display and retry_substitution <= primary_substitution + 0.02:
        return True
    if retry_substitution <= primary_substitution - 0.18 and retry_display >= primary_display and retry_visible >= primary_visible:
        return True
    if retry_analysis.get("quality_class") == "clearly_good" and primary_analysis.get("quality_class") != "clearly_good":
        return True
    trust_delta = float(retry_analysis.get("lexical_trust", 0.0) or 0.0) - float(primary_analysis.get("lexical_trust", 0.0) or 0.0)
    readability_delta = float(retry_analysis.get("readability", 0.0) or 0.0) - float(primary_analysis.get("readability", 0.0) or 0.0)
    wrong_script_delta = float(primary_analysis.get("wrong_script_burden", 0.0) or 0.0) - float(retry_analysis.get("wrong_script_burden", 0.0) or 0.0)
    contamination_delta = float(primary_analysis.get("contamination_burden", 0.0) or 0.0) - float(retry_analysis.get("contamination_burden", 0.0) or 0.0)
    visible_word_delta = int(retry_analysis.get("trusted_visible_word_count", 0) or 0) - int(primary_analysis.get("trusted_visible_word_count", 0) or 0)
    display_delta = int(retry_analysis.get("trusted_display_unit_count", 0) or 0) - int(primary_analysis.get("trusted_display_unit_count", 0) or 0)
    return (
        trust_delta >= 0.06
        or readability_delta >= 0.06
        or wrong_script_delta >= 0.08
        or contamination_delta >= 0.12
        or visible_word_delta >= 4
        or display_delta >= 1
    )


def _apply_bounded_malayalam_faithfulness_recovery(payload: Dict) -> Tuple[Dict, Dict[str, object]]:
    from . import utils as videos_utils

    analysis = _analyze_malayalam_asr_payload(payload)
    decision = {
        "attempted": False,
        "applied": False,
        "reason": "",
        "blocked_reason": "",
    }
    if analysis.get("quality_class") == "clearly_good":
        decision["blocked_reason"] = "primary_result_already_faithful"
        return payload, decision
    if bool(analysis.get("source_language_fidelity_failed", False)) and not bool(analysis.get("full_clip_fidelity_retry_candidate", False)):
        decision["blocked_reason"] = "fidelity_failed_without_recoverable_signal"
        return payload, decision
    if not (
        bool(analysis.get("confusion_candidate", False))
        or float(analysis.get("suspicious_substitution_burden", 0.0) or 0.0) >= 0.24
        or float(analysis.get("wrong_script_burden", 0.0) or 0.0) >= 0.18
    ):
        decision["blocked_reason"] = "no_recoverable_faithfulness_signal"
        return payload, decision

    decision["attempted"] = True
    repaired = videos_utils.repair_malayalam_degraded_transcript(
        str(payload.get("text") or ""),
        list(payload.get("segments") or []),
    )
    candidate_payload = dict(payload)
    candidate_payload["text"] = str(repaired.get("text") or payload.get("text") or "")
    candidate_payload["segments"] = list(repaired.get("segments") or payload.get("segments") or [])
    candidate_payload["metadata"] = {
        **(payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}),
        "malayalam_constrained_correction": repaired.get("metadata", {}) if isinstance(repaired.get("metadata"), dict) else {},
    }
    candidate_analysis = _analyze_malayalam_asr_payload(candidate_payload)
    if (
        bool(candidate_analysis.get("source_language_fidelity_failed", False))
        and not bool(analysis.get("source_language_fidelity_failed", False))
    ):
        decision["blocked_reason"] = "recovery_regressed_fidelity"
        return payload, decision
    if float(candidate_analysis.get("contamination_burden", 0.0) or 0.0) > float(analysis.get("contamination_burden", 0.0) or 0.0) + 0.08:
        decision["blocked_reason"] = "recovery_increased_english_substitution"
        return payload, decision
    if (
        int(candidate_analysis.get("trusted_visible_word_count", 0) or 0) <= int(analysis.get("trusted_visible_word_count", 0) or 0)
        and int(candidate_analysis.get("trusted_display_unit_count", 0) or 0) <= int(analysis.get("trusted_display_unit_count", 0) or 0)
        and float(candidate_analysis.get("wrong_script_burden", 0.0) or 0.0) >= float(analysis.get("wrong_script_burden", 0.0) or 0.0)
        and float(candidate_analysis.get("suspicious_substitution_burden", candidate_analysis.get("contamination_burden", 0.0)) or 0.0)
        >= float(analysis.get("suspicious_substitution_burden", analysis.get("contamination_burden", 0.0)) or 0.0)
    ):
        decision["blocked_reason"] = "recovery_did_not_improve_faithfulness"
        return payload, decision
    if _is_malayalam_second_pass_better(analysis, candidate_analysis):
        decision["applied"] = True
        decision["reason"] = "bounded_malayalam_faithfulness_recovery_improved"
        return candidate_payload, decision
    decision["blocked_reason"] = "recovery_not_materially_better"
    return payload, decision


def _apply_bounded_malayalam_post_asr_correction(payload: Dict) -> Tuple[Dict, Dict[str, object]]:
    return _apply_bounded_malayalam_faithfulness_recovery(payload)


def build_malayalam_specialist_candidate(
    *,
    audio_path: str,
    source_type: str,
    current_payload: Dict,
    route_decision: Dict[str, object],
) -> Tuple[Optional[Dict], Dict[str, object]]:
    analysis = _analyze_malayalam_asr_payload(current_payload)
    decision = {
        "attempted": False,
        "applied": False,
        "reason": "",
        "blocked_reason": "",
        "backend": "",
        "candidate_source": "",
    }
    if analysis.get("quality_class") == "clearly_good":
        decision["blocked_reason"] = "already_faithful_malayalam"
        return None, decision
    if analysis.get("quality_class") == "clearly_hopeless":
        decision["blocked_reason"] = "hopeless_malayalam_no_specialist_signal"
        return None, decision
    if not bool(analysis.get("detected_language") == "ml" and float(analysis.get("detected_language_confidence", 0.0) or 0.0) >= 0.72):
        decision["blocked_reason"] = "not_strong_malayalam_detection"
        return None, decision
    if not bool(
        analysis.get("recoverable_malayalam_fidelity_gap", False)
        or analysis.get("confusion_candidate", False)
        or float(analysis.get("wrong_script_burden", 0.0) or 0.0) >= 0.18
        or float(analysis.get("suspicious_substitution_burden", 0.0) or 0.0) >= 0.20
    ):
        decision["blocked_reason"] = "no_specialist_recovery_signal"
        return None, decision

    decision["attempted"] = True
    decision["reason"] = "recoverable_malayalam_specialist_candidate"
    specialist_model = _malayalam_specialist_recovery_model()
    decision["backend"] = f"local_whisper:{specialist_model}"
    decision["candidate_source"] = "specialist_recovery_candidate"
    try:
        candidate_payload = _run_malayalam_local_model(audio_path, source_type, specialist_model)
    except Exception:
        decision["blocked_reason"] = "specialist_backend_failed"
        return None, decision

    metadata = candidate_payload.get("metadata") if isinstance(candidate_payload.get("metadata"), dict) else {}
    metadata["malayalam_specialist_backend"] = decision["backend"]
    metadata["malayalam_candidate_stage"] = "specialist_recovery_candidate"
    metadata["malayalam_final_candidate_source"] = "specialist_recovery_candidate"
    candidate_payload["metadata"] = metadata
    candidate_analysis = _analyze_malayalam_asr_payload(candidate_payload)
    if (
        int(candidate_analysis.get("trusted_visible_word_count", 0) or 0) <= int(analysis.get("trusted_visible_word_count", 0) or 0)
        and int(candidate_analysis.get("trusted_display_unit_count", 0) or 0) <= int(analysis.get("trusted_display_unit_count", 0) or 0)
        and float(candidate_analysis.get("visible_malayalam_char_ratio", 0.0) or 0.0) <= float(analysis.get("visible_malayalam_char_ratio", 0.0) or 0.0)
        and float(candidate_analysis.get("wrong_script_burden", 0.0) or 0.0) >= float(analysis.get("wrong_script_burden", 0.0) or 0.0)
        and float(candidate_analysis.get("suspicious_substitution_burden", candidate_analysis.get("contamination_burden", 0.0)) or 0.0)
        >= float(analysis.get("suspicious_substitution_burden", analysis.get("contamination_burden", 0.0)) or 0.0)
    ):
        decision["blocked_reason"] = "specialist_candidate_not_faithfully_better"
        return None, decision
    if _is_malayalam_second_pass_better(analysis, candidate_analysis):
        decision["applied"] = True
        decision["reason"] = "specialist_candidate_improved_faithfulness"
        return candidate_payload, decision
    decision["blocked_reason"] = "specialist_candidate_not_faithfully_better"
    return None, decision


def build_malayalam_linguistic_correction_candidate(
    *,
    current_payload: Dict,
) -> Tuple[Optional[Dict], Dict[str, object]]:
    from . import utils as videos_utils

    analysis = _analyze_malayalam_asr_payload(current_payload)
    decision = {
        "attempted": False,
        "applied": False,
        "reason": "",
        "blocked_reason": "",
        "backend": "deterministic_malayalam_repair",
        "candidate_source": "linguistic_correction_candidate",
    }
    if analysis.get("quality_class") == "clearly_good":
        decision["blocked_reason"] = "already_faithful_malayalam"
        return None, decision
    if analysis.get("quality_class") == "clearly_hopeless":
        decision["blocked_reason"] = "hopeless_malayalam_no_linguistic_signal"
        return None, decision
    if bool(analysis.get("source_language_fidelity_failed", False)) and not bool(analysis.get("recoverable_malayalam_fidelity_gap", False)):
        decision["blocked_reason"] = "fidelity_failed_without_recoverable_signal"
        return None, decision
    if int(analysis.get("word_count", 0) or 0) < 3:
        decision["blocked_reason"] = "insufficient_nonempty_signal"
        return None, decision
    if (
        int(analysis.get("trusted_visible_word_count", 0) or 0) >= 8
        and int(analysis.get("trusted_display_unit_count", 0) or 0) >= 1
        and float(analysis.get("visible_malayalam_char_ratio", 0.0) or 0.0) >= 0.34
    ):
        decision["blocked_reason"] = "substantial_trusted_malayalam_already_present"
        return None, decision
    if not bool(
        analysis.get("recoverable_malayalam_fidelity_gap", False)
        or analysis.get("confusion_candidate", False)
        or float(analysis.get("wrong_script_burden", 0.0) or 0.0) >= 0.18
        or float(analysis.get("suspicious_substitution_burden", 0.0) or 0.0) >= 0.18
    ):
        decision["blocked_reason"] = "no_linguistic_correction_signal"
        return None, decision

    decision["attempted"] = True
    decision["reason"] = "recoverable_malayalam_linguistic_correction_candidate"
    repaired = videos_utils.repair_malayalam_degraded_transcript(
        str(current_payload.get("text") or ""),
        list(current_payload.get("segments") or []),
    )
    candidate_text = str(repaired.get("text") or "")
    correction_eval = videos_utils.evaluate_malayalam_linguistic_correction(
        str(current_payload.get("text") or ""),
        candidate_text,
    )
    if not bool(correction_eval.get("allowed", False)):
        decision["blocked_reason"] = str(correction_eval.get("blocked_reason", "") or "linguistic_correction_rejected")
        return None, decision

    candidate_payload = dict(current_payload)
    candidate_payload["text"] = candidate_text or str(current_payload.get("text") or "")
    candidate_payload["segments"] = list(repaired.get("segments") or current_payload.get("segments") or [])
    candidate_meta = {
        **(current_payload.get("metadata") if isinstance(current_payload.get("metadata"), dict) else {}),
        "malayalam_linguistic_correction": repaired.get("metadata", {}) if isinstance(repaired.get("metadata"), dict) else {},
        "malayalam_linguistic_correction_backend": decision["backend"],
        "malayalam_candidate_stage": "linguistic_correction_candidate",
        "malayalam_final_candidate_source": "linguistic_correction_candidate",
    }
    candidate_payload["metadata"] = candidate_meta
    candidate_analysis = _analyze_malayalam_asr_payload(candidate_payload)
    if (
        int(candidate_analysis.get("trusted_visible_word_count", 0) or 0) <= int(analysis.get("trusted_visible_word_count", 0) or 0)
        and int(candidate_analysis.get("trusted_display_unit_count", 0) or 0) <= int(analysis.get("trusted_display_unit_count", 0) or 0)
        and float(candidate_analysis.get("visible_malayalam_char_ratio", 0.0) or 0.0) <= float(analysis.get("visible_malayalam_char_ratio", 0.0) or 0.0)
        and float(candidate_analysis.get("wrong_script_burden", 0.0) or 0.0) >= float(analysis.get("wrong_script_burden", 0.0) or 0.0)
        and float(candidate_analysis.get("suspicious_substitution_burden", candidate_analysis.get("contamination_burden", 0.0)) or 0.0)
        >= float(analysis.get("suspicious_substitution_burden", analysis.get("contamination_burden", 0.0)) or 0.0)
    ):
        decision["blocked_reason"] = "linguistic_correction_not_faithfully_better"
        return None, decision
    if _is_malayalam_second_pass_better(analysis, candidate_analysis):
        decision["applied"] = True
        decision["reason"] = "linguistic_correction_improved_faithfulness"
        return candidate_payload, decision
    decision["blocked_reason"] = "linguistic_correction_not_faithfully_better"
    return None, decision


def _attach_malayalam_strategy_metadata(
    payload: Dict,
    *,
    route_decision: Dict[str, object],
    preprocess_meta: Dict[str, object],
    primary_model_used: str,
    fallback_model_used: str = "",
    retry_model_used: str = "",
) -> Dict:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    metadata["malayalam_asr_strategy"] = str(route_decision.get("malayalam_asr_strategy", "") or "")
    metadata["asr_strategy_reason"] = str(route_decision.get("asr_strategy_reason", "") or "")
    metadata["primary_model_used"] = primary_model_used
    metadata["fallback_model_used"] = fallback_model_used
    metadata["retry_model_used"] = retry_model_used
    metadata["audio_preprocess_applied"] = bool(preprocess_meta.get("audio_preprocess_applied", False))
    metadata["preprocess_steps"] = list(preprocess_meta.get("preprocess_steps", []) or [])
    metadata["chunk_policy_used"] = str(preprocess_meta.get("chunk_policy_used", "") or "")
    payload["metadata"] = metadata
    return payload


def _transcribe_video_router_single(
    audio_path: str,
    source_type: str = "generic",
    requested_language: Optional[str] = None,
    already_preprocessed: bool = False,
    routing_duration_seconds: Optional[float] = None,
) -> Dict:
    """
    Route ASR requests based on language:
    - en => local whisper
    - non-en => Deepgram
    """
    req_lang = normalize_language_code(requested_language, default="auto", allow_auto=True)
    preprocessed_path = None
    duration = 0.0
    start = time.time()
    preprocess_meta: Dict[str, object] = {}

    try:
        preprocessed_path = _preprocess_audio_for_asr(
            audio_path,
            language_hint="" if req_lang == "auto" else req_lang,
            already_preprocessed=already_preprocessed,
        )
        duration = _audio_duration_seconds(preprocessed_path)
        route_duration = max(float(routing_duration_seconds or 0.0), float(duration or 0.0))
        file_size_bytes = _audio_file_size_bytes(preprocessed_path)
        dg_supported = _deepgram_supported_languages()
        preprocess_meta = _describe_audio_preprocess("" if req_lang == "auto" else req_lang, route_duration)

        chosen_lang = req_lang
        detection_confidence = 0.0
        payload: Optional[Dict] = None
        route_reason = ""
        fallback_triggered = False
        fallback_reason = ""
        quality_gate_passed = True
        route_decision: Dict[str, object] = {}
        terminal_malayalam_failure_reason = ""

        # Resolve language first for routing.
        if _render_demo_safe_asr_mode():
            if req_lang != "auto":
                chosen_lang = normalize_language_code(req_lang, default="en", allow_auto=False)
                detection_confidence = 1.0
            else:
                chosen_lang = "auto"
                detection_confidence = 0.0
            route_decision = _choose_demo_safe_engine(
                requested_lang=req_lang,
                chosen_lang=chosen_lang,
                duration_seconds=route_duration,
                file_size_bytes=file_size_bytes,
                detection_confidence=detection_confidence,
            )
        else:
            if req_lang == "auto":
                detected, detection_confidence = _detect_lang_for_router_with_confidence(preprocessed_path)
                chosen_lang = normalize_language_code(detected, default="auto", allow_auto=True)
            else:
                chosen_lang = normalize_language_code(req_lang, default="en", allow_auto=False)
                detection_confidence = 1.0

            route_decision = _choose_primary_engine(
                requested_lang=req_lang,
                chosen_lang=chosen_lang,
                duration_seconds=route_duration,
                deepgram_supported=dg_supported,
                file_size_bytes=file_size_bytes,
                detection_confidence=detection_confidence,
            )
        primary_engine = str(route_decision.get("engine", "whisper_local"))
        route_reason = str(route_decision.get("reason", ""))
        deepgram_supported_lang = chosen_lang in dg_supported
        if chosen_lang == "ml":
            logger.info(
                "[ML_ROUTE] primary_engine=%s model=%s fallback=%s",
                primary_engine,
                route_decision.get("model"),
                ",".join(route_decision.get("fallback_chain", []) or []),
            )
        logger.info(
            "[ASR_ROUTE] selected_asr_engine=%s selected_model=%s reason=%s requested=%s chosen=%s duration=%.1fs file_size_bytes=%s detection_confidence=%.2f provider_name=%s deepgram_supported=%s deepgram_available=%s fallback_chain=%s route_score=%.3f",
            primary_engine,
            route_decision.get("model"),
            route_reason,
            req_lang,
            chosen_lang,
            duration,
            file_size_bytes,
            detection_confidence,
            _provider_name_for_engine(primary_engine),
            deepgram_supported_lang,
            _deepgram_available(),
            route_decision.get("fallback_chain", []),
            float(route_decision.get("score", 0.0) or 0.0),
        )
        speed_optimized_for_longform = bool(route_decision.get("speed_optimized_for_longform", False))

        # Run primary engine.
        engine = primary_engine
        try:
            if primary_engine == "groq_whisper":
                lang = "en" if chosen_lang == "auto" else chosen_lang
                payload = _transcribe_with_groq_whisper(preprocessed_path, source_type, lang)
                logger.info("[ML_ASR_PRIMARY_RESULT] engine=%s model=%s", primary_engine, route_decision.get("model", ""))
            elif primary_engine == "deepgram":
                if req_lang != "auto" and req_lang in dg_supported:
                    payload = _call_deepgram_with_retries(
                        preprocessed_path,
                        language=req_lang,
                        detect_language=False,
                    )
                else:
                    payload = _call_deepgram_with_retries(
                        preprocessed_path,
                        language=None,
                        detect_language=True,
                    )
            else:
                lang = "auto" if chosen_lang == "auto" else chosen_lang
                if chosen_lang == "ml" and str(route_decision.get("malayalam_primary_model_override", "") or ""):
                    payload = _run_malayalam_local_model(
                        preprocessed_path,
                        source_type,
                        str(route_decision.get("malayalam_primary_model_override", "") or ""),
                    )
                else:
                    payload = _transcribe_with_local_whisper(preprocessed_path, source_type, lang)
                logger.info("[ML_ASR_PRIMARY_RESULT] engine=%s model=%s", primary_engine, route_decision.get("model", ""))
        except Exception as primary_err:
            terminal_malayalam_failure_reason = str(primary_err)
            logger.warning(
                "Primary ASR engine failed (engine=%s reason=%s): %s",
                primary_engine,
                route_reason,
                primary_err,
            )
            if _render_demo_safe_asr_mode():
                fallback_lang = "en" if chosen_lang == "auto" else chosen_lang
                if primary_engine == "deepgram" and bool(getattr(settings, "USE_GROQ_WHISPER", True)) and bool(getattr(settings, "GROQ_API_KEY", "")):
                    try:
                        payload = _transcribe_with_groq_whisper(preprocessed_path, source_type, fallback_lang)
                        engine = "groq_whisper"
                        fallback_triggered = True
                        fallback_reason = str(primary_err)
                        route_reason = f"{route_reason}|fallback_groq"
                    except Exception as groq_err:
                        if _is_rate_limit_error(groq_err):
                            _block_groq_fallback_due_to_rate_limit()
                        logger.warning("Render demo-safe Groq fallback failed: %s", groq_err)
                        raise _render_demo_safe_transcription_error(
                            fallback_lang,
                            reason=f"primary={primary_engine};fallback=groq_whisper",
                        ) from groq_err
                elif primary_engine == "groq_whisper" and _deepgram_available():
                    try:
                        payload = _call_deepgram_with_retries(
                            preprocessed_path,
                            language=(None if req_lang == "auto" else fallback_lang),
                            detect_language=(req_lang == "auto"),
                        )
                        engine = "deepgram"
                        fallback_triggered = True
                        fallback_reason = str(primary_err)
                        route_reason = f"{route_reason}|fallback_deepgram"
                    except Exception as dg_err:
                        logger.warning("Render demo-safe Deepgram fallback failed: %s", dg_err)
                        raise _render_demo_safe_transcription_error(
                            fallback_lang,
                            reason=f"primary={primary_engine};fallback=deepgram",
                        ) from dg_err
                else:
                    raise _render_demo_safe_transcription_error(
                        fallback_lang,
                        reason=f"primary={primary_engine}",
                    ) from primary_err
                if payload is None:
                    raise _render_demo_safe_transcription_error(
                        fallback_lang,
                        reason=f"primary={primary_engine};no_remote_fallback",
                    ) from primary_err
            else:
                if _is_malayalam(chosen_lang) and _is_cuda_oom_error(primary_err):
                    logger.warning(
                        "[ML_FAST_FAIL] skipping_groq_fallback reason=malayalam_local_only_after_oom"
                    )
                    raise
                # Local fallback first.
                fallback_lang = "auto" if chosen_lang == "auto" else chosen_lang
                if primary_engine != "whisper_local" or (
                    primary_engine == "whisper_local"
                    and fallback_lang == "ml"
                    and str(route_decision.get("malayalam_primary_model_override", "") or "")
                    and not speed_optimized_for_longform
                ):
                    try:
                        if fallback_lang == "ml":
                            payload = _transcribe_with_malayalam_local(preprocessed_path, source_type)
                        else:
                            payload = _transcribe_with_local_whisper(preprocessed_path, source_type, fallback_lang)
                        engine = "whisper_local"
                        fallback_triggered = True
                        fallback_reason = str(primary_err)
                        route_reason = "deepgram_fallback_to_local" if primary_engine == "deepgram" else f"{route_reason}|fallback_local"
                        if fallback_lang == "ml":
                            logger.info("[ML_ASR_FALLBACK_RESULT] engine=%s model=%s reason=%s", engine, "large-v3", fallback_reason or "")
                    except Exception as local_err:
                        if fallback_lang == "ml":
                            terminal_malayalam_failure_reason = str(local_err)
                        logger.warning("Local Whisper fallback failed: %s", local_err)
                elif payload is None and fallback_lang == "ml" and speed_optimized_for_longform:
                    logger.info(
                        "[ML_ROUTE] local_fallback_skipped=true reason=longform_fast_path_disallows_heavy_local_fallback"
                    )
                # Deepgram fallback (non-English only, if enabled/supported).
                if (
                    payload is None
                    and fallback_lang != "ml"
                    and _deepgram_available()
                    and fallback_lang != "en"
                    and fallback_lang in dg_supported
                ):
                    try:
                        if fallback_lang == "ml":
                            logger.info(
                                "[ML_ROUTE] experimental_deepgram_enabled=%s selected=%s skip_reason=%s",
                                bool(getattr(settings, "ASR_ENABLE_DEEPGRAM_MALAYALAM_EXPERIMENT", False)),
                                True,
                                "fallback_to_deepgram",
                            )
                        payload = _call_deepgram_with_retries(
                            preprocessed_path,
                            language=fallback_lang,
                            detect_language=False,
                        )
                        engine = "deepgram"
                        fallback_triggered = True
                        fallback_reason = str(primary_err)
                        route_reason = f"{route_reason}|fallback_deepgram"
                    except Exception as deepgram_err:
                        if fallback_lang == "ml":
                            logger.info(
                                "[ML_ROUTE] experimental_deepgram_enabled=%s selected=%s skip_reason=%s",
                                bool(getattr(settings, "ASR_ENABLE_DEEPGRAM_MALAYALAM_EXPERIMENT", False)),
                                False,
                                f"deepgram_error:{type(deepgram_err).__name__}",
                            )
                        logger.warning("Deepgram fallback failed: %s", deepgram_err)
                # Groq fallback last for non-Malayalam only. Malayalam must stay local-first;
                # Groq is not allowed to become the persisted final transcript path.
                if payload is None and fallback_lang == "ml":
                    logger.info(
                        "[ML_ROUTE] groq_fallback_skipped=true reason=malayalam_local_only_final_path"
                    )
                if payload is None and fallback_lang != "ml" and primary_engine != "groq_whisper" and _asr_use_groq_fallback():
                    try:
                        payload = _transcribe_with_groq_whisper(preprocessed_path, source_type, fallback_lang)
                        engine = "groq_whisper"
                        fallback_triggered = True
                        fallback_reason = str(primary_err)
                        route_reason = f"{route_reason}|fallback_groq"
                        if fallback_lang == "ml":
                            logger.info("[ML_ASR_FALLBACK_RESULT] engine=%s model=%s reason=%s", engine, "whisper-large-v3", fallback_reason or "")
                    except Exception as groq_err:
                        if _is_rate_limit_error(groq_err):
                            _block_groq_fallback_due_to_rate_limit()
                        logger.warning("Groq fallback failed: %s", groq_err)

        if isinstance(payload, dict):
            chosen_lang = normalize_language_code(
                payload.get("language"),
                default=(chosen_lang if chosen_lang != "auto" else "en"),
                allow_auto=False,
            )
            payload_meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            fallback_triggered = fallback_triggered or bool(payload_meta.get("fallback_triggered"))
            if payload_meta.get("fallback_reason"):
                fallback_reason = payload_meta.get("fallback_reason")
            if "transcript_quality_gate_passed" in payload_meta:
                quality_gate_passed = bool(payload_meta.get("transcript_quality_gate_passed"))

        if payload is None and chosen_lang == "ml":
            payload = _build_terminal_malayalam_failure_payload(
                route_decision=route_decision,
                route_reason=route_reason,
                failure_reason=terminal_malayalam_failure_reason or "malayalam_local_only_terminal_failure",
                detection_confidence=detection_confidence,
                engine="whisper_local",
            )
            logger.warning(
                "[ML_TERMINAL_FAILURE_PAYLOAD] reason=%s route_reason=%s model=%s",
                terminal_malayalam_failure_reason or "malayalam_local_only_terminal_failure",
                route_reason or "none",
                str(route_decision.get("model", "") or _get_malayalam_model()),
            )

        if payload is None:
            raise RuntimeError("ASR router failed to produce transcript payload")

        if chosen_lang == "ml":
            payload_metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            payload_metadata["malayalam_draft_candidate"] = True
            payload_metadata["malayalam_candidate_stage"] = "fast_draft_candidate"
            payload_metadata["malayalam_draft_engine"] = engine
            payload_metadata["malayalam_draft_model"] = str(route_decision.get("primary_model", "") or "")
            payload_metadata["malayalam_final_candidate_source"] = "fast_draft_candidate"
            payload["metadata"] = payload_metadata
            second_pass = _should_attempt_malayalam_second_pass(payload, route_decision)
            logger.info(
                "[ML_ASR_SECOND_PASS_SELECT] attempt=%s reason=%s blocked_reason=%s lexical_trust=%.3f readability=%.3f wrong_script_burden=%.3f contamination_burden=%.3f trusted_visible_words=%s trusted_display_units=%s",
                bool(second_pass.get("attempt_second_pass", False)),
                second_pass.get("reason", "") or "none",
                second_pass.get("blocked_reason", "") or "none",
                float(second_pass.get("analysis", {}).get("lexical_trust", 0.0) or 0.0),
                float(second_pass.get("analysis", {}).get("readability", 0.0) or 0.0),
                float(second_pass.get("analysis", {}).get("wrong_script_burden", 0.0) or 0.0),
                float(second_pass.get("analysis", {}).get("contamination_burden", 0.0) or 0.0),
                int(second_pass.get("analysis", {}).get("trusted_visible_word_count", 0) or 0),
                int(second_pass.get("analysis", {}).get("trusted_display_unit_count", 0) or 0),
            )
            primary_analysis = dict(second_pass.get("analysis", {}) or {})
            if second_pass.get("blocked_reason") == "zero_malayalam_evidence_skips_second_pass":
                logger.info(
                    "[ML_FAST_FAIL] skipping_second_pass reason=zero_malayalam_evidence malayalam_ratio=%.3f dominant_script=%s trusted_words=%s",
                    float(primary_analysis.get("visible_malayalam_char_ratio", 0.0) or 0.0),
                    primary_analysis.get("dominant_script_final", "") or "unknown",
                    int(primary_analysis.get("trusted_visible_word_count", 0) or 0),
                )
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            confusion_retry_candidate = bool(primary_analysis.get("confusion_candidate", False))
            metadata["confusion_retry_candidate"] = confusion_retry_candidate
            metadata["confusion_retry_executed"] = False
            metadata["confusion_retry_model"] = ""
            metadata["confusion_retry_improved"] = False
            metadata["confusion_retry_improvement_reason"] = ""
            metadata["confusion_retry_before_after"] = {
                "before": {
                    "lexical_trust": float(primary_analysis.get("lexical_trust", 0.0) or 0.0),
                    "readability": float(primary_analysis.get("readability", 0.0) or 0.0),
                    "trusted_visible_word_count": int(primary_analysis.get("trusted_visible_word_count", 0) or 0),
                    "trusted_display_unit_count": int(primary_analysis.get("trusted_display_unit_count", 0) or 0),
                    "dominant_script_final": str(primary_analysis.get("dominant_script_final", "") or ""),
                }
            }
            metadata["second_pass_asr_attempted"] = False
            metadata["second_pass_asr_reason"] = str(second_pass.get("reason", "") or "")
            metadata["second_pass_asr_blocked_reason"] = str(second_pass.get("blocked_reason", "") or "")
            metadata["second_pass_asr_model"] = str(route_decision.get("retry_model", "") or "")
            metadata["second_pass_asr_improved"] = False
            payload["metadata"] = metadata
            if confusion_retry_candidate:
                logger.info(
                    "[ML_ASR_CONFUSION_RETRY] candidate=%s executed=%s reason=%s blocked_reason=%s dominant_script=%s detected_language=%s conf=%.3f",
                    True,
                    bool(second_pass.get("attempt_second_pass", False)),
                    second_pass.get("reason", "") or "none",
                    second_pass.get("blocked_reason", "") or "none",
                    primary_analysis.get("dominant_script_final", "") or "unknown",
                    primary_analysis.get("detected_language", "") or "unknown",
                    float(primary_analysis.get("detected_language_confidence", 0.0) or 0.0),
                )
            if bool(primary_analysis.get("full_clip_fidelity_retry_candidate", False)):
                logger.info(
                    "[ML_FULL_CLIP_RETRY_SELECT] candidate=%s executed=%s reason=%s fidelity_failed=%s dominant_script=%s suspicious_substitution_burden=%.3f wrong_script_burden=%.3f",
                    True,
                    bool(second_pass.get("attempt_second_pass", False)),
                    second_pass.get("reason", "") or "none",
                    bool(primary_analysis.get("source_language_fidelity_failed", False)),
                    primary_analysis.get("dominant_script_final", "") or "unknown",
                    float(primary_analysis.get("suspicious_substitution_burden", 0.0) or 0.0),
                    float(primary_analysis.get("wrong_script_burden", 0.0) or 0.0),
                )
            if bool(second_pass.get("attempt_second_pass", False)):
                retry_model = str(route_decision.get("retry_model", "") or _malayalam_second_pass_model())
                current_model = str(
                    metadata.get("actual_local_model_name")
                    or metadata.get("resolved_local_model_name")
                    or route_decision.get("model", "")
                    or ""
                ).strip()
                retry_payload = None
                retry_error = ""
                metadata["confusion_retry_executed"] = confusion_retry_candidate
                metadata["confusion_retry_model"] = retry_model if confusion_retry_candidate else ""
                # Allow retry even with same model if primary quality was degraded
                primary_quality_degraded = (
                    float(primary_analysis.get("lexical_trust", 0.0) or 0.0) < 0.15 or
                    float(primary_analysis.get("readability", 0.0) or 0.0) < 0.15 or
                    int(primary_analysis.get("trusted_visible_word_count", 0) or 0) < 5 or
                    int(primary_analysis.get("trusted_display_unit_count", 0) or 0) < 1
                )
                models_same = bool(retry_model and current_model and retry_model == current_model)
                if models_same and not primary_quality_degraded:
                    # Skip only when models same AND primary quality is acceptable
                    metadata["second_pass_asr_blocked_reason"] = "retry_model_same_as_primary"
                    logger.info(
                        "[ML_ASR_SECOND_PASS_SKIP] reason=retry_model_same_as_primary model=%s",
                        retry_model,
                    )
                else:
                    metadata["second_pass_asr_attempted"] = True
                    try:
                        retry_payload = _run_malayalam_local_model(preprocessed_path, source_type, retry_model)
                    except Exception as retry_exc:
                        retry_error = f"{type(retry_exc).__name__}:{retry_exc}"
                        logger.warning(
                            "Malayalam second-pass model failed (model=%s). Falling back safely to current payload.",
                            retry_model,
                        )
                        metadata["second_pass_asr_blocked_reason"] = "second_pass_model_failed"
                if retry_payload is not None:
                    retry_analysis = _analyze_malayalam_asr_payload(retry_payload)
                    improved = _is_malayalam_second_pass_better(primary_analysis, retry_analysis)
                    metadata["second_pass_asr_improved"] = bool(improved)
                    metadata["second_pass_asr_model"] = retry_model
                    metadata["confusion_retry_improved"] = bool(improved) if confusion_retry_candidate else False
                    metadata["confusion_retry_before_after"] = {
                        "before": dict(metadata.get("confusion_retry_before_after", {}).get("before", {}) or {}),
                        "after": {
                            "lexical_trust": float(retry_analysis.get("lexical_trust", 0.0) or 0.0),
                            "readability": float(retry_analysis.get("readability", 0.0) or 0.0),
                            "trusted_visible_word_count": int(retry_analysis.get("trusted_visible_word_count", 0) or 0),
                            "trusted_display_unit_count": int(retry_analysis.get("trusted_display_unit_count", 0) or 0),
                            "dominant_script_final": str(retry_analysis.get("dominant_script_final", "") or ""),
                        },
                    }
                    if confusion_retry_candidate:
                        improvement_reasons = []
                        if float(retry_analysis.get("lexical_trust", 0.0) or 0.0) > float(primary_analysis.get("lexical_trust", 0.0) or 0.0):
                            improvement_reasons.append("lexical_trust")
                        if float(retry_analysis.get("readability", 0.0) or 0.0) > float(primary_analysis.get("readability", 0.0) or 0.0):
                            improvement_reasons.append("readability")
                        if int(retry_analysis.get("trusted_visible_word_count", 0) or 0) > int(primary_analysis.get("trusted_visible_word_count", 0) or 0):
                            improvement_reasons.append("trusted_visible_words")
                        if int(retry_analysis.get("trusted_display_unit_count", 0) or 0) > int(primary_analysis.get("trusted_display_unit_count", 0) or 0):
                            improvement_reasons.append("trusted_display_units")
                        if str(retry_analysis.get("dominant_script_final", "") or "") != str(primary_analysis.get("dominant_script_final", "") or ""):
                            improvement_reasons.append("dominant_script")
                        metadata["confusion_retry_improvement_reason"] = ",".join(improvement_reasons) or (
                            "materially_better" if improved else "not_materially_better"
                        )
                    logger.info(
                        "[ML_ASR_SECOND_PASS_RESULT] improved=%s retry_model=%s retry_quality_class=%s lexical_trust=%.3f readability=%.3f wrong_script_burden=%.3f contamination_burden=%.3f trusted_visible_words=%s trusted_display_units=%s blocked_reason=%s",
                        bool(improved),
                        retry_model,
                        retry_analysis.get("quality_class", ""),
                        float(retry_analysis.get("lexical_trust", 0.0) or 0.0),
                        float(retry_analysis.get("readability", 0.0) or 0.0),
                        float(retry_analysis.get("wrong_script_burden", 0.0) or 0.0),
                        float(retry_analysis.get("contamination_burden", 0.0) or 0.0),
                        int(retry_analysis.get("trusted_visible_word_count", 0) or 0),
                        int(retry_analysis.get("trusted_display_unit_count", 0) or 0),
                        metadata.get("second_pass_asr_blocked_reason", "") or "none",
                    )
                    if confusion_retry_candidate:
                        logger.info(
                            "[ML_ASR_CONFUSION_RETRY_RESULT] improved=%s model=%s improvement_reason=%s before_script=%s after_script=%s before_visible=%s after_visible=%s before_display=%s after_display=%s",
                            bool(improved),
                            retry_model,
                            metadata.get("confusion_retry_improvement_reason", "") or "none",
                            primary_analysis.get("dominant_script_final", "") or "unknown",
                            retry_analysis.get("dominant_script_final", "") or "unknown",
                            int(primary_analysis.get("trusted_visible_word_count", 0) or 0),
                            int(retry_analysis.get("trusted_visible_word_count", 0) or 0),
                            int(primary_analysis.get("trusted_display_unit_count", 0) or 0),
                            int(retry_analysis.get("trusted_display_unit_count", 0) or 0),
                        )
                    if improved:
                        payload = retry_payload
                        payload_meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
                        payload_meta["malayalam_draft_candidate"] = True
                        payload_meta["malayalam_candidate_stage"] = "recovery_candidate"
                        payload_meta["malayalam_final_candidate_source"] = "bounded_second_pass_recovery"
                        payload_meta["second_pass_asr_attempted"] = True
                        payload_meta["second_pass_asr_reason"] = str(second_pass.get("reason", "") or "")
                        payload_meta["second_pass_asr_model"] = retry_model
                        payload_meta["second_pass_asr_improved"] = True
                        payload_meta["confusion_retry_candidate"] = confusion_retry_candidate
                        payload_meta["confusion_retry_executed"] = confusion_retry_candidate
                        payload_meta["confusion_retry_model"] = retry_model if confusion_retry_candidate else ""
                        payload_meta["confusion_retry_improved"] = bool(improved) if confusion_retry_candidate else False
                        payload_meta["confusion_retry_improvement_reason"] = metadata.get("confusion_retry_improvement_reason", "") or ""
                        payload_meta["confusion_retry_before_after"] = metadata.get("confusion_retry_before_after", {})
                        payload["metadata"] = payload_meta
                        engine = "whisper_local"
                        fallback_triggered = True
                        fallback_reason = "malayalam_bounded_second_pass_improved"
                    elif not retry_error:
                        metadata["second_pass_asr_blocked_reason"] = "second_pass_not_materially_better"
                        if confusion_retry_candidate and not metadata.get("confusion_retry_improvement_reason"):
                            metadata["confusion_retry_improvement_reason"] = "not_materially_better"
                elif retry_error:
                    logger.info(
                        "[ML_ASR_SECOND_PASS_RESULT] improved=%s retry_model=%s retry_quality_class=%s lexical_trust=%.3f readability=%.3f wrong_script_burden=%.3f contamination_burden=%.3f trusted_visible_words=%s trusted_display_units=%s blocked_reason=%s",
                        False,
                        retry_model,
                        "failed",
                        float(primary_analysis.get("lexical_trust", 0.0) or 0.0),
                        float(primary_analysis.get("readability", 0.0) or 0.0),
                        float(primary_analysis.get("wrong_script_burden", 0.0) or 0.0),
                        float(primary_analysis.get("contamination_burden", 0.0) or 0.0),
                        int(primary_analysis.get("trusted_visible_word_count", 0) or 0),
                        int(primary_analysis.get("trusted_display_unit_count", 0) or 0),
                        metadata.get("second_pass_asr_blocked_reason", "") or "second_pass_model_failed",
                    )
                    if confusion_retry_candidate:
                        metadata["confusion_retry_improvement_reason"] = "retry_model_failed"

            recovery_metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            corrected_payload, correction_decision = _apply_bounded_malayalam_faithfulness_recovery(payload)
            recovery_metadata["malayalam_faithfulness_recovery_attempted"] = bool(correction_decision.get("attempted", False))
            recovery_metadata["malayalam_faithfulness_recovery_applied"] = bool(correction_decision.get("applied", False))
            recovery_metadata["malayalam_faithfulness_recovery_reason"] = str(correction_decision.get("reason", "") or "")
            recovery_metadata["malayalam_faithfulness_recovery_blocked_reason"] = str(correction_decision.get("blocked_reason", "") or "")
            recovery_metadata["malayalam_constrained_correction_attempted"] = bool(correction_decision.get("attempted", False))
            recovery_metadata["malayalam_constrained_correction_applied"] = bool(correction_decision.get("applied", False))
            recovery_metadata["malayalam_constrained_correction_reason"] = str(correction_decision.get("reason", "") or "")
            recovery_metadata["malayalam_constrained_correction_blocked_reason"] = str(correction_decision.get("blocked_reason", "") or "")
            payload["metadata"] = recovery_metadata
            logger.info(
                "[ML_FULL_CLIP_RECOVERY] attempted=%s applied=%s reason=%s blocked_reason=%s",
                bool(correction_decision.get("attempted", False)),
                bool(correction_decision.get("applied", False)),
                correction_decision.get("reason", "") or "none",
                correction_decision.get("blocked_reason", "") or "none",
            )
            if bool(correction_decision.get("applied", False)):
                payload = corrected_payload
                payload_meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
                payload_meta["malayalam_draft_candidate"] = True
                payload_meta["malayalam_candidate_stage"] = "final_faithful_candidate"
                payload_meta["malayalam_final_candidate_source"] = "faithfulness_recovery"
                payload["metadata"] = payload_meta

            specialist_metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            specialist_payload, specialist_decision = build_malayalam_specialist_candidate(
                audio_path=preprocessed_path,
                source_type=source_type,
                current_payload=payload,
                route_decision=route_decision,
            )
            specialist_metadata["malayalam_specialist_recovery_attempted"] = bool(specialist_decision.get("attempted", False))
            specialist_metadata["malayalam_specialist_recovery_applied"] = bool(specialist_decision.get("applied", False))
            specialist_metadata["malayalam_specialist_recovery_reason"] = str(specialist_decision.get("reason", "") or "")
            specialist_metadata["malayalam_specialist_recovery_blocked_reason"] = str(specialist_decision.get("blocked_reason", "") or "")
            specialist_metadata["malayalam_specialist_backend"] = str(specialist_decision.get("backend", "") or "")
            payload["metadata"] = specialist_metadata
            logger.info(
                "[ML_SPECIALIST_RECOVERY] attempted=%s applied=%s reason=%s blocked_reason=%s backend=%s",
                bool(specialist_decision.get("attempted", False)),
                bool(specialist_decision.get("applied", False)),
                specialist_decision.get("reason", "") or "none",
                specialist_decision.get("blocked_reason", "") or "none",
                specialist_decision.get("backend", "") or "none",
            )
            if specialist_payload is not None and bool(specialist_decision.get("applied", False)):
                payload = specialist_payload
                payload_meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
                payload_meta["malayalam_draft_candidate"] = True
                payload_meta["malayalam_specialist_recovery_attempted"] = True
                payload_meta["malayalam_specialist_recovery_applied"] = True
                payload_meta["malayalam_specialist_recovery_reason"] = str(specialist_decision.get("reason", "") or "")
                payload_meta["malayalam_specialist_recovery_blocked_reason"] = ""
                payload_meta["malayalam_specialist_backend"] = str(specialist_decision.get("backend", "") or "")
                payload_meta["malayalam_candidate_stage"] = "specialist_recovery_candidate"
                payload_meta["malayalam_final_candidate_source"] = "specialist_recovery_candidate"
                payload["metadata"] = payload_meta

            linguistic_metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            linguistic_payload, linguistic_decision = build_malayalam_linguistic_correction_candidate(
                current_payload=payload,
            )
            linguistic_metadata["malayalam_linguistic_correction_attempted"] = bool(linguistic_decision.get("attempted", False))
            linguistic_metadata["malayalam_linguistic_correction_applied"] = bool(linguistic_decision.get("applied", False))
            linguistic_metadata["malayalam_linguistic_correction_reason"] = str(linguistic_decision.get("reason", "") or "")
            linguistic_metadata["malayalam_linguistic_correction_blocked_reason"] = str(linguistic_decision.get("blocked_reason", "") or "")
            linguistic_metadata["malayalam_linguistic_correction_backend"] = str(linguistic_decision.get("backend", "") or "")
            payload["metadata"] = linguistic_metadata
            logger.info(
                "[ML_LINGUISTIC_CORRECTION] attempted=%s applied=%s reason=%s blocked_reason=%s backend=%s",
                bool(linguistic_decision.get("attempted", False)),
                bool(linguistic_decision.get("applied", False)),
                linguistic_decision.get("reason", "") or "none",
                linguistic_decision.get("blocked_reason", "") or "none",
                linguistic_decision.get("backend", "") or "none",
            )
            if linguistic_payload is not None and bool(linguistic_decision.get("applied", False)):
                payload = linguistic_payload
                payload_meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
                payload_meta["malayalam_draft_candidate"] = True
                payload_meta["malayalam_linguistic_correction_attempted"] = True
                payload_meta["malayalam_linguistic_correction_applied"] = True
                payload_meta["malayalam_linguistic_correction_reason"] = str(linguistic_decision.get("reason", "") or "")
                payload_meta["malayalam_linguistic_correction_blocked_reason"] = ""
                payload_meta["malayalam_linguistic_correction_backend"] = str(linguistic_decision.get("backend", "") or "")
                payload_meta["malayalam_candidate_stage"] = "linguistic_correction_candidate"
                payload_meta["malayalam_final_candidate_source"] = "linguistic_correction_candidate"
                payload["metadata"] = payload_meta

            final_analysis = _analyze_malayalam_asr_payload(payload)
            final_metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            terminal_failure = bool(final_metadata.get("terminal_malayalam_failure"))
            final_metadata["source_language_fidelity_failed"] = bool(
                terminal_failure or final_analysis.get("source_language_fidelity_failed", False)
            )
            final_metadata["transcript_fidelity_state"] = (
                "source_language_fidelity_failed"
                if terminal_failure
                else str(final_analysis.get("transcript_fidelity_state", "") or "")
            )
            final_metadata["recoverable_malayalam_fidelity_gap"] = bool(
                not terminal_failure
                and final_analysis.get("source_language_fidelity_failed", False)
                and not final_analysis.get("catastrophic_wrong_script_failure", False)
                and bool(final_analysis.get("nonempty_signal", False))
            )
            if bool(final_metadata.get("source_language_fidelity_failed", False)):
                final_metadata["malayalam_candidate_stage"] = "suppressed_fidelity_failed"
                final_metadata["malayalam_final_candidate_source"] = "suppressed_fidelity_failed"
            final_metadata["malayalam_draft_candidate_accepted_as_final"] = bool(
                final_metadata.get("malayalam_final_candidate_source") == "fast_draft_candidate"
                and not bool(final_metadata.get("source_language_fidelity_failed", False))
            )
            payload["metadata"] = final_metadata

        # Garble retry path for Deepgram only.
        if engine == "deepgram" and _is_garbled(payload):
            script_type = detect_script_type(payload.get("text", ""))
            retry_langs: List[str] = candidate_languages_for_script(script_type)
            if req_lang != "auto" and req_lang in retry_langs:
                retry_langs = [req_lang] + [x for x in retry_langs if x != req_lang]
            max_probe_attempts = _asr_max_retries()
            retry_langs = retry_langs[:max_probe_attempts]
            logger.warning(
                "Deepgram transcript appears garbled; retrying with restricted languages=%s",
                retry_langs
            )
            fallback_triggered = True
            fallback_reason = "deepgram_garbled_retry"
            repaired = None
            for lang in retry_langs:
                try:
                    attempt = _call_deepgram_with_retries(
                        preprocessed_path,
                        language=lang,
                        detect_language=False,
                    )
                    if not _is_garbled(attempt):
                        repaired = attempt
                        chosen_lang = normalize_language_code(attempt.get("language"), default=lang, allow_auto=False)
                        break
                except DeepgramError:
                    continue
            if repaired is not None:
                payload = repaired
            else:
                if _asr_reject_on_garble():
                    quality_gate_passed = False
                    raise ValueError(
                        "ASR output remained garbled after Deepgram retry with restricted language candidates."
                    )
                logger.warning(
                    "ASR output remained garbled after retries; proceeding because ASR_REJECT_ON_GARBLE=False"
                )

        # Quality guard: if Deepgram returns too little text for the duration, fallback to local Whisper.
        if engine == "deepgram" and _is_low_content_for_duration(payload.get("text", ""), duration):
            fallback_lang = normalize_language_code(payload.get("language"), default=chosen_lang, allow_auto=False)
            if fallback_lang == "auto":
                fallback_lang = "en"
            logger.warning(
                "Deepgram transcript appears low-content for duration (lang=%s, duration=%.1fs). "
                "Falling back to secondary ASR.",
                fallback_lang,
                duration
            )
            fallback_triggered = True
            fallback_reason = "deepgram_low_content"
            fallback_payload = None
            if _asr_use_groq_fallback():
                try:
                    fallback_payload = _transcribe_with_groq_whisper(preprocessed_path, source_type, fallback_lang)
                    engine = "groq_whisper"
                except Exception as e:
                    if _is_rate_limit_error(e):
                        _block_groq_fallback_due_to_rate_limit()
                    logger.warning("Groq Whisper secondary fallback failed: %s", e)
            if fallback_payload is None:
                if _render_demo_safe_asr_mode():
                    quality_gate_passed = False
                    raise _render_demo_safe_transcription_error(
                        fallback_lang,
                        reason="deepgram_low_content_remote_only",
                    )
                fallback_payload = _transcribe_with_local_whisper(preprocessed_path, source_type, fallback_lang)
                engine = "whisper_local"
                route_reason = "deepgram_fallback_to_local"
            fallback_payload.setdefault("metadata", {})
            if isinstance(fallback_payload.get("metadata"), dict):
                fallback_payload["metadata"]["fallback_reason"] = "deepgram_low_content"
            payload = fallback_payload

        # Final quality gate: if chosen engine is still low-content, force local whisper once.
        terminal_malayalam_failure = bool(
            isinstance(payload.get("metadata"), dict)
            and payload.get("metadata", {}).get("terminal_malayalam_failure")
        )
        if _is_low_content_for_duration(payload.get("text", ""), duration) and not terminal_malayalam_failure:
            if engine != "whisper_local":
                if _render_demo_safe_asr_mode():
                    quality_gate_passed = False
                    raise _render_demo_safe_transcription_error(
                        normalize_language_code(payload.get("language"), default=chosen_lang, allow_auto=False),
                        reason="final_quality_gate_low_content_remote_only",
                    )
                forced_lang = normalize_language_code(payload.get("language"), default=chosen_lang, allow_auto=False)
                if forced_lang == "auto":
                    forced_lang = "en"
                logger.warning(
                    "Final ASR quality gate failed (engine=%s, lang=%s). Forcing local Whisper retry.",
                    engine,
                    forced_lang
                )
                payload = _transcribe_with_local_whisper(preprocessed_path, source_type, forced_lang)
                engine = "whisper_local"
                fallback_triggered = True
                fallback_reason = "final_quality_gate_low_content"
                if route_reason == "deepgram_supported_language":
                    route_reason = "deepgram_fallback_to_local"
            if _is_low_content_for_duration(payload.get("text", ""), duration):
                quality_gate_passed = False
                raise ValueError(
                    "ASR transcript quality too low for audio duration after fallback chain. "
                    "Rejecting transcript to prevent low-quality summaries/chat."
                )

        elapsed = time.time() - start
        detected_lang = normalize_language_code(payload.get("language"), default=chosen_lang, allow_auto=False)
        payload["language"] = detected_lang
        primary_model_used = str(route_decision.get("model", "") or "")
        fallback_model_used = ""
        if chosen_lang == "ml" and fallback_triggered and engine == "whisper_local":
            fallback_model_used = str(payload.get("metadata", {}).get("actual_local_model_name", "") or "large-v3")
        elif chosen_lang == "ml" and fallback_triggered and engine == "groq_whisper":
            fallback_model_used = "whisper-large-v3"
        if chosen_lang == "ml":
            payload = _attach_malayalam_strategy_metadata(
                payload,
                route_decision=route_decision,
                preprocess_meta=preprocess_meta,
                primary_model_used=primary_model_used,
                fallback_model_used=fallback_model_used,
                retry_model_used=str(payload.get("metadata", {}).get("second_pass_asr_model", "") or ""),
            )
        payload = _with_engine_metadata(
            payload,
            engine,
            req_lang,
            duration,
            elapsed,
            route_reason=route_reason,
            fallback_triggered=fallback_triggered,
            fallback_reason=fallback_reason,
            quality_gate_passed=quality_gate_passed,
            route_decision=route_decision,
        )
        logger.info(
            "[ASR] engine=%s provider_name=%s req_lang=%s detected=%s duration=%.2fs rtf=%.3f fallback_triggered=%s fallback_reason=%s actual_local_model_name=%s model_reused=%s asr_latency_seconds=%.3f transcript_quality_gate_passed=%s",
            engine,
            _provider_name_for_engine(engine),
            req_lang,
            detected_lang,
            duration,
            float(payload.get("metadata", {}).get("rtf", 0.0)),
            fallback_triggered,
            fallback_reason or "",
            payload.get("metadata", {}).get("actual_local_model_name"),
            payload.get("metadata", {}).get("model_reused"),
            float(payload.get("metadata", {}).get("latency_seconds", 0.0)),
            bool(payload.get("metadata", {}).get("transcript_quality_gate_passed", True)),
        )
        _ASR_PROVIDER_PRIOR_CACHE[(detected_lang, engine)] = {
            "quality_score": float(payload.get("transcript_quality_score", 0.0) or 0.0),
            "latency_seconds": float(payload.get("metadata", {}).get("latency_seconds", 0.0) or 0.0),
        }
        return payload

    finally:
        if preprocessed_path and os.path.exists(preprocessed_path):
            try:
                os.remove(preprocessed_path)
            except Exception:
                pass


def _chunk_value(chunk: ChunkMetadata | Dict[str, object], name: str, default=None):
    if isinstance(chunk, dict):
        return chunk.get(name, default)
    return getattr(chunk, name, default)


def _is_valid_chunk_payload(payload: Dict, requested_language: str) -> bool:
    normalized_requested = normalize_language_code(requested_language, default="auto", allow_auto=True)
    detected_language = normalize_language_code(
        payload.get("language"),
        default=normalized_requested,
        allow_auto=False,
    )
    payload_metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    if bool(payload_metadata.get("terminal_malayalam_failure")):
        return False
    if bool(payload_metadata.get("source_language_fidelity_failed")):
        return False
    if normalized_requested != "ml" and detected_language != "ml":
        return True
    analysis = _analyze_malayalam_asr_payload(payload)
    if bool(analysis.get("source_language_fidelity_failed", False)):
        return False
    trusted_visible_word_count = int(analysis.get("trusted_visible_word_count", 0) or 0)
    trusted_display_unit_count = int(analysis.get("trusted_display_unit_count", 0) or 0)
    readability = float(analysis.get("readability", 0.0) or 0.0)
    lexical_trust = float(analysis.get("lexical_trust", 0.0) or 0.0)
    has_text = bool(str(payload.get("text", "") or "").strip())
    if (
        has_text
        and trusted_visible_word_count == 0
        and trusted_display_unit_count == 0
        and readability < 0.18
        and lexical_trust < 0.12
    ):
        return False
    return True


def _stitch_chunk_payloads(
    chunk_results: List[Dict[str, object]],
    requested_language: str,
) -> Dict:
    normalized_requested = normalize_language_code(requested_language, default="auto", allow_auto=True)
    stitched_segments: List[Dict] = []
    stitched_word_timestamps: List[Dict] = []
    stitched_text_parts: List[str] = []
    successful_payloads: List[Dict] = []
    has_fidelity_gaps = False

    for chunk_result in chunk_results:
        status = str(chunk_result.get("status", "") or "")
        payload = chunk_result.get("payload") if isinstance(chunk_result.get("payload"), dict) else {}
        chunk = chunk_result.get("chunk")
        chunk_id = int(_chunk_value(chunk, "chunk_id", len(chunk_results)) or 0)
        chunk_start = float(_chunk_value(chunk, "start_s", 0.0) or 0.0)
        chunk_end = float(_chunk_value(chunk, "end_s", chunk_start) or chunk_start)
        if status != "ok" or not payload:
            has_fidelity_gaps = True
            logger.warning(
                "[ML_CHUNK_FIDELITY_FAIL] chunk_id=%s start=%.2fs end=%.2fs",
                chunk_id,
                chunk_start,
                chunk_end,
            )
            continue

        successful_payloads.append(payload)
        text = str(payload.get("text", "") or "").strip()
        if text:
            stitched_text_parts.append(text)
        for seg in list(payload.get("segments", []) or []):
            if not isinstance(seg, dict):
                continue
            updated = dict(seg)
            updated["id"] = len(stitched_segments)
            updated["start"] = round(chunk_start + float(seg.get("start", 0.0) or 0.0), 3)
            updated["end"] = round(chunk_start + float(seg.get("end", seg.get("start", 0.0)) or 0.0), 3)
            stitched_segments.append(updated)
        for word in list(payload.get("word_timestamps", []) or []):
            if not isinstance(word, dict):
                stitched_word_timestamps.append(word)
                continue
            updated_word = dict(word)
            if "start" in updated_word:
                updated_word["start"] = round(chunk_start + float(updated_word.get("start", 0.0) or 0.0), 3)
            if "end" in updated_word:
                updated_word["end"] = round(chunk_start + float(updated_word.get("end", updated_word.get("start", 0.0)) or 0.0), 3)
            stitched_word_timestamps.append(updated_word)

    base_payload = dict(successful_payloads[0]) if successful_payloads else {}
    base_metadata = dict(base_payload.get("metadata", {}) or {}) if isinstance(base_payload.get("metadata"), dict) else {}
    base_metadata["chunk_manifest_used"] = True
    base_metadata["chunk_count"] = len(chunk_results)
    base_metadata["successful_chunk_count"] = len(successful_payloads)
    base_metadata["failed_chunk_count"] = sum(1 for item in chunk_results if str(item.get("status", "") or "") != "ok")
    base_metadata["has_fidelity_gaps"] = has_fidelity_gaps
    if not successful_payloads and normalized_requested == "ml":
        base_metadata["source_language_fidelity_failed"] = True
        base_metadata["transcript_fidelity_state"] = "source_language_fidelity_failed"

    stitched_language = normalize_language_code(
        base_payload.get("language"),
        default=("ml" if normalized_requested == "ml" else normalized_requested),
        allow_auto=False,
    )
    return {
        **base_payload,
        "text": " ".join(part for part in stitched_text_parts if part).strip(),
        "segments": stitched_segments,
        "word_timestamps": stitched_word_timestamps,
        "language": stitched_language,
        "metadata": base_metadata,
        "has_fidelity_gaps": has_fidelity_gaps,
    }


def _build_terminal_malayalam_failure_payload(
    *,
    route_decision: Optional[Dict[str, object]] = None,
    route_reason: str = "",
    failure_reason: str = "",
    detection_confidence: float = 0.0,
    engine: str = "whisper_local",
) -> Dict:
    selected_model = str((route_decision or {}).get("model", "") or _get_malayalam_model())
    metadata = {
        "asr_provider_used": engine,
        "asr_engine_used": engine,
        "asr_route_reason": route_reason,
        "selected_model": selected_model,
        "actual_local_model_name": selected_model,
        "resolved_local_model_name": selected_model,
        "source_language_fidelity_failed": True,
        "transcript_fidelity_state": "source_language_fidelity_failed",
        "malayalam_candidate_stage": "suppressed_fidelity_failed",
        "malayalam_final_candidate_source": "suppressed_fidelity_failed",
        "terminal_malayalam_failure": True,
        "terminal_malayalam_failure_reason": failure_reason or "malayalam_local_only_terminal_failure",
        "language_detection_confidence": float(detection_confidence or 0.0),
        "transcript_quality_gate_passed": False,
        "fallback_triggered": False,
        "fallback_reason": failure_reason or "",
    }
    return {
        "text": "",
        "segments": [],
        "word_timestamps": [],
        "language": "ml",
        "language_probability": float(detection_confidence or 0.0),
        "confidence": float(detection_confidence or 0.0),
        "transcript_quality_score": 0.0,
        "metadata": metadata,
        "has_fidelity_gaps": True,
    }


def transcribe_video_router(
    audio_path: str = "",
    source_type: str = "generic",
    requested_language: Optional[str] = None,
    chunks: Optional[List[ChunkMetadata]] = None,
) -> Dict:
    if chunks:
        normalized_requested = normalize_language_code(requested_language, default="auto", allow_auto=True)
        locked_requested_language = normalized_requested
        manifest_duration_seconds = max(
            float(_chunk_value(chunk, "end_s", 0.0) or 0.0)
            for chunk in chunks
        ) if chunks else 0.0
        chunk_results: List[Dict[str, object]] = []
        for chunk in chunks:
            chunk_path = str(_chunk_value(chunk, "path", "") or "")
            if not chunk_path:
                chunk_results.append({"chunk": chunk, "status": "fidelity_failed", "payload": {}})
                continue
            payload = _transcribe_video_router_single(
                audio_path=chunk_path,
                source_type=source_type,
                requested_language=locked_requested_language,
                already_preprocessed=True,
                routing_duration_seconds=manifest_duration_seconds,
            )
            detected_language = normalize_language_code(
                payload.get("language"),
                default=locked_requested_language,
                allow_auto=False,
            )
            detected_confidence = float(
                payload.get("language_probability", payload.get("confidence", 0.0)) or 0.0
            )
            if (
                normalized_requested == "auto"
                and locked_requested_language == "auto"
                and detected_language == "ml"
                and detected_confidence >= 0.85
            ):
                locked_requested_language = "ml"
                logger.info(
                    "[ASR_CHUNK_LANGUAGE_LOCK] language=%s confidence=%.3f chunk_id=%s",
                    detected_language,
                    detected_confidence,
                    int(_chunk_value(chunk, "chunk_id", len(chunk_results)) or 0),
                )
            if not _is_valid_chunk_payload(payload, normalized_requested):
                chunk_results.append({"chunk": chunk, "status": "fidelity_failed", "payload": payload})
                continue
            chunk_results.append({"chunk": chunk, "status": "ok", "payload": payload})
        return _stitch_chunk_payloads(chunk_results, normalized_requested)

    return _transcribe_video_router_single(
        audio_path=audio_path,
        source_type=source_type,
        requested_language=requested_language,
        already_preprocessed=False,
    )
