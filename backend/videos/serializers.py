"""
Serializers for video processing API
"""

import os
import logging
import json
import re
from hashlib import sha1
from rest_framework import serializers
from .models import Video, Transcript, Summary, HighlightSegment, ShortVideo, ProcessingTask
from .summary_schema import build_structured_summary, default_structured_summary, structured_summary_cache_key
from .processing_metadata import build_processing_metadata
from .translation import (
    build_safe_english_view_text,
    build_safe_english_view_structured_summary,
    build_english_view_source_hash,
    evaluate_english_view_policy,
    is_english_view_cache_valid,
    build_english_view_cache_entry,
)
from .utils_metrics import evaluate_summary_quality

logger = logging.getLogger(__name__)


def _structured_summary_has_content(payload):
    if not isinstance(payload, dict):
        return False
    return bool(
        str(payload.get("tldr", "") or "").strip()
        or list(payload.get("key_points") or [])
        or list(payload.get("action_items") or [])
        or list(payload.get("chapters") or [])
    )


def _degraded_safe_malayalam_payload(transcript):
    warning_message = ""
    json_data = transcript.json_data if isinstance(getattr(transcript, "json_data", None), dict) else {}
    if isinstance(json_data, dict):
        warning_message = str(
            json_data.get("transcript_warning_message")
            or "Malayalam transcript quality was too low for reliable summarization."
        ).strip()
    if not warning_message:
        warning_message = "Malayalam transcript quality was too low for reliable summarization."
    return {
        "tldr": "This video appears to contain Malayalam speech, but the transcript quality is too low for reliable summarization.",
        "key_points": [],
        "action_items": [],
        "chapters": [],
        "participants": [],
        "summary_state": "degraded_safe",
        "warning_message": warning_message,
    }


def _fidelity_failed_malayalam_summary_payload(transcript):
    json_data = transcript.json_data if isinstance(getattr(transcript, "json_data", None), dict) else {}
    warning = str(
        json_data.get("transcript_warning_message")
        or "Malayalam speech could not be transcribed faithfully enough for safe display."
    )
    payload = default_structured_summary()
    payload.update({
        "summary_state": "blocked_unfaithful_source",
        "summary_blocked_reason": "malayalam_source_fidelity_failed",
        "summary_warning_message": warning,
        "warning_message": warning,
        "summary_original_language": "ml",
        "summary_english_view_available": False,
        "summary_translation_state": "blocked",
        "summary_translation_warning": warning,
        "summary_translation_blocked_reason": "source_language_fidelity_failed",
        "summary_current_available_views": ["original"],
    })
    return payload


def _blocked_transcript_english_view_payload(source_language: str, blocked_reason: str, warning: str = ""):
    return {
        "original_language": source_language,
        "translated_language": "en",
        "english_view_available": False,
        "english_view_text": "",
        "translation_state": "blocked",
        "translation_blocked_reason": blocked_reason,
        "translation_warning": warning or "",
        "current_available_views": ["original"],
    }


def _blocked_summary_english_view_payload(payload, source_language: str, blocked_reason: str, warning: str = ""):
    enriched = dict(payload or {})
    enriched.update({
        "summary_original_language": source_language,
        "summary_english_view_available": False,
        "summary_translation_state": "blocked",
        "summary_translation_warning": warning or "",
        "summary_translation_blocked_reason": blocked_reason,
        "summary_current_available_views": ["original"],
    })
    return enriched


def _is_stale_bad_malayalam_transcript_payload(json_data, transcript=None):
    if not isinstance(json_data, dict):
        return False
    source_language = str(
        json_data.get("language")
        or getattr(transcript, "transcript_language", "")
        or getattr(transcript, "language", "")
        or ""
    ).strip().lower()
    if source_language != "ml":
        return False
    quality_metrics = json_data.get("quality_metrics", {}) if isinstance(json_data.get("quality_metrics", {}), dict) else {}
    dominant_script = str(
        quality_metrics.get("dominant_script")
        or quality_metrics.get("script_type")
        or ""
    ).strip().lower()
    malayalam_token_coverage = float(quality_metrics.get("malayalam_token_coverage", 0.0) or 0.0)
    trusted_display_unit_count = int(
        json_data.get("trusted_display_unit_count", quality_metrics.get("trusted_display_unit_count", 0)) or 0
    )
    trusted_visible_word_count = int(
        json_data.get("trusted_visible_word_count", quality_metrics.get("trusted_visible_word_count", 0)) or 0
    )
    return bool(
        dominant_script
        and dominant_script != "malayalam"
        and malayalam_token_coverage <= 0.01
        and trusted_display_unit_count == 0
        and trusted_visible_word_count == 0
    )


def _transcript_english_view_source_payload(transcript):
    json_data = transcript.json_data if isinstance(transcript.json_data, dict) else {}
    source_language = str(getattr(transcript, "transcript_language", "") or getattr(transcript, "language", "") or "").strip().lower() or "en"
    transcript_state = str(json_data.get("transcript_state", "") or "").strip().lower()
    visible_text = str(json_data.get("display_readable_transcript") or json_data.get("readable_transcript") or "").strip()
    canonical_english = str(getattr(transcript, "transcript_canonical_en_text", "") or "").strip()
    low_evidence_malayalam = bool(json_data.get("low_evidence_malayalam", False))
    source_language_fidelity_failed = bool(
        json_data.get("source_language_fidelity_failed", False)
        or transcript_state == "source_language_fidelity_failed"
        or str(json_data.get("final_malayalam_fidelity_decision", "") or "").strip().lower() == "source_language_fidelity_failed"
        or bool(json_data.get("catastrophic_latin_substitution_failure", False))
        or _is_stale_bad_malayalam_transcript_payload(json_data, transcript)
    )
    warning = str(json_data.get("transcript_warning_message", "") or "").strip()
    return {
        "source_language": source_language,
        "transcript_state": transcript_state,
        "visible_text": visible_text,
        "canonical_english": canonical_english,
        "low_evidence_malayalam": low_evidence_malayalam,
        "source_language_fidelity_failed": source_language_fidelity_failed,
        "warning": warning,
    }


def _persist_transcript_english_view_cache(transcript, cache_entry, *, source_hash: str):
    if not transcript or not isinstance(getattr(transcript, "json_data", None), dict):
        logger.info("[EN_VIEW_PERSIST_SKIP] kind=transcript transcript_id=%s reason=missing_json_data", getattr(transcript, "id", ""))
        return
    updated = dict(transcript.json_data)
    updated["english_view_cache"] = dict(cache_entry)
    payload = cache_entry.get("payload", {}) if isinstance(cache_entry, dict) else {}
    updated["english_view_available"] = bool(payload.get("english_view_available", False))
    updated["translation_state"] = str(payload.get("translation_state", "") or "")
    updated["translation_blocked_reason"] = str(payload.get("translation_blocked_reason", "") or "")
    updated["current_available_views"] = list(payload.get("current_available_views") or ["original"])
    Transcript.objects.filter(pk=transcript.pk).update(json_data=updated)
    transcript.json_data = updated
    logger.info(
        "[EN_VIEW_PERSIST_RESULT] kind=transcript transcript_id=%s source_hash=%s available=%s",
        getattr(transcript, "id", ""),
        source_hash,
        bool(payload.get("english_view_available", False)),
    )


def _summary_english_view_source_payload(payload, transcript):
    json_data = transcript.json_data if isinstance(transcript.json_data, dict) else {}
    return {
        "source_language": str(getattr(transcript, "transcript_language", "") or getattr(transcript, "language", "") or "").strip().lower() or "en",
        "transcript_state": str(json_data.get("transcript_state", "") or "").strip().lower(),
        "low_evidence_malayalam": bool(json_data.get("low_evidence_malayalam", False)),
        "source_language_fidelity_failed": bool(
            json_data.get("source_language_fidelity_failed", False)
            or str(json_data.get("transcript_state", "") or "").strip().lower() == "source_language_fidelity_failed"
            or str(json_data.get("final_malayalam_fidelity_decision", "") or "").strip().lower() == "source_language_fidelity_failed"
            or bool(json_data.get("catastrophic_latin_substitution_failure", False))
            or _is_stale_bad_malayalam_transcript_payload(json_data, transcript)
        ),
        "warning": str(payload.get("warning_message", "") or json_data.get("transcript_warning_message", "") or "").strip(),
        "summary_payload": {
            "tldr": str(payload.get("tldr", "") or "").strip(),
            "key_points": list(payload.get("key_points") or []),
            "action_items": list(payload.get("action_items") or []),
            "chapters": list(payload.get("chapters") or []),
            "summary_state": str(payload.get("summary_state", "") or "").strip(),
        },
    }


def _persist_structured_summary_english_view_cache(transcript, cache_entry, *, source_hash: str):
    if not transcript or not isinstance(getattr(transcript, "json_data", None), dict):
        logger.info("[EN_VIEW_PERSIST_SKIP] kind=summary transcript_id=%s reason=missing_json_data", getattr(transcript, "id", ""))
        return
    updated = dict(transcript.json_data)
    structured_cache = updated.get("structured_summary_cache", {})
    if not isinstance(structured_cache, dict):
        structured_cache = {}
    structured_cache["english_view_cache"] = dict(cache_entry)
    updated["structured_summary_cache"] = structured_cache
    Transcript.objects.filter(pk=transcript.pk).update(json_data=updated)
    transcript.json_data = updated
    payload = cache_entry.get("payload", {}) if isinstance(cache_entry, dict) else {}
    logger.info(
        "[EN_VIEW_PERSIST_RESULT] kind=summary transcript_id=%s source_hash=%s available=%s",
        getattr(transcript, "id", ""),
        source_hash,
        bool(payload.get("summary_english_view_available", False)),
    )


def _should_allow_legacy_summary_fallback(inputs):
    transcript_state = str((inputs or {}).get("transcript_state", "") or "").strip().lower()
    transcript_language = str((inputs or {}).get("transcript_language", "") or "").strip().lower()
    if transcript_state == "degraded" and transcript_language == "ml":
        return False
    return True


def _is_pending_malayalam_final_state(source_language, transcript_state):
    return bool(
        str(source_language or "").strip().lower() == "ml"
        and str(transcript_state or "").strip().lower() in {"draft", "processing", "pending"}
    )


def _safe_transcript_english_view(transcript):
    source_info = _transcript_english_view_source_payload(transcript)
    source_language = source_info["source_language"]
    transcript_state = source_info["transcript_state"]
    visible_text = source_info["visible_text"]
    canonical_english = source_info["canonical_english"]
    low_evidence_malayalam = bool(source_info["low_evidence_malayalam"])
    source_language_fidelity_failed = bool(source_info.get("source_language_fidelity_failed", False))
    if _is_pending_malayalam_final_state(source_language, transcript_state):
        logger.info(
            "[ML_EN_VIEW_PENDING_FINAL_STATE] kind=transcript transcript_id=%s language=%s state=%s",
            getattr(transcript, "id", ""),
            source_language,
            transcript_state or "pending",
        )
        return {
            "original_language": source_language,
            "translated_language": "en",
            "english_view_available": False,
            "english_view_text": "",
            "translation_state": "pending",
            "translation_blocked_reason": "pending_final_malayalam_state",
            "translation_warning": "Malayalam transcript is still being finalized.",
            "current_available_views": ["original"],
        }
    if source_language == "ml" and source_language_fidelity_failed:
        logger.info(
            "[ML_EN_VIEW_SKIPPED_FIDELITY_FAILED_EARLY] kind=transcript transcript_id=%s",
            getattr(transcript, "id", ""),
        )
        payload = _blocked_transcript_english_view_payload(
            source_language,
            "source_language_fidelity_failed",
            str(source_info["warning"] or ""),
        )
        if isinstance(transcript.json_data, dict):
            updated = dict(transcript.json_data)
            updated["english_view_available"] = False
            updated["translation_state"] = "blocked"
            prior_blocked_reason = str(updated.get("translation_blocked_reason", "") or "")
            existing_cache = updated.get("english_view_cache", {})
            updated["translation_blocked_reason"] = "source_language_fidelity_failed"
            updated["current_available_views"] = ["original"]
            should_sync = bool(existing_cache) or prior_blocked_reason != "source_language_fidelity_failed"
            if should_sync:
                Transcript.objects.filter(pk=transcript.pk).update(json_data=updated)
            transcript.json_data = updated
        transcript._english_view_payload_cache = {"_source_hash": "fidelity_failed", "payload": payload}
        return payload
    if source_language == "ml" and transcript_state == "degraded" and low_evidence_malayalam:
        logger.info(
            "[ML_EN_VIEW_SKIPPED_FIDELITY_FAILED_EARLY] kind=transcript transcript_id=%s reason=low_evidence_malayalam",
            getattr(transcript, "id", ""),
        )
        payload = _blocked_transcript_english_view_payload(
            source_language,
            "degraded_safe_translation_blocked",
            str(source_info["warning"] or ""),
        )
        if isinstance(transcript.json_data, dict):
            updated = dict(transcript.json_data)
            updated["english_view_available"] = False
            updated["translation_state"] = "blocked"
            prior_blocked_reason = str(updated.get("translation_blocked_reason", "") or "")
            existing_cache = updated.get("english_view_cache", {})
            updated["translation_blocked_reason"] = "degraded_safe_translation_blocked"
            updated["current_available_views"] = ["original"]
            should_sync = bool(existing_cache) or prior_blocked_reason != "degraded_safe_translation_blocked"
            if should_sync:
                Transcript.objects.filter(pk=transcript.pk).update(json_data=updated)
            transcript.json_data = updated
        transcript._english_view_payload_cache = {"_source_hash": "low_evidence_malayalam", "payload": payload}
        return payload
    source_hash = build_english_view_source_hash("transcript", source_info)
    memo = getattr(transcript, "_english_view_payload_cache", None)
    if isinstance(memo, dict) and str(memo.get("_source_hash", "") or "") == source_hash:
        return dict(memo.get("payload") or {})
    json_data = transcript.json_data if isinstance(transcript.json_data, dict) else {}
    cached_entry = json_data.get("english_view_cache", {}) if isinstance(json_data.get("english_view_cache", {}), dict) else {}
    logger.info(
        "[EN_VIEW_HASH_COMPARE] kind=transcript transcript_id=%s source_hash=%s cache_hash=%s",
        getattr(transcript, "id", ""),
        source_hash,
        str(cached_entry.get("english_view_source_hash", "") or ""),
    )
    if is_english_view_cache_valid(cached_entry, source_hash):
        payload = dict(cached_entry.get("payload") or {})
        if bool(payload.get("english_view_available")) and not str(payload.get("english_view_text", "") or "").strip():
            logger.info("[EN_VIEW_STALE_BLOCKED] kind=transcript transcript_id=%s reason=missing_payload_text", getattr(transcript, "id", ""))
        else:
            transcript._english_view_payload_cache = {"_source_hash": source_hash, "payload": payload}
            return payload
    elif cached_entry:
        logger.info(
            "[EN_VIEW_INVALIDATE] kind=transcript transcript_id=%s reason=source_changed old_hash=%s new_hash=%s",
            getattr(transcript, "id", ""),
            str(cached_entry.get("english_view_source_hash", "") or ""),
            source_hash,
        )
    logger.info(
        "[EN_VIEW_REBUILD_REQUIRED] kind=transcript transcript_id=%s source_hash=%s cache_miss=%s",
        getattr(transcript, "id", ""),
        source_hash,
        not bool(cached_entry),
    )
    policy = evaluate_english_view_policy(
        content_kind="transcript",
        source_language=source_language,
        source_state=transcript_state,
        has_grounded_text=bool(visible_text or (canonical_english and not source_language_fidelity_failed)),
        low_evidence_malayalam=bool(source_language == "ml" and transcript_state == "degraded" and (low_evidence_malayalam or not visible_text or source_language_fidelity_failed)),
        degraded_low_evidence_reason="source_language_fidelity_failed" if source_language_fidelity_failed else "degraded_safe_translation_blocked",
    )
    if source_language_fidelity_failed:
        policy = {
            **policy,
            "allow_translation": False,
            "blocked_reason": "source_language_fidelity_failed",
            "current_available_views": ["original"],
        }
    logger.info(
        "[TRANSCRIPT_EN_VIEW_REQUEST] transcript_id=%s language=%s state=%s allow=%s blocked_reason=%s",
        getattr(transcript, "id", ""),
        source_language,
        transcript_state,
        policy["allow_translation"],
        policy["blocked_reason"],
    )
    payload = build_safe_english_view_text(
        visible_text or canonical_english,
        source_language,
        allow_translation=bool(policy["allow_translation"]),
        blocked_reason=str(policy["blocked_reason"] or ""),
        warning=str(source_info["warning"] or ""),
        preserve_format=True,
        translated_text=canonical_english if source_language != "en" and canonical_english else "",
    )
    payload["current_available_views"] = list(payload.get("current_available_views") or policy.get("current_available_views") or ["original"])
    if payload.get("english_view_available"):
        logger.info(
            "[TRANSCRIPT_EN_VIEW_RESULT] transcript_id=%s state=%s mode=%s",
            getattr(transcript, "id", ""),
            transcript_state,
            payload.get("translation_state", ""),
        )
    else:
        logger.info(
            "[TRANSCRIPT_EN_VIEW_BLOCKED] transcript_id=%s state=%s reason=%s",
            getattr(transcript, "id", ""),
            transcript_state,
            payload.get("translation_blocked_reason", ""),
        )
    cache_entry = build_english_view_cache_entry(
        payload,
        source_hash=source_hash,
        build_reason="transcript_serializer",
        source_language=source_language,
        policy=policy,
    )
    logger.info(
        "[EN_VIEW_PERSIST] kind=transcript transcript_id=%s source_hash=%s build_reason=%s",
        getattr(transcript, "id", ""),
        source_hash,
        "transcript_serializer",
    )
    _persist_transcript_english_view_cache(transcript, cache_entry, source_hash=source_hash)
    transcript._english_view_payload_cache = {"_source_hash": source_hash, "payload": payload}
    return payload


def _augment_structured_summary_with_english_view(payload, transcript):
    if not isinstance(payload, dict) or not transcript:
        return payload
    source_info = _summary_english_view_source_payload(payload, transcript)
    source_hash = build_english_view_source_hash("summary", source_info)
    memo = getattr(transcript, "_summary_english_view_payload_cache", None)
    if isinstance(memo, dict) and str(memo.get("_source_hash", "") or "") == source_hash:
        return dict(memo.get("payload") or {})
    json_data = transcript.json_data if isinstance(transcript.json_data, dict) else {}
    source_language = source_info["source_language"]
    transcript_state = source_info["transcript_state"]
    low_evidence_malayalam = bool(source_info["low_evidence_malayalam"])
    source_language_fidelity_failed = bool(source_info.get("source_language_fidelity_failed", False))
    if _is_pending_malayalam_final_state(source_language, transcript_state):
        logger.info(
            "[ML_EN_VIEW_PENDING_FINAL_STATE] kind=summary transcript_id=%s language=%s state=%s",
            getattr(transcript, "id", ""),
            source_language,
            transcript_state or "pending",
        )
        enriched = dict(payload)
        enriched.update({
            "summary_original_language": source_language,
            "summary_english_view_available": False,
            "summary_translation_state": "pending",
            "summary_translation_warning": "Malayalam transcript is still being finalized.",
            "summary_translation_blocked_reason": "pending_final_malayalam_state",
            "summary_current_available_views": ["original"],
        })
        return enriched
    if source_language == "ml" and source_language_fidelity_failed:
        logger.info(
            "[ML_EN_VIEW_SKIPPED_FIDELITY_FAILED_EARLY] kind=summary transcript_id=%s",
            getattr(transcript, "id", ""),
        )
        enriched = _blocked_summary_english_view_payload(
            payload,
            source_language,
            "source_language_fidelity_failed",
            str(source_info["warning"] or ""),
        )
        if isinstance(transcript.json_data, dict):
            updated = dict(transcript.json_data)
            structured_cache = updated.get("structured_summary_cache", {})
            if not isinstance(structured_cache, dict):
                structured_cache = {}
            prior_summary_blocked_reason = str(
                (((structured_cache.get("english_view_cache", {}) or {}).get("payload", {}) or {}).get("summary_translation_blocked_reason", "") or "")
            )
            had_summary_cache = bool(structured_cache.get("english_view_cache"))
            structured_cache["english_view_cache"] = {
                "english_view_valid": False,
                "payload": {
                    "summary_english_view_available": False,
                    "summary_translation_state": "blocked",
                    "summary_translation_blocked_reason": "source_language_fidelity_failed",
                },
            }
            updated["structured_summary_cache"] = structured_cache
            should_sync = had_summary_cache or prior_summary_blocked_reason != "source_language_fidelity_failed"
            if should_sync:
                Transcript.objects.filter(pk=transcript.pk).update(json_data=updated)
            transcript.json_data = updated
        transcript._summary_english_view_payload_cache = {"_source_hash": "fidelity_failed", "payload": enriched}
        return enriched
    if source_language == "ml" and transcript_state == "degraded" and low_evidence_malayalam:
        logger.info(
            "[ML_EN_VIEW_SKIPPED_FIDELITY_FAILED_EARLY] kind=summary transcript_id=%s reason=low_evidence_malayalam",
            getattr(transcript, "id", ""),
        )
        enriched = _blocked_summary_english_view_payload(
            payload,
            source_language,
            "low_evidence_source_language",
            str(source_info["warning"] or ""),
        )
        if isinstance(transcript.json_data, dict):
            updated = dict(transcript.json_data)
            structured_cache = updated.get("structured_summary_cache", {})
            if not isinstance(structured_cache, dict):
                structured_cache = {}
            prior_summary_blocked_reason = str(
                (((structured_cache.get("english_view_cache", {}) or {}).get("payload", {}) or {}).get("summary_translation_blocked_reason", "") or "")
            )
            had_summary_cache = bool(structured_cache.get("english_view_cache"))
            structured_cache["english_view_cache"] = {
                "english_view_valid": False,
                "payload": {
                    "summary_english_view_available": False,
                    "summary_translation_state": "blocked",
                    "summary_translation_blocked_reason": "low_evidence_source_language",
                },
            }
            updated["structured_summary_cache"] = structured_cache
            should_sync = had_summary_cache or prior_summary_blocked_reason != "low_evidence_source_language"
            if should_sync:
                Transcript.objects.filter(pk=transcript.pk).update(json_data=updated)
            transcript.json_data = updated
        transcript._summary_english_view_payload_cache = {"_source_hash": "low_evidence_malayalam", "payload": enriched}
        return enriched
    structured_cache = json_data.get("structured_summary_cache", {}) if isinstance(json_data.get("structured_summary_cache", {}), dict) else {}
    cached_entry = structured_cache.get("english_view_cache", {}) if isinstance(structured_cache.get("english_view_cache", {}), dict) else {}
    logger.info(
        "[EN_VIEW_HASH_COMPARE] kind=summary transcript_id=%s source_hash=%s cache_hash=%s",
        getattr(transcript, "id", ""),
        source_hash,
        str(cached_entry.get("english_view_source_hash", "") or ""),
    )
    if is_english_view_cache_valid(cached_entry, source_hash):
        meta = dict(cached_entry.get("payload") or {})
        if bool(meta.get("summary_english_view_available")) and not isinstance(meta.get("english_view_structured_summary"), dict):
            logger.info("[EN_VIEW_STALE_BLOCKED] kind=summary transcript_id=%s reason=missing_structured_payload", getattr(transcript, "id", ""))
        else:
            enriched = dict(payload)
            enriched.update(meta)
            transcript._summary_english_view_payload_cache = {"_source_hash": source_hash, "payload": enriched}
            return enriched
    elif cached_entry:
        logger.info(
            "[EN_VIEW_INVALIDATE] kind=summary transcript_id=%s reason=source_changed old_hash=%s new_hash=%s",
            getattr(transcript, "id", ""),
            str(cached_entry.get("english_view_source_hash", "") or ""),
            source_hash,
        )
    logger.info(
        "[EN_VIEW_REBUILD_REQUIRED] kind=summary transcript_id=%s source_hash=%s cache_miss=%s",
        getattr(transcript, "id", ""),
        source_hash,
        not bool(cached_entry),
    )
    policy = evaluate_english_view_policy(
        content_kind="summary",
        source_language=source_language,
        source_state=transcript_state,
        has_grounded_text=_structured_summary_has_content(payload),
        low_evidence_malayalam=bool(source_language == "ml" and transcript_state == "degraded" and (low_evidence_malayalam or source_language_fidelity_failed)),
        degraded_low_evidence_reason="source_language_fidelity_failed" if source_language_fidelity_failed else "low_evidence_source_language",
    )
    if source_language_fidelity_failed:
        policy = {
            **policy,
            "allow_translation": False,
            "blocked_reason": "source_language_fidelity_failed",
            "current_available_views": ["original"],
        }
    logger.info(
        "[SUMMARY_EN_VIEW_REQUEST] transcript_id=%s language=%s state=%s allow=%s blocked_reason=%s",
        getattr(transcript, "id", ""),
        source_language,
        transcript_state,
        policy["allow_translation"],
        policy["blocked_reason"],
    )
    meta = build_safe_english_view_structured_summary(
        payload,
        source_language,
        allow_translation=bool(policy["allow_translation"]),
        blocked_reason=str(policy["blocked_reason"] or ""),
        warning=str(source_info["warning"] or ""),
    )
    if meta.get("summary_english_view_available"):
        logger.info(
            "[SUMMARY_EN_VIEW_RESULT] transcript_id=%s state=%s mode=%s",
            getattr(transcript, "id", ""),
            transcript_state,
            meta.get("summary_translation_state", ""),
        )
    else:
        logger.info(
            "[SUMMARY_EN_VIEW_BLOCKED] transcript_id=%s state=%s reason=%s",
            getattr(transcript, "id", ""),
            transcript_state,
            meta.get("summary_translation_blocked_reason", ""),
        )
    enriched = dict(payload)
    enriched.update(meta)
    cache_entry = build_english_view_cache_entry(
        meta,
        source_hash=source_hash,
        build_reason="structured_summary_serializer",
        source_language=source_language,
        policy=policy,
    )
    logger.info(
        "[EN_VIEW_PERSIST] kind=summary transcript_id=%s source_hash=%s build_reason=%s",
        getattr(transcript, "id", ""),
        source_hash,
        "structured_summary_serializer",
    )
    _persist_structured_summary_english_view_cache(transcript, cache_entry, source_hash=source_hash)
    transcript._summary_english_view_payload_cache = {"_source_hash": source_hash, "payload": enriched}
    return enriched


def _load_summary_content(summary):
    raw = getattr(summary, "content", None)
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"content": raw}
        except Exception:
            return {"content": raw}
    return {"content": str(raw)}


def _split_legacy_summary_lines(value):
    return [
        re.sub(r"^[\u2022*\-]+\s*", "", line.strip())
        for line in str(value or "").splitlines()
        if line and line.strip()
    ]


def _timestamp_from_seconds(total_seconds):
    seconds = max(0, int(float(total_seconds or 0.0)))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _clean_legacy_chapter_title(value, index):
    title = re.sub(r"\s+", " ", str(value or "")).strip(" -:|")
    if not title:
        return f"Chapter {index + 1}"
    if len(title) > 96:
        return f"{title[:93].rstrip()}..."
    return title


def _build_chapters_from_highlights(video):
    chapters = []
    for index, highlight in enumerate(video.highlight_segments.order_by("start_time")[:8]):
        label = str(getattr(highlight, "reason", "") or "").strip()
        if not label:
            snippet = str(getattr(highlight, "transcript_snippet", "") or "").strip()
            label = snippet.split(".")[0].strip()
        chapters.append({
            "timestamp": _timestamp_from_seconds(getattr(highlight, "start_time", 0.0)),
            "title": _clean_legacy_chapter_title(label, index),
        })
    return chapters


def _extract_legacy_action_items(*texts):
    action_items = []
    seen = set()
    cues = (
        "should ",
        "need to ",
        "try to ",
        "remember to ",
        "make sure ",
        "focus on ",
        "start ",
        "keep ",
        "avoid ",
        "prepare ",
        "review ",
        "practice ",
        "use ",
        "build ",
        "create ",
    )
    for raw in texts:
        for line in _split_legacy_summary_lines(raw):
            item = re.sub(r"\s+", " ", line).strip()
            lower = item.lower()
            if not item or len(item) < 12:
                continue
            if not any(cue in lower or lower.startswith(cue.strip()) for cue in cues):
                continue
            normalized = item.rstrip(".")
            dedupe_key = normalized.lower()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            action_items.append(normalized)
            if len(action_items) >= 4:
                return action_items
    return action_items


def _build_structured_summary_from_legacy_summaries(video):
    summaries_qs = video.summaries.all()
    short = summaries_qs.filter(summary_type='short').order_by('-created_at').first()
    bullet = summaries_qs.filter(summary_type='bullet').order_by('-created_at').first()
    timestamps = summaries_qs.filter(summary_type='timestamps').order_by('-created_at').first()

    short_content = _load_summary_content(short)
    bullet_content = _load_summary_content(bullet)
    timestamp_content = _load_summary_content(timestamps)

    tldr = str(
        short_content.get("content")
        or short_content.get("summary_text")
        or getattr(short, "content", "")
        or ""
    ).strip()
    key_points = bullet_content.get("key_points") or bullet_content.get("key_topics")
    if not isinstance(key_points, list):
        key_points = _split_legacy_summary_lines(
            bullet_content.get("content")
            or bullet_content.get("summary_text")
            or getattr(bullet, "content", "")
        )

    chapters = timestamp_content.get("chapters")
    if not isinstance(chapters, list):
        chapters = []
        for index, line in enumerate(_split_legacy_summary_lines(
            timestamp_content.get("content")
            or timestamp_content.get("summary_text")
            or getattr(timestamps, "content", "")
        )):
            match = re.match(r"(\d{1,2}:\d{2}(?::\d{2})?)\s*[-—:|]?\s*(.*)", line)
            chapters.append({
                "timestamp": (match.group(1) if match else "00:00"),
                "title": ((match.group(2) if match else line) or f"Chapter {index + 1}").strip(),
            })
    if not chapters:
        chapters = _build_chapters_from_highlights(video)

    action_items = _extract_legacy_action_items(
        bullet_content.get("content")
        or bullet_content.get("summary_text")
        or getattr(bullet, "content", ""),
        short_content.get("content")
        or short_content.get("summary_text")
        or getattr(short, "content", ""),
    )

    payload = {
        "tldr": tldr,
        "key_points": [str(item).strip() for item in (key_points or []) if str(item).strip()],
        "action_items": action_items,
        "chapters": [
            {
                "timestamp": str((item or {}).get("timestamp", "00:00") or "00:00").strip() or "00:00",
                "title": str((item or {}).get("title", "") or "").strip(),
            }
            for item in (chapters or [])
            if isinstance(item, dict) and str((item or {}).get("title", "") or "").strip()
        ],
    }
    return payload if _structured_summary_has_content(payload) else default_structured_summary()


def _transcript_cache_hash(transcript, segments):
    payload = "||".join([
        str(getattr(transcript, "id", "") or ""),
        str(getattr(transcript, "transcript_language", "") or ""),
        str(transcript.transcript_original_text or transcript.full_text or ""),
        "|".join(
            f"{float(seg.get('start', 0.0) or 0.0):.2f}:{float(seg.get('end', 0.0) or 0.0):.2f}:{str(seg.get('text', '') or '').strip()[:120]}"
            for seg in (segments or [])
            if isinstance(seg, dict)
        ),
    ])
    return sha1(payload.encode("utf-8")).hexdigest()


def _extract_structured_summary_inputs(video, transcript):
    summaries_qs = video.summaries.all()
    full = summaries_qs.filter(summary_type='full').order_by('-created_at').first()
    bullet = summaries_qs.filter(summary_type='bullet').order_by('-created_at').first()
    short = summaries_qs.filter(summary_type='short').order_by('-created_at').first()

    segs = []
    assembled_units = []
    internal_evidence_units = []
    json_data = transcript.json_data
    if isinstance(json_data, dict):
        raw = json_data.get('segments', [])
        if isinstance(raw, list):
            segs = raw
        raw_assembled = json_data.get('assembled_transcript_units', [])
        if isinstance(raw_assembled, list):
            assembled_units = raw_assembled
        raw_internal = json_data.get('internal_evidence_units', [])
        if isinstance(raw_internal, list):
            internal_evidence_units = raw_internal
    elif isinstance(json_data, list):
        segs = json_data

    transcript_language = str(getattr(transcript, "transcript_language", "") or getattr(transcript, "language", "") or '')
    transcript_state = str((json_data or {}).get('transcript_state', '') or '')
    assembled_units_for_summary = assembled_units
    evidence_units = assembled_units or segs
    if transcript_language == "ml" and str(transcript_state or "").strip().lower() == "degraded" and internal_evidence_units:
        evidence_units = assembled_units or internal_evidence_units or segs
        assembled_units_for_summary = assembled_units or internal_evidence_units
    transcript_text = (json_data.get('readable_transcript', '') if isinstance(json_data, dict) else '') or transcript.transcript_original_text or transcript.full_text or ''
    if transcript_language == "ml":
        preferred_text_units = assembled_units or evidence_units
        evidence_text = " ".join(
            str(seg.get("text", "")).strip()
            for seg in (preferred_text_units or [])
            if isinstance(seg, dict) and str(seg.get("text", "")).strip()
        ).strip()
        if evidence_text:
            transcript_text = evidence_text

    return {
        "video_id": str(video.id),
        "transcript_hash": _transcript_cache_hash(transcript, evidence_units),
        "transcript_text": transcript_text,
        "segments": evidence_units,
        "raw_segments": segs,
        "assembled_units": assembled_units_for_summary,
        "internal_evidence_units": internal_evidence_units,
        "transcript_state": transcript_state,
        "transcript_language": transcript_language,
        "full_summary": (full.content if full else ''),
        "bullet_summary": (bullet.content if bullet else ''),
        "short_summary": (short.content if short else ''),
    }


def get_or_build_structured_summary(video, transcript):
    if not transcript:
        return _build_structured_summary_from_legacy_summaries(video)

    if bool(getattr(settings, 'RENDER_TRANSCRIPT_ONLY_MODE', False)):
        json_data = transcript.json_data if isinstance(transcript.json_data, dict) else {}
        cached = json_data.get('structured_summary_cache', {})
        if isinstance(cached, dict) and isinstance(cached.get('payload'), dict):
            payload = cached.get('payload')
            if _structured_summary_has_content(payload):
                logger.info(
                    "[STRUCTURED_SUMMARY_CACHE] render_transcript_only_mode=cache_only video_id=%s transcript_id=%s cache_hit=True",
                    getattr(video, "id", ""),
                    getattr(transcript, "id", ""),
                )
                return _augment_structured_summary_with_english_view(payload, transcript)
        logger.info(
            "[STRUCTURED_SUMMARY_CACHE] render_transcript_only_mode=skip_rebuild video_id=%s transcript_id=%s",
            getattr(video, "id", ""),
            getattr(transcript, "id", ""),
        )
        return {
            **default_structured_summary(),
            "summary_state": "disabled_live_demo",
            "summary_blocked_reason": "render_transcript_only_mode",
        }

    json_data = transcript.json_data if isinstance(transcript.json_data, dict) else {}
    transcript_state = str(json_data.get('transcript_state', '') or '').strip().lower()
    transcript_language = str(getattr(transcript, "transcript_language", "") or getattr(transcript, "language", "") or "").strip().lower()
    if _is_pending_malayalam_final_state(transcript_language, transcript_state):
        logger.info(
            "[ML_SUMMARY_DEFERRED_PENDING_FINAL_STATE] video_id=%s transcript_id=%s transcript_state=%s",
            getattr(video, "id", ""),
            getattr(transcript, "id", ""),
            transcript_state or "pending",
        )
        return default_structured_summary()
    if (
        transcript_language == "ml"
        and (
            bool(json_data.get("source_language_fidelity_failed", False))
            or transcript_state == "source_language_fidelity_failed"
            or str(json_data.get("final_malayalam_fidelity_decision", "") or "").strip().lower() == "source_language_fidelity_failed"
            or bool(json_data.get("catastrophic_latin_substitution_failure", False))
        )
        and int(json_data.get("trusted_display_unit_count", 0) or 0) == 0
        and int(json_data.get("trusted_visible_word_count", 0) or 0) == 0
        and float((json_data.get("quality_metrics", {}) or {}).get("malayalam_token_coverage", 0.0) or 0.0) <= 0.08
    ):
        logger.info(
            "[ML_SUMMARY_SKIPPED_FIDELITY_FAILED_EARLY] video_id=%s transcript_id=%s",
            getattr(video, "id", ""),
            getattr(transcript, "id", ""),
        )
        payload = _fidelity_failed_malayalam_summary_payload(transcript)
        existing_structured_cache = json_data.get("structured_summary_cache", {}) if isinstance(json_data.get("structured_summary_cache", {}), dict) else {}
        existing_summary_reason = str(
            (((existing_structured_cache.get("english_view_cache", {}) or {}).get("payload", {}) or {}).get("summary_translation_blocked_reason", "") or "")
        )
        if existing_structured_cache or existing_summary_reason != "source_language_fidelity_failed":
            updated = dict(json_data)
            structured_cache = updated.get("structured_summary_cache", {})
            if not isinstance(structured_cache, dict):
                structured_cache = {}
            structured_cache["english_view_cache"] = {
                "english_view_valid": False,
                "payload": {
                    "summary_english_view_available": False,
                    "summary_translation_state": "blocked",
                    "summary_translation_blocked_reason": "source_language_fidelity_failed",
                    "summary_current_available_views": ["original"],
                },
            }
            updated["structured_summary_cache"] = structured_cache
            Transcript.objects.filter(pk=transcript.pk).update(json_data=updated)
            transcript.json_data = updated
        return payload
    if transcript_state == 'failed':
        logger.info(
            "[STRUCTURED_SUMMARY_CACHE] skipped_failed_transcript=True video_id=%s transcript_id=%s",
            getattr(video, "id", ""),
            getattr(transcript, "id", ""),
        )
        fallback = _build_structured_summary_from_legacy_summaries(video)
        return fallback if _structured_summary_has_content(fallback) else default_structured_summary()
    if transcript_language == "ml" and (
        bool(json_data.get("source_language_fidelity_failed", False))
        or transcript_state == "source_language_fidelity_failed"
        or str(json_data.get("final_malayalam_fidelity_decision", "") or "").strip().lower() == "source_language_fidelity_failed"
        or bool(json_data.get("catastrophic_latin_substitution_failure", False))
    ):
        logger.info(
            "[STRUCTURED_SUMMARY_CACHE] skipped_fidelity_failed_malayalam=True video_id=%s transcript_id=%s",
            getattr(video, "id", ""),
            getattr(transcript, "id", ""),
        )
        payload = _fidelity_failed_malayalam_summary_payload(transcript)
        existing_structured_cache = json_data.get("structured_summary_cache", {}) if isinstance(json_data.get("structured_summary_cache", {}), dict) else {}
        existing_summary_reason = str(
            (((existing_structured_cache.get("english_view_cache", {}) or {}).get("payload", {}) or {}).get("summary_translation_blocked_reason", "") or "")
        )
        if existing_structured_cache or existing_summary_reason != "source_language_fidelity_failed":
            updated = dict(json_data)
            structured_cache = updated.get("structured_summary_cache", {})
            if not isinstance(structured_cache, dict):
                structured_cache = {}
            structured_cache["english_view_cache"] = {
                "english_view_valid": False,
                "payload": {
                    "summary_english_view_available": False,
                    "summary_translation_state": "blocked",
                    "summary_translation_blocked_reason": "source_language_fidelity_failed",
                    "summary_current_available_views": ["original"],
                },
            }
            updated["structured_summary_cache"] = structured_cache
            Transcript.objects.filter(pk=transcript.pk).update(json_data=updated)
            transcript.json_data = updated
        return payload

    inputs = _extract_structured_summary_inputs(video, transcript)
    if (
        str(inputs.get("transcript_state", "") or "").strip().lower() == "degraded"
        and str(inputs.get("transcript_language", "") or "").strip().lower() == "ml"
    ):
        degraded_fallback = _degraded_safe_malayalam_payload(transcript)
    else:
        degraded_fallback = None
    cache_key = structured_summary_cache_key(**inputs)
    build_inputs = {
        "transcript_text": inputs["transcript_text"],
        "segments": inputs["segments"],
        "raw_segments": inputs["raw_segments"],
        "assembled_units": inputs["assembled_units"],
        "transcript_state": inputs["transcript_state"],
        "transcript_language": inputs["transcript_language"],
        "full_summary": inputs["full_summary"],
        "bullet_summary": inputs["bullet_summary"],
        "short_summary": inputs["short_summary"],
    }
    cached = json_data.get('structured_summary_cache', {})
    if isinstance(cached, dict) and cached.get('cache_key') == cache_key and isinstance(cached.get('payload'), dict):
        payload = cached.get('payload')
        if _structured_summary_has_content(payload):
            logger.info(
                "[STRUCTURED_SUMMARY_CACHE] served_from_cache=True video_id=%s cache_key=%s transcript_hash=%s",
                inputs["video_id"],
                cache_key,
                inputs["transcript_hash"],
            )
            return _augment_structured_summary_with_english_view(payload, transcript)
        if degraded_fallback:
            logger.info(
                "[STRUCTURED_SUMMARY_DEGRADED_SAFE_FALLBACK] source=cached_empty video_id=%s cache_key=%s transcript_hash=%s",
                inputs["video_id"],
                cache_key,
                inputs["transcript_hash"],
            )
            return _augment_structured_summary_with_english_view(degraded_fallback, transcript)
        if _should_allow_legacy_summary_fallback(inputs):
            legacy_fallback = _build_structured_summary_from_legacy_summaries(video)
            logger.info(
                "[STRUCTURED_SUMMARY_LEGACY_FALLBACK] source=cached_empty video_id=%s cache_key=%s transcript_hash=%s has_legacy=%s",
                inputs["video_id"],
                cache_key,
                inputs["transcript_hash"],
                _structured_summary_has_content(legacy_fallback),
            )
            return _augment_structured_summary_with_english_view(legacy_fallback, transcript)
        return _augment_structured_summary_with_english_view(payload, transcript)

    payload = build_structured_summary(**build_inputs)
    if not _structured_summary_has_content(payload):
        if degraded_fallback:
            logger.info(
                "[STRUCTURED_SUMMARY_DEGRADED_SAFE_FALLBACK] source=built_empty video_id=%s cache_key=%s transcript_hash=%s",
                inputs["video_id"],
                cache_key,
                inputs["transcript_hash"],
            )
            payload = degraded_fallback
        if _should_allow_legacy_summary_fallback(inputs):
            legacy_fallback = _build_structured_summary_from_legacy_summaries(video)
            if _structured_summary_has_content(legacy_fallback):
                logger.info(
                    "[STRUCTURED_SUMMARY_LEGACY_FALLBACK] source=built_empty video_id=%s cache_key=%s transcript_hash=%s",
                    inputs["video_id"],
                    cache_key,
                    inputs["transcript_hash"],
                )
                payload = legacy_fallback
    quality_metrics = evaluate_summary_quality(payload, inputs.get("transcript_text", ""))
    updated = dict(json_data)
    updated['structured_summary_cache'] = {
        "cache_key": cache_key,
        "video_id": inputs["video_id"],
        "transcript_hash": inputs["transcript_hash"],
        "payload": payload,
        "quality_metrics": quality_metrics,
        "quality_score": float(quality_metrics.get("final_quality_score", 0.0) or 0.0),
    }
    Transcript.objects.filter(pk=transcript.pk).update(json_data=updated)
    transcript.json_data = updated
    trace = payload.get("_trace") if isinstance(payload, dict) else {}
    logger.info(
        "[STRUCTURED_SUMMARY_CACHE] served_from_cache=False regenerated=True cache_miss=True video_id=%s cache_key=%s transcript_hash=%s",
        inputs["video_id"],
        cache_key,
        inputs["transcript_hash"],
    )
    logger.info(
        "[ML_STRUCTURED_REBUILD_RESULT] transcript_state=%s route=%s input_source=%s blocked_reason=%s has_content=%s",
        inputs.get("transcript_state", ""),
        str((trace or {}).get("structured_summary_route", "") or ""),
        str((trace or {}).get("structured_input_source", "") or ""),
        str((trace or {}).get("structured_summary_blocked_reason", "") or ""),
        _structured_summary_has_content(payload),
    )
    return _augment_structured_summary_with_english_view(payload, transcript)


class VideoUploadSerializer(serializers.Serializer):
    """Serializer for video upload."""
    title = serializers.CharField(max_length=255)
    description = serializers.CharField(required=False, allow_blank=True, default='')
    file = serializers.FileField()
    transcription_language = serializers.CharField(required=False, allow_blank=True, default='auto', max_length=16)
    output_language = serializers.CharField(required=False, allow_blank=True, default='auto', max_length=16)
    
    def validate_file(self, value):
        """Validate uploaded file is a video."""
        allowed_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv']
        ext = os.path.splitext(value.name)[1].lower()
        if ext not in allowed_extensions:
            raise serializers.ValidationError(
                f'Invalid file type. Allowed types: {", ".join(allowed_extensions)}'
            )
        
        # Validate file size (max 500MB)
        max_size = 500 * 1024 * 1024  # 500MB
        if value.size > max_size:
            raise serializers.ValidationError(
                f'File too large. Maximum size is 500MB. Your file is {value.size / (1024*1024):.1f}MB'
            )
        
        return value


class VideoSerializer(serializers.ModelSerializer):
    """Serializer for Video model."""
    
    filename = serializers.CharField(read_only=True)
    transcripts_count = serializers.SerializerMethodField()
    summaries_count = serializers.SerializerMethodField()
    shorts_count = serializers.SerializerMethodField()
    
    class Meta:
        model = Video
        fields = [
            'id', 'title', 'description', 'original_file', 'youtube_url', 'filename',
            'duration', 'file_size', 'file_format', 'status',
            'processing_progress', 'error_message', 'created_at',
            'updated_at', 'processed_at', 'transcripts_count',
            'summaries_count', 'shorts_count'
        ]
        read_only_fields = ['id', 'filename', 'duration', 'file_size', 
                           'file_format', 'status', 'processing_progress',
                           'created_at', 'updated_at', 'processed_at']
    
    def get_transcripts_count(self, obj):
        return obj.transcripts.count()
    
    def get_summaries_count(self, obj):
        return obj.summaries.count()
    
    def get_shorts_count(self, obj):
        return obj.short_videos.count()


class TranscriptSerializer(serializers.ModelSerializer):
    """Serializer for Transcript model."""
    
    word_count = serializers.SerializerMethodField()
    transcript_state = serializers.SerializerMethodField()
    readable_transcript = serializers.SerializerMethodField()
    captions = serializers.SerializerMethodField()
    original_language = serializers.SerializerMethodField()
    english_view_available = serializers.SerializerMethodField()
    english_view_text = serializers.SerializerMethodField()
    current_available_views = serializers.SerializerMethodField()
    translation_state = serializers.SerializerMethodField()
    translation_warning = serializers.SerializerMethodField()
    translation_blocked_reason = serializers.SerializerMethodField()
    transcript_quality = serializers.SerializerMethodField()
    
    class Meta:
        model = Transcript
        fields = [
            'id', 'video', 'language', 'transcript_language', 'canonical_language',
            'script_type', 'asr_engine', 'asr_engine_used', 'detection_confidence', 'transcript_quality_score',
            'full_text', 'transcript_original_text', 'transcript_canonical_text', 'transcript_canonical_en_text', 'json_data',
            'word_timestamps', 'word_count', 'transcript_state', 'readable_transcript', 'captions',
            'original_language', 'english_view_available', 'english_view_text', 'current_available_views', 'translation_state',
            'translation_warning', 'translation_blocked_reason', 'transcript_quality', 'created_at'
        ]
        read_only_fields = ['id', 'video', 'created_at']
    
    def get_word_count(self, obj):
        return obj.get_word_count()

    def get_transcript_state(self, obj):
        if isinstance(obj.json_data, dict):
            if _is_stale_bad_malayalam_transcript_payload(obj.json_data, obj):
                return 'source_language_fidelity_failed'
            return obj.json_data.get('transcript_state', '')
        return ''

    def get_readable_transcript(self, obj):
        if isinstance(obj.json_data, dict):
            if bool(
                obj.json_data.get('source_language_fidelity_failed', False)
                or str(obj.json_data.get('transcript_state', '') or '').strip().lower() == 'source_language_fidelity_failed'
                or str(obj.json_data.get('final_malayalam_fidelity_decision', '') or '').strip().lower() == 'source_language_fidelity_failed'
                or bool(obj.json_data.get('catastrophic_latin_substitution_failure', False))
                or _is_stale_bad_malayalam_transcript_payload(obj.json_data, obj)
            ):
                return ''
            return obj.json_data.get('readable_transcript', obj.full_text)
        return obj.full_text

    def get_captions(self, obj):
        if isinstance(obj.json_data, dict):
            return obj.json_data.get('captions', {})
        return {}

    def get_original_language(self, obj):
        if isinstance(obj.json_data, dict):
            return obj.json_data.get('original_language') or obj.json_data.get('detected_language') or obj.transcript_language or obj.language
        return obj.transcript_language or obj.language

    def get_english_view_available(self, obj):
        return bool(_safe_transcript_english_view(obj).get("english_view_available", False))

    def get_english_view_text(self, obj):
        return str(_safe_transcript_english_view(obj).get("english_view_text", "") or "")

    def get_current_available_views(self, obj):
        return list(_safe_transcript_english_view(obj).get("current_available_views") or ["original"])

    def get_translation_state(self, obj):
        return str(_safe_transcript_english_view(obj).get("translation_state", "") or "")

    def get_translation_warning(self, obj):
        return str(_safe_transcript_english_view(obj).get("translation_warning", "") or "")

    def get_translation_blocked_reason(self, obj):
        return str(_safe_transcript_english_view(obj).get("translation_blocked_reason", "") or "")

    def get_transcript_quality(self, obj):
        json_data = obj.json_data if isinstance(obj.json_data, dict) else {}
        source_language = str(
            json_data.get("language")
            or getattr(obj, "transcript_language", "")
            or getattr(obj, "language", "")
            or ""
        ).strip().lower()
        hard_failed = bool(
            (
                source_language == "ml"
            ) and (
                bool(json_data.get("source_language_fidelity_failed", False))
                or str(json_data.get("transcript_state", "") or "").strip().lower() == "source_language_fidelity_failed"
                or str(json_data.get("final_malayalam_fidelity_decision", "") or "").strip().lower() == "source_language_fidelity_failed"
                or bool(json_data.get("catastrophic_latin_substitution_failure", False))
                or _is_stale_bad_malayalam_transcript_payload(json_data, obj)
            )
        )
        if hard_failed:
            return {
                "status": "blocked",
                "warning": (
                    "Malayalam transcript could not be recovered faithfully enough for safe use."
                ),
                "warning_ml": (
                    "ഈ മലയാളം ട്രാൻസ്ക്രിപ്റ്റ് സുരക്ഷിതമായി ഉപയോഗിക്കാൻ മതിയായ വിശ്വസനീയതയോടെ വീണ്ടെടുക്കാൻ കഴിഞ്ഞില്ല."
                ),
            }
        if bool(json_data.get("has_fidelity_gaps", False)):
            return {
                "status": "partial",
                "warning": (
                    "Parts of this Malayalam transcript could not be recovered faithfully. "
                    "Some sections of the video may be missing."
                ),
                "warning_ml": (
                    "ഈ മലയാളം ട്രാൻസ്ക്രിപ്റ്റിന്റെ ചില ഭാഗങ്ങൾ വിശ്വസനീയമായി വീണ്ടെടുക്കാൻ കഴിഞ്ഞില്ല."
                ),
            }
        return {"status": "complete"}

    def to_representation(self, instance):
        data = super().to_representation(instance)
        json_data = instance.json_data if isinstance(instance.json_data, dict) else {}
        hide_text = bool(
            (
                str(getattr(instance, "transcript_language", "") or getattr(instance, "language", "") or "").strip().lower() == "ml"
            ) and (
                bool(json_data.get("source_language_fidelity_failed", False))
                or str(json_data.get("transcript_state", "") or "").strip().lower() == "source_language_fidelity_failed"
                or str(json_data.get("final_malayalam_fidelity_decision", "") or "").strip().lower() == "source_language_fidelity_failed"
                or bool(json_data.get("catastrophic_latin_substitution_failure", False))
                or _is_stale_bad_malayalam_transcript_payload(json_data, instance)
            )
        )
        if hide_text:
            data["full_text"] = ""
            data["transcript_original_text"] = ""
            data["transcript_canonical_text"] = ""
            data["transcript_canonical_en_text"] = ""
            data["readable_transcript"] = ""
            if isinstance(data.get("json_data"), dict):
                scrubbed = dict(data["json_data"])
                scrubbed["readable_transcript"] = ""
                scrubbed["display_readable_transcript"] = ""
                scrubbed["evidence_readable_transcript"] = ""
                data["json_data"] = scrubbed
        return data


class SummarySerializer(serializers.ModelSerializer):
    """Serializer for Summary model."""
    
    class Meta:
        model = Summary
        fields = [
            'id', 'video', 'summary_type', 'title', 'content',
            'key_topics', 'summary_language', 'summary_source_language', 'translation_used',
            'model_used', 'generation_time', 'created_at'
        ]
        read_only_fields = ['id', 'video', 'model_used', 'generation_time', 'created_at']


class SummaryGenerateSerializer(serializers.Serializer):
    """Serializer for generating summary request."""
    
    summary_type = serializers.ChoiceField(
        choices=['full', 'bullet', 'short', 'timestamps'],
        default='full'
    )
    max_length = serializers.IntegerField(required=False, min_value=50, max_value=1000)
    min_length = serializers.IntegerField(required=False, min_value=20, max_value=500)
    output_language = serializers.CharField(required=False, allow_blank=True, default='auto', max_length=16)
    summary_language_mode = serializers.ChoiceField(
        choices=['same_as_transcript', 'force_output_language'],
        default='same_as_transcript',
        required=False
    )
    # English View: Request translation to English
    english_view = serializers.BooleanField(required=False, default=False)


class HighlightSegmentSerializer(serializers.ModelSerializer):
    """Serializer for HighlightSegment model."""
    
    duration = serializers.FloatField(read_only=True)
    
    class Meta:
        model = HighlightSegment
        fields = [
            'id', 'video', 'start_time', 'end_time', 'duration',
            'importance_score', 'reason', 'transcript_snippet',
            'used_in_short', 'created_at'
        ]
        read_only_fields = ['id', 'video', 'created_at']


class ShortVideoSerializer(serializers.ModelSerializer):
    """Serializer for ShortVideo model."""
    
    filename = serializers.CharField(read_only=True)
    
    class Meta:
        model = ShortVideo
        fields = [
            'id', 'video', 'file', 'filename', 'duration',
            'thumbnail', 'style', 'include_music', 'music_track',
            'caption_style', 'font_size', 'status', 'created_at'
        ]
        read_only_fields = ['id', 'video', 'created_at']


class ShortVideoGenerateSerializer(serializers.Serializer):
    """Serializer for generating short video request."""
    
    max_duration = serializers.FloatField(default=60.0, required=False, help_text='Maximum short video duration in seconds')
    style = serializers.CharField(default='default', max_length=100, required=False)
    include_music = serializers.BooleanField(default=False, required=False)
    caption_style = serializers.CharField(default='default', max_length=100, required=False)
    font_size = serializers.IntegerField(default=24, required=False)


class ProcessingTaskSerializer(serializers.ModelSerializer):
    """Serializer for ProcessingTask model."""
    
    class Meta:
        model = ProcessingTask
        fields = [
            'id', 'task_type', 'task_id', 'video', 'status', 'progress',
            'message', 'error', 'created_at', 'started_at', 'completed_at'
        ]
        read_only_fields = ['id', 'task_type', 'task_id', 'video']


class VideoDetailSerializer(serializers.ModelSerializer):
    """Detailed serializer for video with nested relationships."""
    
    transcripts = serializers.SerializerMethodField()
    summaries = SummarySerializer(many=True, read_only=True)
    highlight_segments = HighlightSegmentSerializer(many=True, read_only=True)
    short_videos = ShortVideoSerializer(many=True, read_only=True)
    tasks = ProcessingTaskSerializer(many=True, read_only=True)
    structured_summary = serializers.SerializerMethodField()
    processing_metadata = serializers.SerializerMethodField()
    
    class Meta:
        model = Video
        fields = [
            'id', 'title', 'description', 'original_file', 'youtube_url', 'filename',
            'duration', 'file_size', 'file_format', 'status',
            'processing_progress', 'error_message', 'created_at',
            'updated_at', 'processed_at', 'transcripts', 'summaries',
            'highlight_segments', 'short_videos', 'tasks', 'structured_summary',
            'processing_metadata'
        ]

    def get_transcripts(self, obj):
        transcript = obj.transcripts.order_by('-created_at', '-id').first()
        if not transcript:
            return []
        transcript_json = transcript.json_data if isinstance(transcript.json_data, dict) else {}
        if str(transcript_json.get("transcript_state", "") or "").strip().lower() == "failed":
            return []
        return [TranscriptSerializer(transcript, context=self.context).data]

    def get_structured_summary(self, obj):
        try:
            transcript = obj.transcripts.order_by('-created_at', '-id').first()
            if not transcript:
                return _build_structured_summary_from_legacy_summaries(obj)
            transcript_json = transcript.json_data if isinstance(transcript.json_data, dict) else {}
            if str(transcript_json.get("transcript_state", "") or "").strip().lower() == "failed":
                return default_structured_summary()
            if bool(
                transcript_json.get("source_language_fidelity_failed", False)
                or str(transcript_json.get("transcript_state", "") or "").strip().lower() == "source_language_fidelity_failed"
                or str(transcript_json.get("final_malayalam_fidelity_decision", "") or "").strip().lower() == "source_language_fidelity_failed"
                or bool(transcript_json.get("catastrophic_latin_substitution_failure", False))
                or _is_stale_bad_malayalam_transcript_payload(transcript_json, transcript)
            ):
                return _fidelity_failed_malayalam_summary_payload(transcript)
            payload = get_or_build_structured_summary(obj, transcript)
            if (
                str(transcript_json.get("transcript_state", "") or "").strip().lower() == "degraded"
                and str(getattr(transcript, "transcript_language", "") or getattr(transcript, "language", "") or "").strip().lower() == "ml"
                and isinstance(payload, dict)
                and not str(payload.get("summary_state", "") or "").strip()
                and not bool(
                    transcript_json.get("source_language_fidelity_failed", False)
                    or str(transcript_json.get("transcript_state", "") or "").strip().lower() == "source_language_fidelity_failed"
                    or str(transcript_json.get("final_malayalam_fidelity_decision", "") or "").strip().lower() == "source_language_fidelity_failed"
                    or bool(transcript_json.get("catastrophic_latin_substitution_failure", False))
                    or _is_stale_bad_malayalam_transcript_payload(transcript_json, transcript)
                )
            ):
                payload = _degraded_safe_malayalam_payload(transcript)
            return _augment_structured_summary_with_english_view(payload, transcript)
        except Exception:
            transcript = obj.transcripts.order_by('-created_at').first()
            transcript_json = transcript.json_data if isinstance(getattr(transcript, "json_data", None), dict) else {}
            if (
                transcript
                and str(transcript_json.get("transcript_state", "") or "").strip().lower() == "degraded"
                and str(getattr(transcript, "transcript_language", "") or getattr(transcript, "language", "") or "").strip().lower() == "ml"
            ):
                if bool(
                    transcript_json.get("source_language_fidelity_failed", False)
                    or str(transcript_json.get("transcript_state", "") or "").strip().lower() == "source_language_fidelity_failed"
                    or str(transcript_json.get("final_malayalam_fidelity_decision", "") or "").strip().lower() == "source_language_fidelity_failed"
                    or bool(transcript_json.get("catastrophic_latin_substitution_failure", False))
                    or _is_stale_bad_malayalam_transcript_payload(transcript_json, transcript)
                ):
                    return _fidelity_failed_malayalam_summary_payload(transcript)
                return _augment_structured_summary_with_english_view(_degraded_safe_malayalam_payload(transcript), transcript)
            fallback = _build_structured_summary_from_legacy_summaries(obj)
            return _augment_structured_summary_with_english_view(fallback, transcript) if transcript else fallback

    def get_processing_metadata(self, obj):
        try:
            transcript = obj.transcripts.order_by('-created_at').first()
            metadata = build_processing_metadata(obj, transcript)
            if transcript:
                en_view = _safe_transcript_english_view(transcript)
                metadata.update({
                    "english_view_available": bool(en_view.get("english_view_available", False)),
                    "current_available_views": list(en_view.get("current_available_views") or ["original"]),
                    "translation_state": str(en_view.get("translation_state", "") or ""),
                    "translation_blocked_reason": str(en_view.get("translation_blocked_reason", "") or ""),
                })
            return metadata
        except Exception:
            return {
                "asr_engine": "",
                "language": "",
                "processing_time_seconds": 0.0,
                "transcript_quality_score": 0.0,
                "transcript_state": "",
                "summary_ready": False,
                "chat_ready": False,
            }
