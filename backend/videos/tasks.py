"""
Celery tasks for video processing
"""

import os
import logging
import shutil
import subprocess
import re
import tempfile
import uuid
from pathlib import Path
from celery import shared_task
from django.conf import settings
from django.db import transaction, IntegrityError
from django.utils import timezone
from django.core.files.storage import default_storage

from .models import Video, Transcript, Summary, HighlightSegment, ShortVideo, ProcessingTask
from .audio_preprocessor import chunk_on_silence_boundaries, condition_audio_for_asr, normalize_to_lufs
from .asr_router import transcribe_video_router, _deepgram_supported_languages
from .utils import (
    extract_audio, summarize_text,
    detect_highlights, create_short_video, get_video_duration,
    apply_entity_corrections, resolve_output_language, clean_transcript
)
from .canonical import build_canonical_text
from .language import detect_script_type, normalize_language_code
from .utils_metrics import evaluate_transcript_quality

logger = logging.getLogger(__name__)
transcribe_video = transcribe_video_router
_VIDEO_PROCESSING_STARTABLE_STATUSES = {'pending', 'uploaded', 'failed'}
_VIDEO_INTERRUPTED_STATUSES = {
    'processing',
    'extracting_audio',
    'transcribing',
    'cleaning_transcript',
    'transcript_ready',
    'summarizing_quick',
    'summarizing_final',
    'summarizing',
    'indexing_chat',
}


def _video_storage_backend_label(video) -> str:
    try:
        if video.original_file:
            return video.original_file.storage.__class__.__name__
    except Exception:
        pass
    return default_storage.__class__.__name__


def _video_original_exists(video) -> bool:
    if not getattr(video, 'original_file', None):
        return False
    try:
        exists = bool(video.original_file.storage.exists(video.original_file.name))
    except Exception as exc:
        logger.warning(
            "[VIDEO_SOURCE_CHECK] video_id=%s storage_backend=%s file_name=%s exists=unknown error=%s",
            getattr(video, 'id', ''),
            _video_storage_backend_label(video),
            str(getattr(video.original_file, 'name', '') or ''),
            exc,
        )
        return False

    logger.warning(
        "[VIDEO_SOURCE_CHECK] video_id=%s storage_backend=%s file_name=%s exists=%s",
        getattr(video, 'id', ''),
        _video_storage_backend_label(video),
        str(getattr(video.original_file, 'name', '') or ''),
        exists,
    )
    return exists


def _stage_uploaded_video_to_local_temp(video, *, purpose: str) -> tuple[str, str]:
    """
    Materialize an uploaded video to a local path for ffmpeg/moviepy processing.
    Local dev continues to use the filesystem path directly when available.
    Production object storage downloads to /tmp on demand.
    """
    if not getattr(video, 'original_file', None):
        raise FileNotFoundError("No uploaded video file is attached to this record.")

    storage_backend = _video_storage_backend_label(video)
    file_name = str(getattr(video.original_file, 'name', '') or '')

    if not _video_original_exists(video):
        raise FileNotFoundError(
            f"Uploaded video file is unavailable in storage for {purpose}. "
            f"backend={storage_backend} file={file_name}"
        )

    try:
        local_path = video.original_file.path
    except Exception:
        local_path = ''

    if local_path and os.path.exists(local_path):
        logger.warning(
            "[VIDEO_SOURCE_READY] video_id=%s purpose=%s storage_backend=%s mode=filesystem path=%s",
            getattr(video, 'id', ''),
            purpose,
            storage_backend,
            local_path,
        )
        return local_path, ''

    suffix = os.path.splitext(file_name)[1] or '.mp4'
    temp_dir = tempfile.mkdtemp(prefix=f'video_source_{purpose}_')
    local_copy_path = os.path.join(temp_dir, f'{uuid.uuid4()}{suffix}')
    logger.warning(
        "[VIDEO_SOURCE_DOWNLOAD_START] video_id=%s purpose=%s storage_backend=%s file_name=%s temp_path=%s",
        getattr(video, 'id', ''),
        purpose,
        storage_backend,
        file_name,
        local_copy_path,
    )
    bytes_written = 0
    with video.original_file.open('rb') as src, open(local_copy_path, 'wb') as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
            bytes_written += len(chunk)
    logger.warning(
        "[VIDEO_SOURCE_DOWNLOAD_DONE] video_id=%s purpose=%s bytes_written=%s temp_path=%s",
        getattr(video, 'id', ''),
        purpose,
        bytes_written,
        local_copy_path,
    )
    return local_copy_path, temp_dir


def _cleanup_local_staged_video(temp_dir: str, *, video_id: str, purpose: str) -> None:
    if not temp_dir:
        return
    shutil.rmtree(temp_dir, ignore_errors=True)
    logger.warning(
        "[VIDEO_SOURCE_CLEANUP_DONE] video_id=%s purpose=%s temp_dir=%s",
        video_id,
        purpose,
        temp_dir,
    )


def _sync_mode_max_video_seconds() -> float:
    try:
        if bool(getattr(settings, 'RENDER_DEMO_SAFE_ASR_MODE', False)):
            return max(0.0, float(getattr(settings, 'RENDER_DEMO_SAFE_MAX_VIDEO_SECONDS', 300) or 0.0))
        return max(0.0, float(getattr(settings, 'DEV_SYNC_MAX_VIDEO_SECONDS', 480) or 0.0))
    except Exception:
        return 480.0


def _sync_mode_lite_enabled() -> bool:
    return bool(
        getattr(settings, 'DEV_SYNC_MODE', False)
        and (
            getattr(settings, 'DEV_SYNC_LITE_MODE', False)
            or getattr(settings, 'RENDER_LIGHT_MODE', False)
        )
    )


def _ensure_sync_mode_duration_allowed(video, duration_seconds: float, *, source_type: str) -> None:
    """
    Keep free/sync deployments from accepting jobs that are too long for the web
    process to complete reliably.
    """
    if not getattr(settings, 'DEV_SYNC_MODE', False):
        return
    max_seconds = _sync_mode_max_video_seconds()
    if max_seconds <= 0:
        return

    actual_seconds = max(0.0, float(duration_seconds or 0.0))
    if actual_seconds <= max_seconds:
        return

    max_minutes = max_seconds / 60.0
    actual_minutes = actual_seconds / 60.0
    source_label = 'YouTube video' if source_type == 'youtube' else 'uploaded video'
    raise ValueError(
        f"{source_label} is too long for the current live server mode. "
        f"This deployment currently supports videos up to {max_minutes:.1f} minutes "
        f"for reliable processing, but this one is {actual_minutes:.1f} minutes."
    )


def _retain_malayalam_debug_audio(
    *,
    video_id,
    transcription_language: str,
    prep_meta: dict | None,
    result: dict | None,
) -> None:
    if not bool(getattr(settings, 'ASR_MALAYALAM_DEBUG_RETAIN_AUDIO', False)):
        return
    normalized_requested_language = normalize_language_code(
        transcription_language,
        default='auto',
        allow_auto=True,
    )
    if normalized_requested_language not in {'ml', 'auto'}:
        return
    if (result or {}).get('status') == 'success':
        return
    conditioned_path = str((prep_meta or {}).get('conditioned_path') or '').strip()
    if not conditioned_path or not os.path.exists(conditioned_path):
        return

    debug_dir = Path(str(getattr(settings, 'ASR_MALAYALAM_DEBUG_OUTPUT_DIR', 'backend/videos/debug_audio') or 'backend/videos/debug_audio'))
    debug_dir.mkdir(parents=True, exist_ok=True)
    retained_path = debug_dir / f"{video_id}_conditioned.wav"
    shutil.copy2(conditioned_path, retained_path)

    normalization = (prep_meta or {}).get('normalization')
    input_lufs = float(getattr(normalization, 'input_lufs', 0.0) or 0.0)
    chunks = list((prep_meta or {}).get('chunks', []) or [])
    logger.info(
        "[ML_DEBUG_RETAIN] video_id=%s retained_audio=%s input_lufs=%.2f chunks=%d",
        video_id,
        retained_path,
        input_lufs,
        len(chunks),
    )


def _prepare_audio_for_pipeline(
    audio_path: str,
    *,
    transcription_language: str = 'auto',
    apply_dynaudnorm: bool | None = None,
    apply_speech_band_filter: bool | None = None,
) -> tuple[str, dict]:
    temp_dir = tempfile.mkdtemp(prefix='audio_preprocess_')
    normalized_path = os.path.join(temp_dir, 'normalized.wav')
    conditioned_path = os.path.join(temp_dir, 'conditioned.wav')
    chunk_dir = os.path.join(temp_dir, 'chunks')
    
    normalized_requested_language = normalize_language_code(
        transcription_language,
        default='auto',
        allow_auto=True,
    )
    
    # OPTIMIZATION: English fast path - skip Malayalam-specific preprocessing
    # Groq Whisper is robust and doesn't need dynaudnorm/speech_band_filter
    use_english_fast_path = normalized_requested_language == 'en' and bool(
        getattr(settings, 'ASR_ENGLISH_USE_FAST_PATH', True)
    )
    
    if use_english_fast_path:
        # English: skip conditioning filters for faster processing
        # Just do LUFS normalization and pass through
        normalization = normalize_to_lufs(audio_path, normalized_path)
        # Return normalized (or original) path directly, skip conditioning
        final_path = normalization.output_path
        
        # For English, we don't need to create separate conditioned file
        # Just use the normalized path
        chunks = []  # Will chunk later if needed
        logger.info(
            "[ENGLISH_FAST_PATH] Skipping dynaudnorm/speech_band_filter for English - using Groq Whisper optimized path"
        )
    else:
        # Malayalam/other languages: use full preprocessing
        normalization = normalize_to_lufs(audio_path, normalized_path)
        conditioned = condition_audio_for_asr(
            normalization.output_path,
            conditioned_path,
            apply_dynaudnorm=(
                bool(getattr(settings, 'ASR_MALAYALAM_ENABLE_VOLUME_NORMALIZATION', True))
                if apply_dynaudnorm is None
                else bool(apply_dynaudnorm)
            ),
            apply_speech_band_filter=(
                bool(getattr(settings, 'ASR_MALAYALAM_ENABLE_SPEECH_BAND_FILTER', True))
                if apply_speech_band_filter is None
                else bool(apply_speech_band_filter)
            ),
        )
        final_path = conditioned.output_path
    
    chunk_max_duration_seconds = 30.0
    # Get Deepgram supported languages for optimization
    deepgram_supported = _deepgram_supported_languages()
    
    if normalized_requested_language == 'en':
        # English: use larger chunks for faster processing
        # Groq Whisper handles full audio files efficiently
        chunk_max_duration_seconds = float(
            getattr(settings, 'ASR_ENGLISH_CHUNK_MAX_DURATION_SECONDS', 120.0) or 120.0
        )
    elif normalized_requested_language == 'ml':
        # Malayalam: use smaller chunks for better accuracy
        chunk_max_duration_seconds = float(
            getattr(settings, 'ASR_MALAYALAM_CHUNK_MAX_DURATION_SECONDS', 18.0) or 18.0
        )
    elif normalized_requested_language == 'auto':
        # Auto-detect: use moderate chunks (60s default)
        # This balances between accuracy and speed
        chunk_max_duration_seconds = float(
            getattr(settings, 'ASR_AUTO_CHUNK_MAX_DURATION_SECONDS', 60.0) or 60.0
        )
    elif normalized_requested_language in deepgram_supported:
        # Deepgram-supported languages (hi, ta, te, kn, etc.): use larger chunks
        # Deepgram is optimized for longer audio and faster than chunked processing
        chunk_max_duration_seconds = float(
            getattr(settings, 'ASR_DEEPGRAM_CHUNK_MAX_DURATION_SECONDS', 120.0) or 120.0
        )
    
    # Chunk the audio (for English fast path, use the final_path after normalization)
    if use_english_fast_path:
        # For English, still need to chunk for ASR - but use normalized path
        chunks = chunk_on_silence_boundaries(
            final_path,
            chunk_dir,
            max_chunk_duration_s=chunk_max_duration_seconds,
        )
    else:
        chunks = chunk_on_silence_boundaries(
            final_path,
            chunk_dir,
            max_chunk_duration_s=chunk_max_duration_seconds,
        )
    
    metadata = {
        'temp_dir': temp_dir,
        'normalized_path': normalized_path,
        'conditioned_path': final_path if not use_english_fast_path else normalized_path,
        'normalization': normalization,
        'conditioning': None if use_english_fast_path else conditioned,
        'chunks': chunks,
        'chunk_max_duration_seconds': chunk_max_duration_seconds,
        'english_fast_path': use_english_fast_path,
    }
    
    # Log regardless of path
    if use_english_fast_path:
        logger.info(
            "[AUDIO_PREPROCESS_ENGLISH] english_fast_path=True input_lufs=%.2f output_lufs=%.2f reencoded=%s chunk_max_seconds=%.1f chunks=%d",
            float(normalization.input_lufs or 0.0),
            float(normalization.output_lufs or 0.0),
            bool(normalization.was_reencoded),
            float(chunk_max_duration_seconds or 0.0),
            len(chunks),
        )
    else:
        logger.info(
            "[AUDIO_PREPROCESS] input_lufs=%.2f output_lufs=%.2f reencoded=%s conditioned_noop=%s dynaudnorm=%s speech_band_filter=%s chunk_max_seconds=%.1f chunks=%s normalized_path=%s conditioned_path=%s",
            float(normalization.input_lufs or 0.0),
            float(normalization.output_lufs or 0.0),
            bool(normalization.was_reencoded),
            bool(conditioned.was_noop) if not use_english_fast_path else True,
            bool(conditioned.dynaudnorm_applied) if not use_english_fast_path else False,
            bool(conditioned.speech_band_filter_applied) if not use_english_fast_path else False,
            float(chunk_max_duration_seconds or 0.0),
            len(chunks),
            normalized_path,
            final_path,
        )
    return final_path, metadata


def _youtube_cookie_browsers() -> list[str]:
    configured = getattr(settings, 'YTDLP_COOKIES_FROM_BROWSERS', None)
    if isinstance(configured, str):
        return [item.strip() for item in configured.split(',') if item.strip()]
    if isinstance(configured, (list, tuple)):
        return [str(item).strip() for item in configured if str(item).strip()]
    return ['edge', 'chrome']


def _classify_youtube_download_failure(stderr: str) -> dict:
    message = str(stderr or '').strip()
    lowered = message.lower()
    rate_limited = 'http error 429' in lowered or 'too many requests' in lowered
    bot_blocked = 'sign in to confirm you’' in lowered or "sign in to confirm you're not a bot" in lowered or 'not a bot' in lowered
    js_runtime_missing = 'no supported javascript runtime could be found' in lowered
    cookies_needed = '--cookies-from-browser' in lowered or '--cookies' in lowered or 'exporting youtube cookies' in lowered
    browser_cookie_unavailable = (
        'could not find edge cookies database' in lowered
        or 'could not find chrome cookies database' in lowered
        or 'could not find cookies database' in lowered
        or 'failed to load cookies from browser' in lowered
        or 'could not copy edge cookie database' in lowered
        or 'could not copy chrome cookie database' in lowered
        or 'browser cookies are only supported on' in lowered
    )

    if rate_limited or bot_blocked:
        return {
            'blocked_reason': 'youtube_bot_protection',
            'retryable': False,
            'message': (
                'YouTube blocked automated download for this video. '
                'The app should retry with hardened yt-dlp settings and browser-cookie fallback, '
                'but this video still needs an authenticated browser-cookie path or a later retry.'
            ),
            'detail': message,
            'js_runtime_missing': js_runtime_missing,
            'cookies_needed': cookies_needed,
        }
    if browser_cookie_unavailable:
        return {
            'blocked_reason': 'youtube_browser_cookie_unavailable',
            'retryable': True,
            'message': 'Browser cookie fallback was unavailable on the current server.',
            'detail': message,
            'js_runtime_missing': js_runtime_missing,
            'cookies_needed': cookies_needed,
        }
    if js_runtime_missing:
        return {
            'blocked_reason': 'youtube_js_runtime_missing',
            'retryable': False,
            'message': 'yt-dlp could not use a supported JavaScript runtime for YouTube extraction.',
            'detail': message,
            'js_runtime_missing': True,
            'cookies_needed': cookies_needed,
        }
    return {
        'blocked_reason': 'youtube_download_failed',
        'retryable': True,
        'message': 'YouTube audio download failed.',
        'detail': message,
        'js_runtime_missing': js_runtime_missing,
        'cookies_needed': cookies_needed,
    }


def _build_ytdlp_command(video_url: str, audio_path: str, *, browser: str = '') -> list[str]:
    command = [
        'yt-dlp',
        '--no-playlist',
        '--extractor-args', 'youtube:player_client=android,web',
        '--extractor-retries', str(int(getattr(settings, 'YTDLP_EXTRACTOR_RETRIES', 2))),
        '--retries', str(int(getattr(settings, 'YTDLP_DOWNLOAD_RETRIES', 2))),
        '--fragment-retries', str(int(getattr(settings, 'YTDLP_FRAGMENT_RETRIES', 2))),
        '--sleep-requests', str(float(getattr(settings, 'YTDLP_SLEEP_REQUESTS_SECONDS', 1.0))),
        '-x',
        '--audio-format', 'wav',
        '--audio-quality', '0',
        '-o', audio_path,
    ]
    js_runtimes = str(getattr(settings, 'YTDLP_JS_RUNTIMES', '') or '').strip()
    if js_runtimes:
        command.extend(['--js-runtimes', js_runtimes])
    user_agent = str(getattr(settings, 'YTDLP_USER_AGENT', '') or '').strip()
    if user_agent:
        command.extend(['--user-agent', user_agent])
    if browser:
        command.extend(['--cookies-from-browser', browser])
    command.append(video_url)
    return command


def _download_youtube_audio(video_url: str, audio_path: str) -> None:
    attempts: list[tuple[str, list[str]]] = [('default', _build_ytdlp_command(video_url, audio_path))]
    for browser in _youtube_cookie_browsers():
        attempts.append((f'cookies:{browser}', _build_ytdlp_command(video_url, audio_path, browser=browser)))

    last_error = None
    preferred_error = None
    for attempt_name, command in attempts:
        logger.info("[YT_DOWNLOAD_ATTEMPT] mode=%s url=%s", attempt_name, video_url)
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            logger.info("[YT_DOWNLOAD_SUCCESS] mode=%s url=%s", attempt_name, video_url)
            return

        classification = _classify_youtube_download_failure(result.stderr)
        last_error = classification
        logger.error("yt-dlp failed: %s", result.stderr)
        logger.info(
            "[YT_DOWNLOAD_RETRY] mode=%s blocked_reason=%s retryable=%s js_runtime_missing=%s cookies_needed=%s",
            attempt_name,
            classification.get('blocked_reason', ''),
            bool(classification.get('retryable', False)),
            bool(classification.get('js_runtime_missing', False)),
            bool(classification.get('cookies_needed', False)),
        )
        blocked_reason = classification.get('blocked_reason')
        if blocked_reason == 'youtube_bot_protection':
            preferred_error = classification
            continue
        if blocked_reason == 'youtube_browser_cookie_unavailable' and attempt_name.startswith('cookies:'):
            continue
        if attempt_name == 'default' and classification.get('retryable', False):
            preferred_error = classification
            continue
        if preferred_error and blocked_reason == 'youtube_download_failed':
            last_error = preferred_error
            break
        break

    final_error = last_error or preferred_error
    if final_error:
        detail = str(final_error.get('detail', '') or '').strip()
        raise Exception(
            f"{final_error.get('message', 'YouTube audio download failed.')}"
            + (f" Details: {detail}" if detail else "")
        )
    raise Exception("YouTube audio download failed.")


def _video_still_exists(video_id) -> bool:
    return Video.objects.filter(id=video_id).exists()


def _normalized_segments(transcript_data):
    """Return transcript segments as a list of dicts."""
    if isinstance(transcript_data, dict):
        segments = transcript_data.get('segments', [])
        return segments if isinstance(segments, list) else []
    if isinstance(transcript_data, list):
        return transcript_data
    return []


def _summary_fields(summary_result, default_title):
    """Normalize summarizer outputs from different code paths."""
    content = summary_result.get('content') or summary_result.get('summary', '')
    return {
        'title': summary_result.get('title', default_title),
        'content': content,
        'key_topics': summary_result.get('key_topics', []),
        'summary_language': summary_result.get('summary_language', 'en'),
        'summary_source_language': summary_result.get('summary_source_language', 'en'),
        'translation_used': bool(summary_result.get('translation_used', False)),
        'model_used': summary_result.get('model_used') or summary_result.get('model', 'facebook/bart-large-cnn'),
        'generation_time': summary_result.get('generation_time', 0)
    }


def _format_srt_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0.0))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def _format_vtt_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0.0))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"


def _build_caption_artifacts(segments):
    normalized = []
    for seg in segments or []:
        if not isinstance(seg, dict):
            continue
        text = re.sub(r'\s+', ' ', str(seg.get('text', '')).strip())
        if not text:
            continue
        start = float(seg.get('start', 0.0) or 0.0)
        end = float(seg.get('end', start) or start)
        if end <= start:
            end = start + 0.8
        normalized.append((start, end, text))

    if not normalized:
        return {"srt": "", "vtt": "WEBVTT\n\n"}

    srt_lines = []
    vtt_lines = ["WEBVTT", ""]
    for idx, (start, end, text) in enumerate(normalized, start=1):
        srt_lines.extend([
            str(idx),
            f"{_format_srt_timestamp(start)} --> {_format_srt_timestamp(end)}",
            text,
            "",
        ])
        vtt_lines.extend([
            f"{_format_vtt_timestamp(start)} --> {_format_vtt_timestamp(end)}",
            text,
            "",
        ])

    return {
        "srt": "\n".join(srt_lines).strip(),
        "vtt": "\n".join(vtt_lines).strip(),
    }


def _quality_badge(score: float) -> str:
    value = float(score or 0.0)
    if value >= 0.78:
        return "high"
    if value >= 0.62:
        return "medium"
    if value >= 0.45:
        return "low"
    return "unreliable"


def _update_video_stage(video, status: str, progress: int, *, error_message: str | None = None, processed: bool = False):
    """Persist additive production stages without changing API shape."""
    video.status = status
    video.processing_progress = progress
    update_fields = ['status', 'processing_progress', 'updated_at']
    if error_message is not None:
        video.error_message = error_message
        update_fields.append('error_message')
    if processed:
        video.processed_at = timezone.now()
        update_fields.append('processed_at')
    video.save(update_fields=update_fields)


def _claim_video_processing(video_id, *, started_status: str = 'processing', started_progress: int = 5):
    claimed = Video.objects.filter(
        id=video_id,
        status__in=list(_VIDEO_PROCESSING_STARTABLE_STATUSES),
    ).update(
        status=started_status,
        processing_progress=started_progress,
        error_message='',
        updated_at=timezone.now(),
    )
    video = Video.objects.filter(id=video_id).first()
    return video, bool(claimed)


def _requeue_interrupted_video(video_id) -> tuple[Video | None, bool]:
    """
    Move a stranded in-progress video back to `uploaded` so sync recovery can retry it.
    Uses a conditional update so only one process can claim the recovery.
    """
    video = Video.objects.filter(id=video_id).first()
    if video is None or video.status not in _VIDEO_INTERRUPTED_STATUSES:
        return video, False
    updated = Video.objects.filter(
        id=video_id,
        status=video.status,
    ).update(
        status='uploaded',
        error_message='Recovered after interrupted in-app processing restart.',
        updated_at=timezone.now(),
    )
    video.refresh_from_db()
    return video, bool(updated)


def resume_interrupted_video_processing_sync(
    video_id,
    transcription_language: str = 'auto',
    output_language: str = 'auto',
    summary_language_mode: str = 'same_as_transcript'
):
    """
    Best-effort recovery path for DEV_SYNC_MODE on platforms where the web service
    can restart mid-generation. This is not a substitute for a dedicated worker,
    but it prevents videos from remaining stranded forever after a reboot.
    """
    video, claimed = _requeue_interrupted_video(video_id)
    if video is None:
        return {'status': 'error', 'message': 'Video not found', 'video_id': str(video_id)}
    if not claimed:
        return {'status': 'skipped', 'message': 'Video was already recovered or finished', 'video_id': str(video_id)}

    if video.youtube_url:
        logger.info("Recovering interrupted YouTube processing for %s", video_id)
        return process_youtube_video_sync(
            video_id,
            transcription_language=transcription_language,
            output_language=output_language,
            summary_language_mode=summary_language_mode,
        )

    if _video_original_exists(video):
        logger.info("Recovering interrupted uploaded-file processing for %s", video_id)
        return process_video_transcription_sync(
            video_id,
            transcription_language=transcription_language,
            output_language=output_language,
            summary_language_mode=summary_language_mode,
        )

    _fail_video_with_logged_status(
        video,
        Exception('Original media file is unavailable after service restart; recovery cannot continue.'),
        source='sync_recovery',
    )
    return {
        'status': 'error',
        'message': 'Original media file is unavailable after service restart; recovery cannot continue.',
        'video_id': str(video_id),
    }


def _draft_transcript_text(transcript_payload: dict) -> str:
    text = (transcript_payload or {}).get('draft_text') or (transcript_payload or {}).get('text') or ''
    if text:
        return str(text).strip()
    return " ".join(
        str(seg.get('text', '')).strip()
        for seg in _normalized_segments(transcript_payload)
        if isinstance(seg, dict) and str(seg.get('text', '')).strip()
    ).strip()


def _safe_debug_preview(text: str, limit: int = 180) -> str:
    compact = " ".join(str(text or "").split()).strip()
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit].rstrip()}..."


def _malayalam_script_ratio(text: str) -> float:
    from . import utils as videos_utils

    dist = videos_utils._script_distribution(text)  # pylint: disable=protected-access
    total = max(sum(dist.values()), 1)
    return float(dist.get('malayalam', 0) or 0) / float(total)


def _should_use_malayalam_repair_candidate(
    original_text: str,
    repaired_text: str,
    quality_before: dict,
    quality_after: dict,
) -> tuple[bool, str]:
    state_rank = {'failed': 0, 'low_confidence': 1, 'degraded': 2, 'cleaned': 3}
    before_state = str((quality_before or {}).get('state') or 'failed')
    after_state = str((quality_after or {}).get('state') or 'failed')
    if state_rank.get(after_state, -1) > state_rank.get(before_state, -1):
        return True, f"state_improved:{before_state}->{after_state}"

    before_qa = (quality_before or {}).get('qa_metrics', {}) or {}
    after_qa = (quality_after or {}).get('qa_metrics', {}) or {}
    before_garble = float(before_qa.get('garbled_detector_score', 1.0) or 1.0)
    after_garble = float(after_qa.get('garbled_detector_score', 1.0) or 1.0)
    before_ratio = _malayalam_script_ratio(original_text)
    after_ratio = _malayalam_script_ratio(repaired_text)
    from . import utils as videos_utils
    before_metrics = videos_utils._extract_asr_metrics(original_text, [])  # pylint: disable=protected-access
    after_metrics = videos_utils._extract_asr_metrics(repaired_text, [])  # pylint: disable=protected-access
    before_repeat = float(before_metrics.get('repeated_token_ratio', 1.0) or 1.0)
    after_repeat = float(after_metrics.get('repeated_token_ratio', 1.0) or 1.0)
    before_info = float(before_metrics.get('info_density', 0.0) or 0.0)
    after_info = float(after_metrics.get('info_density', 0.0) or 0.0)
    before_readability = videos_utils._score_malayalam_segment_readability(original_text)  # pylint: disable=protected-access
    after_readability = videos_utils._score_malayalam_segment_readability(repaired_text)  # pylint: disable=protected-access

    if after_state == before_state and after_garble + 0.05 < before_garble:
        return True, "garble_reduced"
    if after_state == before_state and after_ratio > before_ratio + 0.12:
        return True, "malayalam_script_ratio_improved"
    if after_state == before_state and float(after_readability.get('score', 0.0) or 0.0) > float(before_readability.get('score', 0.0) or 0.0) + 0.06:
        return True, "readability_improved"
    if after_state == before_state and after_repeat + 0.08 < before_repeat:
        return True, "repetition_reduced"
    if after_state == before_state and after_info > before_info + 0.08:
        return True, "information_density_improved"
    return False, "no_quality_gain"


def _resolve_low_trust_malayalam_outcome(
    *,
    structurally_usable: bool,
    meaningful_malayalam_evidence: bool,
    malayalam_truly_unusable: bool,
    garble_score: float,
    lexical_trust_score: float,
    overall_readability: float,
    trusted_malayalam_segments: int,
    corrupted_segments: int,
    decision_reason: str,
) -> tuple[str, str]:
    low_trust_reasons = {
        'semantic_trust_too_low',
        'pseudo_phonetic_ratio_too_high',
        'low_trust_malayalam_segment_share_too_high',
        'corrupted_segment_share_too_high',
        'malformed_malayalam_segments_too_high',
        'overall_readability_too_low',
        'dominant_script_not_malayalam',
        'malayalam_ratio_too_low',
        'token_coverage_too_low',
        'other_indic_ratio_too_high',
        'final_garble_above_cleaned_threshold',
    }
    if decision_reason not in low_trust_reasons:
        return '', 'not_low_trust_malayalam'

    if not meaningful_malayalam_evidence:
        return 'failed', 'no_surviving_malayalam_evidence'

    if structurally_usable and not malayalam_truly_unusable:
        return 'degraded', 'structurally_usable_low_trust_malayalam'

    if not structurally_usable or malayalam_truly_unusable:
        return 'failed', 'structurally_unusable_low_trust_malayalam'

    if (
        garble_score >= 0.85
        or (lexical_trust_score <= 0.05 and overall_readability <= 0.12)
        or (trusted_malayalam_segments <= 0 and corrupted_segments >= 3)
    ):
        return 'failed', 'unusable_low_trust_malayalam'

    return 'degraded', 'fallback_low_trust_malayalam'


def _is_degraded_low_trust_malayalam_quality(quality: dict, source_language: str) -> bool:
    if normalize_language_code(source_language, default='en', allow_auto=False) != 'ml':
        return False
    if bool((quality or {}).get('qa_metrics', {}).get('source_language_fidelity_failed', False)):
        return False
    if not bool((quality or {}).get('qa_metrics', {}).get('meaningful_malayalam_evidence', True)):
        return False
    if str((quality or {}).get('state') or '') != 'degraded':
        return False
    reason = str((quality or {}).get('malayalam_post_asr_reason') or (quality or {}).get('qa_metrics', {}).get('malayalam_post_asr_accept_reason') or '')
    return reason in {
        'semantic_trust_too_low',
        'pseudo_phonetic_ratio_too_high',
        'low_trust_malayalam_segment_share_too_high',
        'corrupted_segment_share_too_high',
        'malformed_malayalam_segments_too_high',
        'overall_readability_too_low',
        'dominant_script_not_malayalam',
        'malayalam_ratio_too_low',
        'token_coverage_too_low',
        'other_indic_ratio_too_high',
        'final_garble_above_cleaned_threshold',
        'structurally_usable_low_trust_malayalam',
        'fallback_low_trust_malayalam',
    }


def _is_low_trust_malayalam_exception(exc: Exception) -> bool:
    visited: set[int] = set()
    stack = [exc]
    while stack:
        current = stack.pop()
        if current is None:
            continue
        marker = id(current)
        if marker in visited:
            continue
        visited.add(marker)
        values = [str(current)]
        values.extend(str(item) for item in getattr(current, 'args', ()) if item is not None)
        if any(str(value).strip().strip("'").strip('"') == 'low_trust_malayalam' for value in values):
            return True
        cause = getattr(current, '__cause__', None)
        context = getattr(current, '__context__', None)
        if cause is not None:
            stack.append(cause)
        if context is not None:
            stack.append(context)
    return False


def _latest_resolved_low_trust_malayalam_transcript(video) -> Transcript | None:
    transcript_obj = Transcript.objects.filter(video=video).order_by('-id').first()
    if not transcript_obj:
        return None
    payload = transcript_obj.json_data if isinstance(transcript_obj.json_data, dict) else {}
    quality = {
        'state': payload.get('transcript_state', ''),
        'malayalam_post_asr_reason': payload.get('malayalam_post_asr_reason', ''),
        'qa_metrics': payload.get('quality_metrics', {}) if isinstance(payload.get('quality_metrics', {}), dict) else {},
    }
    source_language = payload.get('language') or transcript_obj.transcript_language or transcript_obj.language or ''
    if _is_degraded_low_trust_malayalam_quality(quality, source_language):
        return transcript_obj
    return None


def should_continue_degraded_low_trust_malayalam(
    *,
    video=None,
    exc: Exception | None = None,
    quality: dict | None = None,
    source_language: str = '',
    stage: str = '',
) -> dict:
    matched_low_trust = _is_low_trust_malayalam_exception(exc) if exc is not None else False
    transcript_obj = _latest_resolved_low_trust_malayalam_transcript(video) if video is not None else None
    persisted_payload = transcript_obj.json_data if transcript_obj and isinstance(transcript_obj.json_data, dict) else {}
    persisted_state = str(persisted_payload.get('transcript_state', '') or '')
    persisted_reason = str(persisted_payload.get('malayalam_post_asr_reason', '') or '')
    persisted_structurally_usable = bool(
        persisted_payload.get('quality_metrics', {}).get('structurally_usable', False)
    ) if isinstance(persisted_payload.get('quality_metrics', {}), dict) else False
    quality_match = _is_degraded_low_trust_malayalam_quality(quality or {}, source_language) if quality else False
    quality_metrics = (quality or {}).get('qa_metrics', {}) if isinstance((quality or {}).get('qa_metrics', {}), dict) else {}
    quality_state = str((quality or {}).get('state', '') or '')
    quality_reason = str((quality or {}).get('malayalam_post_asr_reason', '') or quality_metrics.get('malayalam_post_asr_accept_reason', '') or '')
    quality_structurally_usable = bool(quality_metrics.get('structurally_usable', False))
    transcript_state = persisted_state or quality_state
    decision_reason = persisted_reason or quality_reason
    structurally_usable = persisted_structurally_usable or quality_structurally_usable or decision_reason == 'structurally_usable_low_trust_malayalam'
    should_continue = bool(matched_low_trust and (transcript_obj or quality_match) and transcript_state == 'degraded')
    logger.info(
        "[ML_EXCEPTION_PATH] stage=%s exc_type=%s exc_repr=%r matched_low_trust=%s persisted_state=%s quality_state=%s structurally_usable=%s should_continue=%s",
        stage or 'unknown',
        type(exc).__name__ if exc is not None else 'None',
        str(exc) if exc is not None else '',
        matched_low_trust,
        persisted_state or 'none',
        quality_state or 'none',
        structurally_usable,
        should_continue,
    )
    return {
        'continue_degraded': should_continue,
        'matched_low_trust': matched_low_trust,
        'transcript_obj': transcript_obj,
        'transcript_state': transcript_state,
        'reason': decision_reason,
        'structurally_usable': structurally_usable,
    }


def _continue_degraded_low_trust_malayalam(video, exc: Exception, *, source: str, quality: dict | None = None, source_language: str = '') -> dict | None:
    decision = should_continue_degraded_low_trust_malayalam(
        video=video,
        exc=exc,
        quality=quality,
        source_language=source_language,
        stage=source,
    )
    transcript_obj = decision.get('transcript_obj')
    if not decision.get('continue_degraded'):
        if decision.get('matched_low_trust'):
            logger.warning(
                "[ML_LOW_TRUST_FATAL] source=%s transcript_state=%s structurally_usable=%s decision=remain_fatal reason=%s",
                source,
                decision.get('transcript_state', '') or 'unknown',
                decision.get('structurally_usable', False),
                decision.get('reason', '') or 'unknown',
            )
        return None
    payload = transcript_obj.json_data if transcript_obj and isinstance(transcript_obj.json_data, dict) else {}
    warning = str(payload.get('transcript_warning_message', '') or (quality or {}).get('transcript_warning_message', '') or '')
    reason = str(payload.get('malayalam_post_asr_reason', '') or decision.get('reason', '') or '')
    logger.warning(
        "[ML_LOW_TRUST_CONTINUE] source=%s transcript_state=%s structurally_usable=%s reason=%s exception=%s decision=continue_degraded",
        source,
        payload.get('transcript_state', '') or decision.get('transcript_state', ''),
        decision.get('structurally_usable', False),
        reason or 'unknown',
        str(exc),
    )
    if video is not None:
        _update_video_stage(video, 'completed', 100, processed=True, error_message='')
        logger.info(
            "[ML_FINAL_STATUS] transcript_state=%s video_status=completed continue_low_trust_degraded=True source=%s",
            payload.get('transcript_state', '') or decision.get('transcript_state', ''),
            source,
        )
    return {
        'status': 'success',
        'video_id': str(video.id),
        'transcript_id': transcript_obj.id if transcript_obj is not None else None,
        'warning': warning,
    }


def _fail_video_with_logged_status(video, exc: Exception, *, source: str):
    if _is_low_trust_malayalam_exception(exc):
        logger.warning(
            "[ML_LOW_TRUST_FATAL] source=%s transcript_state=unknown structurally_usable=False decision=fail exception=%s",
            source,
            str(exc),
        )
    _update_video_stage(video, 'failed', video.processing_progress or 0, error_message=str(exc))
    logger.info(
        "[ML_FINAL_STATUS] transcript_state=failed video_status=failed continue_low_trust_degraded=False source=%s",
        source,
    )


def _persist_minimal_low_trust_malayalam_checkpoint(
    transcript_obj,
    *,
    transcript_payload: dict,
    canonical_payload: dict,
    cleaned_text: str,
    cleaned_segments: list,
    processing_seconds: float,
    quality: dict,
) -> None:
    language = (
        (transcript_payload or {}).get('language')
        or transcript_obj.transcript_language
        or transcript_obj.language
        or ''
    )
    transcript_obj.full_text = cleaned_text
    transcript_obj.transcript_original_text = cleaned_text
    transcript_obj.transcript_canonical_text = canonical_payload.get('canonical_text', cleaned_text)
    transcript_obj.transcript_canonical_en_text = canonical_payload.get('canonical_text', cleaned_text)
    transcript_obj.canonical_language = canonical_payload.get('canonical_language', 'en')
    transcript_obj.transcript_quality_score = float((quality or {}).get('quality_score', 0.0) or 0.0)
    transcript_state = str((quality or {}).get('state') or 'degraded')
    visible_text = '' if transcript_state in {'degraded', 'failed', 'source_language_fidelity_failed'} else cleaned_text
    qa_metrics = (quality or {}).get('qa_metrics', {}) if isinstance((quality or {}).get('qa_metrics', {}), dict) else {}
    source_language_fidelity_failed = bool(
        qa_metrics.get('source_language_fidelity_failed', False)
        or transcript_state == 'source_language_fidelity_failed'
    )
    transcript_obj.json_data = {
        'segments': cleaned_segments,
        'raw_transcript_segments': cleaned_segments,
        'assembled_transcript_units': [],
        'display_transcript_units': [],
        'readable_transcript': visible_text,
        'display_readable_transcript': visible_text,
        'evidence_readable_transcript': cleaned_text,
        'language': language,
        'transcript_state': transcript_state,
        'transcript_warning_message': str((quality or {}).get('transcript_warning_message', '') or ''),
        'malayalam_post_asr_mode': str((quality or {}).get('malayalam_post_asr_mode', '') or ''),
        'malayalam_post_asr_reason': str((quality or {}).get('malayalam_post_asr_reason', '') or ''),
        'quality_metrics': qa_metrics,
        'transcript_warnings': list((quality or {}).get('warnings', []) or []),
        'processing_time_seconds': round(float(processing_seconds or 0.0), 4),
        'source_language_fidelity_failed': source_language_fidelity_failed,
        'transcript_fidelity_state': str(qa_metrics.get('transcript_fidelity_state', '') or ''),
        'final_malayalam_fidelity_decision': str(qa_metrics.get('final_malayalam_fidelity_decision', '') or ''),
        'catastrophic_latin_substitution_failure': bool(qa_metrics.get('catastrophic_latin_substitution_failure', False)),
        'summary_blocked_reason': str((quality or {}).get('summary_blocked_reason', '') or qa_metrics.get('summary_blocked_reason', '') or ''),
        'chatbot_blocked_reason': str((quality or {}).get('chatbot_blocked_reason', '') or qa_metrics.get('chatbot_blocked_reason', '') or ''),
        'transcript_display_mode': 'suppressed_unfaithful_source' if source_language_fidelity_failed else ('suppressed_low_evidence' if transcript_state == 'degraded' else 'visible'),
    }
    transcript_obj.word_timestamps = (transcript_payload or {}).get('word_timestamps', [])
    transcript_obj.save(update_fields=[
        'full_text',
        'transcript_original_text',
        'transcript_canonical_text',
        'transcript_canonical_en_text',
        'canonical_language',
        'transcript_quality_score',
        'json_data',
        'word_timestamps',
    ])


def _should_suppress_low_trust_malayalam_outputs(transcript_obj) -> tuple[bool, str]:
    gate = _evaluate_malayalam_low_evidence_downstream_gate(transcript_obj)
    return bool(gate.get('suppress', False)), str(gate.get('reason', '') or '')


def _evaluate_malayalam_low_evidence_downstream_gate(transcript_obj) -> dict:
    json_data = transcript_obj.json_data if isinstance(transcript_obj.json_data, dict) else {}
    transcript_state = str(json_data.get('transcript_state', '') or '').strip().lower()
    language = normalize_language_code(
        str(json_data.get('language') or transcript_obj.transcript_language or transcript_obj.language or ''),
        default='en',
        allow_auto=False,
    )
    default_gate = {
        'suppress': False,
        'reason': '',
        'trusted_visible_word_count': 0,
        'trusted_display_unit_count': 0,
        'low_evidence_malayalam': False,
        'overall_readability': 0.0,
        'lexical_trust_score': 0.0,
    }
    if language != 'ml' or transcript_state not in {'degraded', 'source_language_fidelity_failed'}:
        return default_gate
    if bool(json_data.get('source_language_fidelity_failed', False)):
        gate = {
            'suppress': True,
            'reason': 'malayalam_source_fidelity_failed',
            'trusted_visible_word_count': int(json_data.get('trusted_visible_word_count', 0) or 0),
            'trusted_display_unit_count': int(json_data.get('trusted_display_unit_count', 0) or 0),
            'low_evidence_malayalam': True,
            'overall_readability': round(float((json_data.get('quality_metrics', {}) or {}).get('overall_readability', 0.0) or 0.0), 4),
            'lexical_trust_score': round(float((json_data.get('quality_metrics', {}) or {}).get('lexical_trust_score', 0.0) or 0.0), 4),
        }
        logger.info(
            "[ML_DOWNSTREAM_GATE] transcript_state=%s suppress=%s reason=%s trusted_visible_word_count=%s trusted_display_unit_count=%s visible_word_count=%s overall_readability=%.3f lexical_trust=%.3f contamination_burden=%.3f",
            transcript_state,
            True,
            gate['reason'],
            gate['trusted_visible_word_count'],
            gate['trusted_display_unit_count'],
            0,
            gate['overall_readability'],
            gate['lexical_trust_score'],
            1.0,
        )
        return gate

    display_units = json_data.get('display_transcript_units', [])
    if not isinstance(display_units, list):
        display_units = []
    asr_metadata = json_data.get('asr_metadata', {}) if isinstance(json_data.get('asr_metadata', {}), dict) else {}
    display_meta = asr_metadata.get('malayalam_display_refinement', {}) if isinstance(asr_metadata.get('malayalam_display_refinement', {}), dict) else {}
    transcript_trust = asr_metadata.get('transcript_trust', {}) if isinstance(asr_metadata.get('transcript_trust', {}), dict) else {}
    quality_metrics = json_data.get('quality_metrics', {}) if isinstance(json_data.get('quality_metrics', {}), dict) else {}
    visible_word_count = int(display_meta.get('final_visible_word_count', 0) or 0)
    trusted_units = 0
    trusted_visible_word_count = 0
    for unit in display_units:
        if not isinstance(unit, dict):
            continue
        if (
            float(unit.get('unit_readability', 0.0) or 0.0) >= 0.34
            and float(unit.get('wrong_script_ratio', 0.0) or 0.0) <= 0.10
            and float(unit.get('contamination_score', 0.0) or 0.0) < 0.52
        ):
            trusted_units += 1
            trusted_visible_word_count += len(re.findall(r'\w+', str(unit.get('text') or ''), flags=re.UNICODE))

    overall_readability = float(
        transcript_trust.get('overall_readability', quality_metrics.get('overall_readability', 0.0)) or 0.0
    )
    lexical_trust = float(
        transcript_trust.get('lexical_trust_score', quality_metrics.get('lexical_trust_score', 0.0)) or 0.0
    )
    wrong_script_segments = int(transcript_trust.get('wrong_script_segments', 0) or 0)
    corrupted_segments = int(transcript_trust.get('corrupted_segments', 0) or 0)
    total_segments = max(int(transcript_trust.get('total_segments', 0) or 0), 1)
    contamination_burden = float(wrong_script_segments + corrupted_segments) / float(total_segments)

    suppress = False
    reason = ''
    if not display_units or trusted_units == 0 or trusted_visible_word_count == 0:
        suppress = True
        reason = 'no_trusted_display_units'
    elif (
        visible_word_count < 10
        and trusted_visible_word_count < 8
        and trusted_units <= 1
        and overall_readability < 0.24
        and lexical_trust < 0.20
        and contamination_burden >= 0.50
    ):
        suppress = True
        reason = 'weak_visible_content_and_low_trust'

    gate = {
        'suppress': suppress,
        'reason': reason,
        'trusted_visible_word_count': int(trusted_visible_word_count),
        'trusted_display_unit_count': int(trusted_units),
        'low_evidence_malayalam': bool(suppress),
        'overall_readability': round(overall_readability, 4),
        'lexical_trust_score': round(lexical_trust, 4),
    }
    logger.info(
        "[ML_DOWNSTREAM_GATE] transcript_state=%s suppress=%s reason=%s trusted_visible_word_count=%s trusted_display_unit_count=%s visible_word_count=%s overall_readability=%.3f lexical_trust=%.3f contamination_burden=%.3f",
        transcript_state,
        suppress,
        reason or 'none',
        gate['trusted_visible_word_count'],
        gate['trusted_display_unit_count'],
        visible_word_count,
        gate['overall_readability'],
        gate['lexical_trust_score'],
        contamination_burden,
    )
    return gate


def _persist_malayalam_downstream_gate_metadata(transcript_obj, gate: dict) -> None:
    if not isinstance(transcript_obj.json_data, dict):
        transcript_obj.json_data = {}
    json_data = transcript_obj.json_data
    json_data['downstream_suppressed'] = bool(gate.get('suppress', False))
    json_data['downstream_suppression_reason'] = str(gate.get('reason', '') or '')
    json_data['trusted_visible_word_count'] = int(gate.get('trusted_visible_word_count', 0) or 0)
    json_data['trusted_display_unit_count'] = int(gate.get('trusted_display_unit_count', 0) or 0)
    json_data['low_evidence_malayalam'] = bool(gate.get('low_evidence_malayalam', False))
    quality_metrics = json_data.get('quality_metrics', {}) if isinstance(json_data.get('quality_metrics', {}), dict) else {}
    if gate.get('suppress', False):
        blocked_reason = 'malayalam_source_fidelity_failed' if str(gate.get('reason', '') or '') == 'malayalam_source_fidelity_failed' else 'low_evidence_malayalam_gate'
        quality_metrics['summary_blocked_reason'] = blocked_reason
        quality_metrics['chatbot_blocked_reason'] = blocked_reason
    json_data['quality_metrics'] = quality_metrics


def _suppress_low_evidence_malayalam_downstream_outputs(video, transcript_obj, gate: dict) -> None:
    video.summaries.filter(summary_type__in=['full', 'bullet', 'short']).delete()
    video.highlight_segments.all().delete()
    logger.info(
        "[ML_SUMMARY_SKIPPED_LOW_EVIDENCE] video_id=%s transcript_id=%s reason=%s trusted_visible_word_count=%s trusted_display_unit_count=%s",
        getattr(video, 'id', ''),
        getattr(transcript_obj, 'id', ''),
        gate.get('reason', '') or 'unknown',
        int(gate.get('trusted_visible_word_count', 0) or 0),
        int(gate.get('trusted_display_unit_count', 0) or 0),
    )
    logger.info(
        "[ML_INDEX_SKIPPED_LOW_EVIDENCE] video_id=%s transcript_id=%s reason=%s",
        getattr(video, 'id', ''),
        getattr(transcript_obj, 'id', ''),
        gate.get('reason', '') or 'unknown',
    )


def _compute_transcript_state(
    *,
    cleaned_text: str,
    cleaned_segments: list,
    transcript_payload: dict,
    audio_duration_seconds: float,
    transcript_language: str,
) -> dict:
    """Deterministic transcript quality gate."""
    from . import utils as videos_utils

    qa = evaluate_transcript_quality(cleaned_text, cleaned_segments)
    asr_metrics = videos_utils._extract_asr_metrics(  # pylint: disable=protected-access
        cleaned_text,
        (transcript_payload or {}).get('word_timestamps', []) if isinstance(transcript_payload, dict) else [],
    )
    word_count = int(qa.get('word_count', 0) or 0)
    minutes = max(float(audio_duration_seconds or 0.0) / 60.0, 0.1)
    wpm = word_count / minutes
    base_score = float((transcript_payload or {}).get('transcript_quality_score', 0.0) or 0.0)
    confidence = float((transcript_payload or {}).get('confidence', 0.0) or 0.0)
    script_type = detect_script_type(cleaned_text)
    normalized_language = normalize_language_code(transcript_language, default='en', allow_auto=False)
    metadata = (transcript_payload or {}).get('metadata', {}) if isinstance(transcript_payload, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    asr_engine = str(metadata.get('asr_provider_used') or metadata.get('asr_engine_used') or '').strip()
    garble_snapshot = videos_utils._garble_debug_snapshot(  # pylint: disable=protected-access
        cleaned_text,
        language_hint=normalized_language,
    )
    garbled_score = float(garble_snapshot.get('garbled_score', 0.0) or 0.0)

    warnings = []
    state = 'cleaned'
    malayalam_post_asr_accepted = False
    malayalam_post_asr_accept_reason = ''
    malayalam_post_asr_mode = ''
    transcript_warning_message = ''
    groq_mixed_script_accepted = False

    low_content_wpm_min = float(getattr(settings, 'ASR_LOW_CONTENT_WPM_MIN', 20.0))
    long_min_words = int(getattr(settings, 'ASR_LOW_CONTENT_MIN_WORDS_LONG', 80))
    low_content_word_count = audio_duration_seconds >= 180 and word_count < long_min_words
    low_words_per_minute = audio_duration_seconds >= 180 and (word_count / minutes) < low_content_wpm_min
    text_low_content = bool(videos_utils._is_low_content_transcript(cleaned_text))  # pylint: disable=protected-access
    structurally_usable = (
        word_count >= 20
        and not text_low_content
        and qa.get('very_short_segment_ratio', 1.0) <= 0.7
        and qa.get('duplicate_word_loops', 0) < 6
        and (qa.get('final_quality_score', 0.0) >= 0.5 or base_score >= 0.8)
    )

    if normalized_language == 'ml':
        malayalam_first_pass = videos_utils._should_accept_usable_malayalam_first_pass({  # pylint: disable=protected-access
            'text': cleaned_text,
            'language': 'ml',
            'word_timestamps': (transcript_payload or {}).get('word_timestamps', []),
        })
        if malayalam_first_pass.get('first_pass_accepted'):
            malayalam_post_asr_accepted = True
            malayalam_post_asr_accept_reason = str(
                malayalam_first_pass.get('first_pass_accept_reason') or 'usable_first_pass_allowed'
            )
        elif (
            asr_engine in {'groq_whisper', 'groq_whisper_chunked'}
            and structurally_usable
            and script_type in {'mixed', 'malayalam'}
            and garbled_score < 0.72
        ):
            malayalam_post_asr_accepted = True
            malayalam_post_asr_accept_reason = 'groq_malayalam_usable_mixed_script'
            groq_mixed_script_accepted = True

    ml_final_dominant_script = str(asr_metrics.get('dominant_script', script_type) or script_type)
    ml_final_script_ratio = float(asr_metrics.get('malayalam_ratio', 0.0) or 0.0)
    ml_final_other_indic_ratio = float(asr_metrics.get('other_indic_ratio', 0.0) or 0.0)
    ml_final_token_coverage = float(asr_metrics.get('malayalam_token_coverage', 0.0) or 0.0)
    ml_transcript_trust = {}
    ml_allow_cleaned = True
    ml_accept_decision_reason = ''
    lexical_trust_score = 0.0
    pseudo_phonetic_ratio = 0.0
    low_trust_segment_share = 0.0
    overall_readability = 0.0
    corrupted_segment_share = 0.0
    source_fidelity = {}
    trusted_malayalam_segments = 0
    corrupted_segments = 0
    meaningful_malayalam_evidence = False
    if normalized_language == 'ml':
        final_min_script_ratio = float(getattr(settings, 'ASR_MALAYALAM_FINAL_MIN_SCRIPT_RATIO', 0.60))
        final_max_other_indic_ratio = float(getattr(settings, 'ASR_MALAYALAM_FINAL_MAX_OTHER_INDIC_RATIO', 0.20))
        final_min_token_coverage = float(getattr(settings, 'ASR_MALAYALAM_FINAL_MIN_TOKEN_COVERAGE', 0.55))
        final_fail_other_indic_ratio = float(getattr(settings, 'ASR_MALAYALAM_FINAL_FAIL_OTHER_INDIC_RATIO', 0.45))
        min_intended_segment_ratio = float(getattr(settings, 'ASR_MALAYALAM_FINAL_MIN_INTENDED_SEGMENT_RATIO', 0.55))
        max_corrupted_segment_share = float(getattr(settings, 'ASR_MALAYALAM_FINAL_MAX_CORRUPTED_SEGMENT_SHARE', 0.34))
        min_overall_readability = float(getattr(settings, 'ASR_MALAYALAM_FINAL_MIN_OVERALL_READABILITY', 0.34))
        max_cleaned_garble = float(getattr(settings, 'ASR_MALAYALAM_FINAL_MAX_CLEANED_GARBLE_SCORE', 0.45))
        min_lexical_trust = float(getattr(settings, 'ASR_MALAYALAM_FINAL_MIN_LEXICAL_TRUST', 0.48))
        max_pseudo_phonetic_ratio = float(getattr(settings, 'ASR_MALAYALAM_FINAL_MAX_PSEUDO_PHONETIC_RATIO', 0.30))
        max_low_trust_segment_share = float(getattr(settings, 'ASR_MALAYALAM_FINAL_MAX_LOW_TRUST_SEGMENT_SHARE', 0.34))
        ml_transcript_trust = videos_utils.build_malayalam_transcript_trust(cleaned_segments)
        source_fidelity = videos_utils.analyze_malayalam_source_fidelity(
            cleaned_segments,
            detected_language='ml',
            detected_language_confidence=confidence or metadata.get('language_detection_confidence', 0.0),
            dominant_script_final=str(ml_transcript_trust.get('dominant_script_final', '') or ''),
            script_detector_result=str(script_type or ''),
            lexical_trust_score=float(ml_transcript_trust.get('lexical_trust_score', 0.0) or 0.0),
            overall_readability=float(ml_transcript_trust.get('overall_readability', 0.0) or 0.0),
            malayalam_token_coverage=float(ml_transcript_trust.get('malayalam_token_coverage', 0.0) or 0.0),
            malayalam_dominant_segments=int(ml_transcript_trust.get('malayalam_dominant_segments', 0) or 0),
            english_only_segments=int(ml_transcript_trust.get('english_only_segments', 0) or 0),
        )
        intended_ratio = float(ml_transcript_trust.get('malayalam_intended_ratio', 0.0) or 0.0)
        corrupted_segment_share = float(ml_transcript_trust.get('corrupted_segment_share', 0.0) or 0.0)
        overall_readability = float(ml_transcript_trust.get('overall_readability', 0.0) or 0.0)
        lexical_trust_score = float(ml_transcript_trust.get('lexical_trust_score', 0.0) or 0.0)
        pseudo_phonetic_ratio = float(ml_transcript_trust.get('pseudo_phonetic_ratio', 0.0) or 0.0)
        low_trust_segments = int(ml_transcript_trust.get('low_trust_malayalam_segments', 0) or 0)
        trusted_english_ratio = float(ml_transcript_trust.get('trusted_english_preservation', 0.0) or 0.0)
        english_segments = int(ml_transcript_trust.get('english_only_segments', 0) or 0)
        mixed_segments = int(ml_transcript_trust.get('mixed_lang_segments', 0) or 0)
        clean_malayalam_segments = int(ml_transcript_trust.get('clean_malayalam_segments', 0) or 0)
        trusted_malayalam_segments = clean_malayalam_segments
        corrupted_segments = int(ml_transcript_trust.get('corrupted_segments', 0) or 0)
        low_trust_segment_share = float(low_trust_segments) / float(max(int(ml_transcript_trust.get('malayalam_intended_segments', 0) or 0), 1))
        meaningful_malayalam_evidence = bool(
            clean_malayalam_segments >= 1
            or int(ml_transcript_trust.get('trusted_malayalam_intended_segments', 0) or 0) >= 1
            or bool(source_fidelity.get('substantial_visible_malayalam', False))
            or (
                int(ml_transcript_trust.get('malayalam_dominant_segments', 0) or 0) >= 1
                and ml_final_token_coverage >= 0.18
            )
        )
        mixed_lang_override = (
            ml_final_dominant_script in {'latin', 'mixed'}
            and (clean_malayalam_segments >= 1 or mixed_segments >= 1)
            and intended_ratio >= min_intended_segment_ratio
            and corrupted_segment_share <= min(max_corrupted_segment_share, 0.12)
            and overall_readability >= min_overall_readability
            and ml_final_other_indic_ratio <= 0.08
            and ml_final_token_coverage >= max(0.35, final_min_token_coverage - 0.20)
            and (lexical_trust_score >= min_lexical_trust or trusted_english_ratio >= 0.18)
            and pseudo_phonetic_ratio <= max_pseudo_phonetic_ratio
        )
        logger.info(
            "[ML_LEXICAL_TRUST] lexical_trust_score=%.3f pseudo_phonetic_ratio=%.3f low_trust_segment_share=%.3f low_trust_segments=%s overall_readability=%.3f",
            lexical_trust_score,
            pseudo_phonetic_ratio,
            low_trust_segment_share,
            low_trust_segments,
            overall_readability,
        )
        if garbled_score >= max_cleaned_garble:
            ml_allow_cleaned = False
            ml_accept_decision_reason = 'final_garble_above_cleaned_threshold'
        elif ml_final_dominant_script != 'malayalam' and not mixed_lang_override:
            ml_allow_cleaned = False
            ml_accept_decision_reason = 'dominant_script_not_malayalam'
        elif ml_final_script_ratio < final_min_script_ratio and not mixed_lang_override:
            ml_allow_cleaned = False
            ml_accept_decision_reason = 'malayalam_ratio_too_low'
        elif ml_final_other_indic_ratio > final_max_other_indic_ratio:
            ml_allow_cleaned = False
            ml_accept_decision_reason = 'other_indic_ratio_too_high'
        elif ml_final_token_coverage < final_min_token_coverage and not mixed_lang_override:
            ml_allow_cleaned = False
            ml_accept_decision_reason = 'token_coverage_too_low'
        elif corrupted_segment_share > max_corrupted_segment_share:
            ml_allow_cleaned = False
            ml_accept_decision_reason = 'corrupted_segment_share_too_high'
        elif intended_ratio < min_intended_segment_ratio:
            ml_allow_cleaned = False
            ml_accept_decision_reason = 'malformed_malayalam_segments_too_high'
        elif overall_readability < min_overall_readability:
            ml_allow_cleaned = False
            ml_accept_decision_reason = 'overall_readability_too_low'
        elif mixed_lang_override:
            ml_accept_decision_reason = 'mixed_lang_but_trusted'
        elif lexical_trust_score < min_lexical_trust:
            ml_allow_cleaned = False
            ml_accept_decision_reason = 'semantic_trust_too_low'
        elif pseudo_phonetic_ratio > max_pseudo_phonetic_ratio:
            ml_allow_cleaned = False
            ml_accept_decision_reason = 'pseudo_phonetic_ratio_too_high'
        elif low_trust_segment_share > max_low_trust_segment_share:
            ml_allow_cleaned = False
            ml_accept_decision_reason = 'low_trust_malayalam_segment_share_too_high'
        elif mixed_segments > 0 or english_segments > 0:
            ml_accept_decision_reason = 'english_segments_excluded_from_malayalam_ratio'
        else:
            ml_accept_decision_reason = 'final_script_gate_passed'
        if ml_final_other_indic_ratio >= final_fail_other_indic_ratio:
            warnings.append('malayalam_wrong_script_dominance')
        logger.info(
            "[ML_ACCEPT] final_script=%s malayalam_ratio=%.3f other_indic_ratio=%.3f token_coverage=%.3f "
            "allow_cleaned=%s decision_reason=%s raw_active_scripts=%s final_active_scripts=%s",
            ml_final_dominant_script,
            ml_final_script_ratio,
            ml_final_other_indic_ratio,
            ml_final_token_coverage,
            ml_allow_cleaned,
            ml_accept_decision_reason or 'none',
            ",".join(garble_snapshot.get('raw_active_scripts', []) or []),
            ",".join(garble_snapshot.get('active_scripts', []) or []),
        )
        logger.info(
            "[ML_TRANSCRIPT_TRUST] total_segments=%s trusted_malayalam_segments=%s clean_english_segments=%s mixed_segments=%s corrupted_segments=%s "
            "rescued_segments=%s malayalam_intended_ratio=%.3f trusted_english_ratio=%.3f overall_readability=%.3f lexical_trust_score=%.3f pseudo_phonetic_ratio=%.3f dominant_script_final=%s",
            int(ml_transcript_trust.get('total_segments', 0) or 0),
            int(ml_transcript_trust.get('clean_malayalam_segments', 0) or 0),
            int(ml_transcript_trust.get('english_only_segments', 0) or 0),
            int(ml_transcript_trust.get('mixed_lang_segments', 0) or 0),
            int(ml_transcript_trust.get('corrupted_segments', 0) or 0),
            int(ml_transcript_trust.get('rescued_segments', 0) or 0),
            float(ml_transcript_trust.get('malayalam_intended_ratio', 0.0) or 0.0),
            float(ml_transcript_trust.get('trusted_english_preservation', 0.0) or 0.0),
            float(ml_transcript_trust.get('overall_readability', 0.0) or 0.0),
            lexical_trust_score,
            pseudo_phonetic_ratio,
            ml_transcript_trust.get('dominant_script_final', 'unknown'),
        )
        logger.info(
            "[ML_SOURCE_FIDELITY] detected_language=%s confidence=%.3f dominant_script_final=%s trusted_visible_word_count=%s trusted_display_unit_count=%s suspicious_substitution_burden=%.3f wrong_script_burden=%.3f dominant_non_malayalam_visible_ratio=%.3f english_heavy_visible_ratio=%.3f visible_malayalam_char_ratio=%.3f malayalam_token_coverage=%.3f fidelity_failed=%s",
            'ml',
            float(confidence or metadata.get('language_detection_confidence', 0.0) or 0.0),
            str(ml_transcript_trust.get('dominant_script_final', '') or 'unknown'),
            int(source_fidelity.get('trusted_visible_word_count', 0) or 0),
            int(source_fidelity.get('trusted_display_unit_count', 0) or 0),
            float(source_fidelity.get('suspicious_substitution_burden', 0.0) or 0.0),
            float(source_fidelity.get('wrong_script_burden', 0.0) or 0.0),
            float(source_fidelity.get('dominant_non_malayalam_visible_ratio', 0.0) or 0.0),
            float(source_fidelity.get('english_heavy_visible_ratio', 0.0) or 0.0),
            float(source_fidelity.get('visible_malayalam_char_ratio', 0.0) or 0.0),
            float(source_fidelity.get('malayalam_token_coverage', 0.0) or 0.0),
            bool(source_fidelity.get('source_language_fidelity_failed', False)),
        )
        logger.info(
            "[ML_ACCEPT_GUARD] garble_score=%.3f lexical_trust=%.3f pseudo_phonetic_ratio=%.3f low_trust_segment_share=%.3f allow_cleaned=%s blocked_reason=%s",
            garbled_score,
            lexical_trust_score,
            pseudo_phonetic_ratio,
            low_trust_segment_share,
            ml_allow_cleaned,
            ml_accept_decision_reason or 'none',
        )

    if videos_utils._looks_garbled_multiscript(cleaned_text):  # pylint: disable=protected-access
        if malayalam_post_asr_accepted:
            warnings.append('malayalam_mixed_script_accepted')
        else:
            warnings.append('garbled_multiscript_detected')
            state = 'failed'

    if qa.get('duplicate_word_loops', 0) >= 4:
        warnings.append('repeated_token_anomaly')
    if qa.get('lexical_diversity', 1.0) < 0.12 and word_count >= 120:
        warnings.append('low_lexical_diversity')
    if qa.get('mid_sentence_caps', 0) >= 12:
        warnings.append('capitalization_anomaly')
    if qa.get('period_cap_artifacts', 0) >= 4 or qa.get('question_dot_artifacts', 0) >= 2:
        warnings.append('punctuation_artifact')

    if low_content_word_count:
        warnings.append('low_content_word_count')
    if low_words_per_minute:
        warnings.append('low_words_per_minute')

    non_latin_languages = {'ml', 'hi', 'ta', 'te', 'kn', 'ar', 'ru', 'ja', 'ko', 'zh'}
    if normalized_language in non_latin_languages and script_type == 'latin':
        warnings.append('language_script_mismatch')

    if confidence and confidence < 0.45:
        warnings.append('low_confidence_span')

    penalty = 0.0
    penalty += 0.22 if 'garbled_multiscript_detected' in warnings else 0.0
    penalty += 0.14 if 'language_script_mismatch' in warnings else 0.0
    penalty += 0.12 if 'low_words_per_minute' in warnings else 0.0
    penalty += 0.10 if 'low_content_word_count' in warnings else 0.0
    penalty += 0.08 if 'repeated_token_anomaly' in warnings else 0.0
    penalty += 0.06 if 'low_confidence_span' in warnings else 0.0
    penalty += 0.05 if 'capitalization_anomaly' in warnings else 0.0
    penalty += 0.05 if 'punctuation_artifact' in warnings else 0.0
    penalty += 0.05 if 'low_lexical_diversity' in warnings else 0.0
    final_score = max(0.0, min(1.0, base_score - penalty))

    blocking_warnings = {
        'language_script_mismatch',
        'low_words_per_minute',
        'low_content_word_count',
        'repeated_token_anomaly',
        'low_confidence_span',
    }
    soft_warnings = {
        'capitalization_anomaly',
        'punctuation_artifact',
        'low_lexical_diversity',
    }

    if state != 'failed':
        if malayalam_post_asr_accepted:
            ignored_for_accepted_ml = {'language_script_mismatch', 'low_words_per_minute', 'low_content_word_count'}
            effective_blocking = {
                flag for flag in blocking_warnings if flag not in ignored_for_accepted_ml
            }
            if any(flag in warnings for flag in effective_blocking):
                state = 'low_confidence'
        elif any(flag in warnings for flag in blocking_warnings):
            state = 'low_confidence'
        else:
            soft_count = sum(1 for flag in warnings if flag in soft_warnings)
            if base_score < 0.55 and soft_count >= 2:
                state = 'low_confidence'

    if normalized_language == 'ml':
        malayalam_truly_unusable = (
            text_low_content
            or word_count < 12
            or qa.get('duplicate_word_loops', 0) >= 8
            or qa.get('very_short_segment_ratio', 0.0) >= 0.9
            or (base_score < 0.25 and qa.get('final_quality_score', 0.0) < 0.25)
        )
        if state == 'failed' and malayalam_post_asr_accepted and ml_allow_cleaned:
            state = 'cleaned'
        if state in {'failed', 'low_confidence'} and not malayalam_truly_unusable:
            if structurally_usable and meaningful_malayalam_evidence:
                state = 'degraded'
                warnings.append('malayalam_degraded_but_usable')
                transcript_warning_message = (
                    'Malayalam transcript quality is imperfect, but usable. Summary and chatbot are continuing in degraded mode.'
                )
                if not malayalam_post_asr_accept_reason:
                    malayalam_post_asr_accept_reason = 'mixed_script_but_structurally_usable'
            elif meaningful_malayalam_evidence:
                # Even if not structurally usable, allow degraded mode if there's meaningful evidence
                # This prevents transcripts with some quality issues from being stuck in low_confidence
                state = 'degraded'
                warnings.append('malayalam_degraded_but_usable')
                transcript_warning_message = (
                    'Malayalam transcript quality is imperfect, but usable. Summary and chatbot are continuing in degraded mode.'
                )
                if not malayalam_post_asr_accept_reason:
                    malayalam_post_asr_accept_reason = 'meaningful_evidence_despite_structural_issues'
            elif structurally_usable:
                # Has structural usability but no meaningful evidence - this is truly problematic
                state = 'failed'
                transcript_warning_message = 'Malayalam transcript did not retain enough faithful Malayalam evidence to continue safely.'
        if state == 'cleaned' and not ml_allow_cleaned:
            state = 'degraded' if not malayalam_truly_unusable else 'failed'
            warnings.append('malayalam_script_acceptance_blocked')
            malayalam_post_asr_accepted = False
            malayalam_post_asr_accept_reason = ml_accept_decision_reason or 'final_script_gate_failed'
            transcript_warning_message = (
                'Malayalam transcript is structurally usable but script purity is insufficient; continuing in degraded mode.'
                if state == 'degraded' else
                'Malayalam transcript failed final script validation.'
            )
        low_trust_state, low_trust_reason = _resolve_low_trust_malayalam_outcome(
            structurally_usable=structurally_usable,
            meaningful_malayalam_evidence=meaningful_malayalam_evidence,
            malayalam_truly_unusable=malayalam_truly_unusable,
            garble_score=garbled_score,
            lexical_trust_score=lexical_trust_score,
            overall_readability=overall_readability,
            trusted_malayalam_segments=trusted_malayalam_segments,
            corrupted_segments=corrupted_segments,
            decision_reason=ml_accept_decision_reason or '',
        )
        logger.info(
            "[ML_LOW_TRUST_OUTCOME] lexical_trust=%.3f readability=%.3f structurally_usable=%s decision=%s reason=%s",
            lexical_trust_score,
            overall_readability,
            structurally_usable,
            low_trust_state or 'unchanged',
            low_trust_reason,
        )
        if low_trust_state == 'degraded' and state in {'failed', 'low_confidence'}:
            state = 'degraded'
            warnings.append('malayalam_low_trust_degraded')
            malayalam_post_asr_accepted = False
            malayalam_post_asr_mode = 'degraded'
            malayalam_post_asr_accept_reason = ml_accept_decision_reason or low_trust_reason
            transcript_warning_message = (
                'Malayalam transcript is low-trust but structurally usable; continuing in degraded mode.'
            )
        elif low_trust_state == 'degraded' and state == 'degraded':
            transcript_warning_message = (
                'Malayalam transcript is low-trust but structurally usable; continuing in degraded mode.'
            )
        elif low_trust_state == 'failed':
            state = 'failed'
            malayalam_post_asr_accepted = False
            malayalam_post_asr_mode = 'failed'
            malayalam_post_asr_accept_reason = ml_accept_decision_reason or low_trust_reason
            transcript_warning_message = 'Malayalam transcript quality is too weak to continue safely.'

        if bool(source_fidelity.get('source_language_fidelity_failed', False)):
            state = 'source_language_fidelity_failed'
            malayalam_post_asr_accepted = False
            malayalam_post_asr_mode = 'source_language_fidelity_failed'
            malayalam_post_asr_accept_reason = 'malayalam_source_fidelity_failed'
            transcript_warning_message = 'Malayalam speech could not be transcribed faithfully enough for safe display.'
            warnings.append('malayalam_source_fidelity_failed')
            if bool(source_fidelity.get('catastrophic_latin_substitution_failure', False)):
                logger.info(
                    "[ML_HARD_FAIL_LATIN_COLLAPSE] detected_language=%s confidence=%.3f dominant_script_final=%s malayalam_token_coverage=%.3f visible_malayalam_char_ratio=%.3f trusted_visible_word_count=%s trusted_display_unit_count=%s english_only_segments=%s malayalam_dominant_segments=%s",
                    'ml',
                    float(confidence or metadata.get('language_detection_confidence', 0.0) or 0.0),
                    str(ml_transcript_trust.get('dominant_script_final', '') or 'unknown'),
                    float(source_fidelity.get('malayalam_token_coverage', 0.0) or 0.0),
                    float(source_fidelity.get('visible_malayalam_char_ratio', 0.0) or 0.0),
                    int(source_fidelity.get('trusted_visible_word_count', 0) or 0),
                    int(source_fidelity.get('trusted_display_unit_count', 0) or 0),
                    int(source_fidelity.get('english_only_segments', 0) or 0),
                    int(source_fidelity.get('malayalam_dominant_segments', 0) or 0),
                )
            if bool(source_fidelity.get('catastrophic_wrong_script_failure', False)):
                logger.info(
                    "[ML_SOURCE_FIDELITY_HARD_FAIL] detected_language=%s confidence=%.3f dominant_script_final=%s wrong_script_burden=%.3f visible_malayalam_char_ratio=%.3f trusted_visible_word_count=%s trusted_display_unit_count=%s",
                    'ml',
                    float(confidence or metadata.get('language_detection_confidence', 0.0) or 0.0),
                    str(ml_transcript_trust.get('dominant_script_final', '') or 'unknown'),
                    float(source_fidelity.get('wrong_script_burden', 0.0) or 0.0),
                    float(source_fidelity.get('visible_malayalam_char_ratio', 0.0) or 0.0),
                    int(source_fidelity.get('trusted_visible_word_count', 0) or 0),
                    int(source_fidelity.get('trusted_display_unit_count', 0) or 0),
                )
            logger.info(
                "[ML_SOURCE_FIDELITY_FAIL] detected_language=%s confidence=%.3f dominant_script_final=%s suspicious_substitution_burden=%.3f wrong_script_burden=%.3f trusted_visible_word_count=%s trusted_display_unit_count=%s",
                'ml',
                float(confidence or metadata.get('language_detection_confidence', 0.0) or 0.0),
                str(ml_transcript_trust.get('dominant_script_final', '') or 'unknown'),
                float(source_fidelity.get('suspicious_substitution_burden', 0.0) or 0.0),
                float(source_fidelity.get('wrong_script_burden', 0.0) or 0.0),
                int(source_fidelity.get('trusted_visible_word_count', 0) or 0),
                int(source_fidelity.get('trusted_display_unit_count', 0) or 0),
            )
        elif (
            state == 'degraded'
            and groq_mixed_script_accepted
            and garbled_score < 0.72
            and str(ml_transcript_trust.get('dominant_script_final', '') or '') == 'malayalam'
            and bool(source_fidelity.get('substantial_visible_malayalam', False))
            and not malayalam_truly_unusable
        ):
            state = 'cleaned'
            malayalam_post_asr_accepted = True
            malayalam_post_asr_accept_reason = 'groq_malayalam_usable_mixed_script'
            transcript_warning_message = ''

        if state == 'cleaned':
            malayalam_post_asr_mode = 'accepted'
        elif state == 'degraded':
            malayalam_post_asr_mode = malayalam_post_asr_mode or 'degraded'
        elif state == 'failed':
            malayalam_post_asr_mode = 'failed'

        logger.info(
            "[MALAYALAM_QA_DEBUG] stage=final_state_eval state=%s malayalam_post_asr_accepted=%s "
            "malayalam_post_asr_reason=%s script_detector_result=%s garble_detector_score=%.3f "
            "structurally_usable=%s warnings=%s preview=%r",
            state,
            malayalam_post_asr_accepted,
            malayalam_post_asr_accept_reason or 'none',
            script_type,
            garbled_score,
            structurally_usable,
            ",".join(warnings) if warnings else 'none',
            _safe_debug_preview(cleaned_text),
        )
        logger.info(
            "[ML_GARBLE] stage=final score=%.3f script_penalty=%.3f odd_token_penalty=%.3f replacement_penalty=%.3f "
            "dominant_script=%s dominant_ratio=%.3f active_scripts=%s raw_active_scripts=%s odd_token_ratio=%.3f "
            "malayalam_adjustment=%.3f",
            garbled_score,
            float(garble_snapshot.get('script_penalty', 0.0) or 0.0),
            float(garble_snapshot.get('odd_token_penalty', 0.0) or 0.0),
            float(garble_snapshot.get('replacement_penalty', 0.0) or 0.0),
            str(garble_snapshot.get('dominant_script', '') or 'unknown'),
            float(garble_snapshot.get('dominant_script_ratio', 0.0) or 0.0),
            ",".join(garble_snapshot.get('active_scripts', []) or []),
            ",".join(garble_snapshot.get('raw_active_scripts', []) or []),
            float(garble_snapshot.get('odd_token_ratio', 0.0) or 0.0),
            float(garble_snapshot.get('malayalam_adjustment', 0.0) or 0.0),
        )
        logger.info(
            "[ML_GARBLE_UNICODE] stage=final replacement_chars=%s suspicious_count=%s suspicious_codepoints=%s "
            "normalization_changed=%s escaped_preview=%r",
            int(garble_snapshot.get('replacement_chars', 0) or 0),
            int(garble_snapshot.get('suspicious_count', 0) or 0),
            ",".join(garble_snapshot.get('suspicious_codepoints', []) or []),
            bool(garble_snapshot.get('normalization_changed', False)),
            str(garble_snapshot.get('escaped_preview', '') or ''),
        )

    # For non-Malayalam languages (English, etc.) in low_confidence state, allow degraded mode
    # This prevents transcripts from being stuck in low_confidence when they have some usable content
    if normalized_language != 'ml' and state == 'low_confidence':
        # Allow degraded mode if there's meaningful content
        if word_count >= 10 and base_score >= 0.3:
            state = 'degraded'
            warnings.append('non_malayalam_degraded_allow')
            transcript_warning_message = (
                'Transcript has quality concerns but contains usable content. Continuing in degraded mode.'
            )

    summary_blocked_reason = ''
    chatbot_blocked_reason = ''
    if state == 'failed':
        message = 'Transcript quality failed the garble checks. Summaries and chatbot were not generated.'
        summary_blocked_reason = 'transcript_failed_garble_gate'
        chatbot_blocked_reason = 'transcript_failed_garble_gate'
    elif state == 'low_confidence':
        message = 'Transcript quality is low-confidence. Summaries and chatbot were withheld to avoid garbage output.'
        summary_blocked_reason = 'transcript_low_confidence_gate'
        chatbot_blocked_reason = 'transcript_low_confidence_gate'
    elif state == 'degraded':
        message = transcript_warning_message or 'Transcript quality is degraded, but summaries and chatbot were allowed to continue.'
    else:
        message = ''

    if normalized_language == 'ml' and bool(source_fidelity.get('source_language_fidelity_failed', False)):
        summary_blocked_reason = 'malayalam_source_fidelity_failed'
        chatbot_blocked_reason = 'malayalam_source_fidelity_failed'
        message = transcript_warning_message or 'Malayalam speech could not be transcribed faithfully enough for safe display.'

    return {
        'state': state,
        'warnings': warnings,
        'qa_metrics': {
            **qa,
            'dominant_script': ml_final_dominant_script if normalized_language == 'ml' else str(asr_metrics.get('dominant_script', script_type) or script_type),
            'malayalam_ratio': round(ml_final_script_ratio if normalized_language == 'ml' else float(asr_metrics.get('malayalam_ratio', 0.0) or 0.0), 4),
            'other_indic_ratio': round(ml_final_other_indic_ratio if normalized_language == 'ml' else float(asr_metrics.get('other_indic_ratio', 0.0) or 0.0), 4),
            'malayalam_token_coverage': round(ml_final_token_coverage if normalized_language == 'ml' else float(asr_metrics.get('malayalam_token_coverage', 0.0) or 0.0), 4),
            'words_per_minute': round(wpm, 2),
            'script_type': script_type,
            'script_detector_result': script_type,
            'confidence': round(confidence, 4),
            'garbled_detector_score': round(garbled_score, 4),
            'malayalam_post_asr_accepted': malayalam_post_asr_accepted,
            'malayalam_post_asr_mode': malayalam_post_asr_mode,
            'malayalam_post_asr_accept_reason': malayalam_post_asr_accept_reason,
            'transcript_warning_message': transcript_warning_message,
            'summary_blocked_reason': summary_blocked_reason,
            'chatbot_blocked_reason': chatbot_blocked_reason,
            'structurally_usable': structurally_usable,
            'transcript_trust': ml_transcript_trust,
            'source_language_fidelity_failed': bool(source_fidelity.get('source_language_fidelity_failed', False)),
            'transcript_fidelity_state': str(source_fidelity.get('transcript_fidelity_state', '')),
            'final_malayalam_fidelity_decision': (
                'source_language_fidelity_failed'
                if bool(source_fidelity.get('source_language_fidelity_failed', False))
                else 'cleaned'
                if state == 'cleaned'
                else 'degraded'
                if state == 'degraded'
                else state
            ),
            'catastrophic_wrong_script_failure': bool(source_fidelity.get('catastrophic_wrong_script_failure', False)),
            'suspicious_substitution_burden': round(float(source_fidelity.get('suspicious_substitution_burden', 0.0) or 0.0), 4),
            'dominant_non_malayalam_visible_ratio': round(float(source_fidelity.get('dominant_non_malayalam_visible_ratio', 0.0) or 0.0), 4),
            'meaningful_malayalam_evidence': meaningful_malayalam_evidence,
        },
        'quality_score': round(final_score, 4),
        'message': message,
        'malayalam_post_asr_mode': malayalam_post_asr_mode,
        'malayalam_post_asr_reason': malayalam_post_asr_accept_reason,
        'transcript_warning_message': transcript_warning_message,
        'summary_blocked_reason': summary_blocked_reason,
        'chatbot_blocked_reason': chatbot_blocked_reason,
    }


def _build_transcript_json_payload(
    transcript_payload,
    canonical_payload,
    *,
    draft_payload=None,
    assembled_units=None,
    display_units=None,
    internal_evidence_units=None,
    transcript_state: str = 'cleaned',
    transcript_warnings=None,
    qa_metrics=None,
    readable_transcript: str | None = None,
    display_readable_transcript: str | None = None,
    processing_time_seconds: float = 0.0,
    stage_metrics: dict | None = None,
    transcript_warning_message: str = '',
    malayalam_post_asr_mode: str = '',
    malayalam_post_asr_reason: str = '',
):
    segments = _normalized_segments(transcript_payload)
    assembled = assembled_units if isinstance(assembled_units, list) else []
    display = display_units if isinstance(display_units, list) else []
    internal_evidence = internal_evidence_units if isinstance(internal_evidence_units, list) else []
    draft_segments = _normalized_segments(draft_payload)
    canonical_segments = []
    if isinstance(canonical_payload, dict):
        raw_canon_segments = canonical_payload.get('canonical_segments', [])
        if isinstance(raw_canon_segments, list):
            canonical_segments = raw_canon_segments
    assembled_text = " ".join(str(seg.get('text') or '').strip() for seg in assembled if isinstance(seg, dict)).strip()
    display_text = " ".join(str(seg.get('text') or '').strip() for seg in display if isinstance(seg, dict)).strip()
    cleaned_text = readable_transcript if readable_transcript is not None else (display_text or assembled_text or ((transcript_payload or {}).get('text', '') if isinstance(transcript_payload, dict) else ''))
    visible_display_text = display_readable_transcript if display_readable_transcript is not None else (display_text or cleaned_text)
    captions = _build_caption_artifacts(assembled or segments)
    quality_score = float((transcript_payload or {}).get('transcript_quality_score', 0.0) or 0.0) if isinstance(transcript_payload, dict) else 0.0
    metadata = (transcript_payload or {}).get('metadata', {}) if isinstance(transcript_payload, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    original_language = str((transcript_payload or {}).get('language', '') if isinstance(transcript_payload, dict) else '').strip()
    source_language_fidelity_failed = bool((qa_metrics or {}).get('source_language_fidelity_failed', False))
    transcript_fidelity_state = str((qa_metrics or {}).get('transcript_fidelity_state', '') or '')
    detected_language_confidence = float(
        (transcript_payload or {}).get('detection_confidence', 0.0) if isinstance(transcript_payload, dict) else 0.0
    ) or float(metadata.get('language_detection_confidence', 0.0) or 0.0)
    all_text = " ".join(
        str(seg.get('text') or '').strip()
        for seg in (display or assembled or segments)
        if isinstance(seg, dict) and str(seg.get('text') or '').strip()
    ).strip()
    english_token_hits = len(re.findall(r"\b[A-Za-z]{3,}\b", all_text))
    non_ascii_alpha_hits = len(re.findall(r"[^\x00-\x7F]", all_text))
    is_multilingual_content = bool(original_language and original_language != 'en' and english_token_hits > 0 and non_ascii_alpha_hits > 0)
    normalized_original_language = normalize_language_code(original_language, default='en', allow_auto=False)
    if normalized_original_language == 'ml' and str(transcript_state or '').strip().lower() in {'draft', 'processing', 'pending'}:
        english_view_available = False
        translation_state = 'pending'
        translation_blocked_reason = 'pending_final_malayalam_state'
        current_available_views = ['original']
        transcript_display_mode = 'visible'
    elif source_language_fidelity_failed and normalized_original_language == 'ml':
        visible_display_text = ''
        english_view_available = False
        translation_state = 'blocked'
        translation_blocked_reason = 'source_language_fidelity_failed'
        current_available_views = ['original']
        transcript_display_mode = 'suppressed_unfaithful_source'
        logger.info(
            "[TRANSCRIPT_DISPLAY_SUPPRESSED] transcript_state=%s visible_display=False reason=source_language_fidelity_failed",
            str(transcript_state or '').strip().lower() or 'unknown',
        )
    else:
        english_view_available = bool(original_language and original_language != 'en' and (visible_display_text or canonical_segments))
        translation_state = 'available' if english_view_available else ('same_as_original' if original_language == 'en' else 'blocked')
        translation_blocked_reason = '' if english_view_available or original_language == 'en' else 'insufficient_grounded_text'
        current_available_views = ['original', 'english'] if english_view_available else ['original']
        transcript_display_mode = 'visible'
    logger.info(
        "[LANG_DETECT] detected_language=%s confidence=%.3f is_multilingual=%s english_view_available=%s",
        original_language,
        float(detected_language_confidence or 0.0),
        is_multilingual_content,
        english_view_available,
    )

    return {
        'segments': segments,
        'raw_transcript_segments': segments,
        'assembled_transcript_units': assembled,
        'display_transcript_units': display,
        'internal_evidence_units': internal_evidence,
        'draft_segments': draft_segments,
        'canonical_segments': canonical_segments,
        'readable_transcript': visible_display_text,
        'display_readable_transcript': visible_display_text,
        'evidence_readable_transcript': cleaned_text,
        'draft_transcript': _draft_transcript_text(draft_payload or transcript_payload),
        'captions': captions,
        'language': (transcript_payload or {}).get('language', '') if isinstance(transcript_payload, dict) else '',
        'asr_engine': metadata.get('asr_provider_used') or metadata.get('asr_engine_used', ''),
        'processing_time_seconds': round(float(processing_time_seconds or 0.0), 4),
        'transcript_state': transcript_state,
        'transcript_warning_message': transcript_warning_message,
        'malayalam_post_asr_mode': malayalam_post_asr_mode,
        'malayalam_post_asr_reason': malayalam_post_asr_reason,
        'transcript_warnings': list(transcript_warnings or []),
        'quality_metrics': qa_metrics or {},
        'quality_badge': _quality_badge(quality_score),
        'asr_metadata': metadata,
        'processing_metrics': stage_metrics or {},
        'original_language': original_language,
        'detected_language': original_language,
        'detected_language_confidence': round(float(detected_language_confidence or 0.0), 4),
        'is_multilingual_content': is_multilingual_content,
        'english_view_available': english_view_available,
        'current_available_views': current_available_views,
        'translation_state': translation_state,
        'translation_blocked_reason': translation_blocked_reason,
        'source_language_fidelity_failed': source_language_fidelity_failed,
        'transcript_fidelity_state': transcript_fidelity_state,
        'final_malayalam_fidelity_decision': str((qa_metrics or {}).get('final_malayalam_fidelity_decision', '') or ''),
        'catastrophic_latin_substitution_failure': bool((qa_metrics or {}).get('catastrophic_latin_substitution_failure', False)),
        'summary_blocked_reason': str((qa_metrics or {}).get('summary_blocked_reason', '') or ''),
        'chatbot_blocked_reason': str((qa_metrics or {}).get('chatbot_blocked_reason', '') or ''),
        'transcript_display_mode': transcript_display_mode,
    }


def _build_malayalam_observability(
    *,
    transcript_payload: dict,
    transcript_state: str,
    processing_metrics: dict,
) -> dict:
    metadata = (transcript_payload or {}).get('metadata', {}) if isinstance(transcript_payload, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    selected_model = str(
        metadata.get('selected_model')
        or metadata.get('actual_local_model_name')
        or metadata.get('configured_model_name')
        or ''
    ).strip()
    resolved_model = str(
        metadata.get('resolved_model_name')
        or metadata.get('actual_local_model_name')
        or selected_model
    ).strip()
    total_asr_seconds = float(
        metadata.get('total_asr_seconds', processing_metrics.get('transcription_seconds', 0.0)) or 0.0
    )
    observability = {
        'enabled': True,
        'first_pass_accepted': bool(metadata.get('first_pass_accepted', False)),
        'first_pass_accept_reason': str(metadata.get('first_pass_accept_reason', '') or ''),
        'retry_considered': bool(metadata.get('retry_considered', False)),
        'retry_executed': bool(metadata.get('retry_executed', False)),
        'retry_skipped_reason': str(metadata.get('retry_skipped_reason', '') or ''),
        'retry_decision_reason': str(metadata.get('retry_decision_reason', '') or ''),
        'transcript_state': str(transcript_state or ''),
        'state_bucket': str(transcript_state or ''),
        'fallback_triggered': bool(metadata.get('fallback_triggered', False)),
        'fallback_reason': str(metadata.get('fallback_reason', '') or ''),
        'model_path': str(metadata.get('asr_route_reason', '') or ''),
        'selected_model': selected_model,
        'resolved_model': resolved_model,
        'provider': str(metadata.get('asr_provider_used') or metadata.get('asr_engine_used') or ''),
        'forced_transcription_language': str(metadata.get('forced_transcription_language', '') or ''),
        'asr_task_used': str(metadata.get('asr_task_used', '') or ''),
        'malayalam_local_prompt_bias_used': bool(metadata.get('malayalam_local_prompt_bias_used', False)),
        'total_asr_seconds': round(total_asr_seconds, 4),
        'first_pass_transcription_seconds': round(float(metadata.get('first_pass_transcription_seconds', 0.0) or 0.0), 4),
        'retry_transcription_seconds': round(float(metadata.get('retry_transcription_seconds', 0.0) or 0.0), 4),
        'total_asr_passes': int(metadata.get('total_asr_passes', 1) or 1),
        'transcript_quality_gate_passed': bool(metadata.get('transcript_quality_gate_passed', False)),
        'garbled_detector_score': round(float(metadata.get('garbled_detector_score', 0.0) or 0.0), 4),
        'language': str((transcript_payload or {}).get('language', '') or ''),
        'confusion_retry_candidate': bool(metadata.get('confusion_retry_candidate', False)),
        'confusion_retry_executed': bool(metadata.get('confusion_retry_executed', False)),
        'confusion_retry_model': str(metadata.get('confusion_retry_model', '') or ''),
        'confusion_retry_improved': bool(metadata.get('confusion_retry_improved', False)),
        'confusion_retry_improvement_reason': str(metadata.get('confusion_retry_improvement_reason', '') or ''),
        'confusion_retry_before_after': dict(metadata.get('confusion_retry_before_after', {}) or {}),
        'malayalam_specialist_recovery_attempted': bool(metadata.get('malayalam_specialist_recovery_attempted', False)),
        'malayalam_specialist_recovery_applied': bool(metadata.get('malayalam_specialist_recovery_applied', False)),
        'malayalam_specialist_recovery_reason': str(metadata.get('malayalam_specialist_recovery_reason', '') or ''),
        'malayalam_specialist_recovery_blocked_reason': str(metadata.get('malayalam_specialist_recovery_blocked_reason', '') or ''),
        'malayalam_specialist_backend': str(metadata.get('malayalam_specialist_backend', '') or ''),
        'malayalam_linguistic_correction_attempted': bool(metadata.get('malayalam_linguistic_correction_attempted', False)),
        'malayalam_linguistic_correction_applied': bool(metadata.get('malayalam_linguistic_correction_applied', False)),
        'malayalam_linguistic_correction_reason': str(metadata.get('malayalam_linguistic_correction_reason', '') or ''),
        'malayalam_linguistic_correction_blocked_reason': str(metadata.get('malayalam_linguistic_correction_blocked_reason', '') or ''),
        'malayalam_linguistic_correction_backend': str(metadata.get('malayalam_linguistic_correction_backend', '') or ''),
        'malayalam_final_candidate_source': str(metadata.get('malayalam_final_candidate_source', '') or ''),
        'summary_blocked_reason': str(processing_metrics.get('summary_blocked_reason', '') or ''),
        'chatbot_blocked_reason': str(processing_metrics.get('chatbot_blocked_reason', '') or ''),
        'downstream_suppression_reason': str(processing_metrics.get('downstream_suppression_reason', '') or ''),
        'trusted_visible_word_count': int(processing_metrics.get('trusted_visible_word_count', 0) or 0),
        'trusted_display_unit_count': int(processing_metrics.get('trusted_display_unit_count', 0) or 0),
        'lexical_trust_score': round(float(processing_metrics.get('lexical_trust_score', 0.0) or 0.0), 4),
        'overall_readability': round(float(processing_metrics.get('overall_readability', 0.0) or 0.0), 4),
        'final_downstream_suppressed': bool(processing_metrics.get('downstream_suppressed', False)),
        'final_transcript_state': str(transcript_state or ''),
        'source_language_fidelity_failed': bool(processing_metrics.get('source_language_fidelity_failed', False)),
        'transcript_fidelity_state': str(processing_metrics.get('transcript_fidelity_state', '') or ''),
    }
    return observability


def _highlight_fields(highlight):
    """Normalize highlight keys from legacy and new detectors."""
    return {
        'start_time': highlight.get('start_time', highlight.get('start', 0)),
        'end_time': highlight.get('end_time', highlight.get('end', 0)),
        'importance_score': highlight.get('importance_score', highlight.get('score', 0.5)),
        'reason': highlight.get('reason', ''),
        'transcript_snippet': highlight.get('transcript_snippet', highlight.get('text', ''))
    }


def _create_draft_transcript_record(video, transcript_payload, source_language, script_type, asr_engine, detection_confidence):
    """Persist draft transcript early for time-to-first-value."""
    draft_text = _draft_transcript_text(transcript_payload)
    draft_payload = {
        **(transcript_payload or {}),
        'text': draft_text,
        'transcript_quality_score': float((transcript_payload or {}).get('transcript_quality_score', 0.0) or 0.0),
    }
    transcript_obj = Transcript.objects.create(
        video=video,
        language=(transcript_payload or {}).get('language', source_language) or source_language,
        full_text=draft_text,
        transcript_language=source_language,
        canonical_language='en',
        script_type=script_type,
        asr_engine=asr_engine,
        asr_engine_used=asr_engine,
        detection_confidence=detection_confidence,
        transcript_quality_score=float((transcript_payload or {}).get('transcript_quality_score', 0.0) or 0.0),
        transcript_original_text=draft_text,
        transcript_canonical_text='',
        transcript_canonical_en_text='',
        json_data=_build_transcript_json_payload(
            draft_payload,
            {'canonical_segments': []},
            draft_payload=draft_payload,
            transcript_state='draft',
            transcript_warnings=[],
            qa_metrics={},
            readable_transcript=draft_text,
            processing_time_seconds=0.0,
            stage_metrics={},
            transcript_warning_message='',
            malayalam_post_asr_mode='',
            malayalam_post_asr_reason='',
        ),
        word_timestamps=(transcript_payload or {}).get('word_timestamps', []),
    )
    return transcript_obj


def _finalize_transcript_record(transcript_obj, transcript_payload, canonical_payload, *, audio_duration_seconds: float, audio_path: str = ""):
    """Replace draft transcript with cleaned, quality-gated transcript."""
    draft_payload = {
        **(transcript_payload or {}),
        'text': _draft_transcript_text(transcript_payload),
    }
    draft_segments = _normalized_segments(draft_payload)
    clean_start = timezone.now()
    cleaned = clean_transcript(draft_segments or _normalized_segments(transcript_payload))
    cleaned_segments = cleaned.get('segments') or draft_segments
    cleaned_text = (
        cleaned.get('full_text')
        or cleaned.get('text')
        or (transcript_payload or {}).get('text', '')
        or _draft_transcript_text(transcript_payload)
    )
    transcript_language = transcript_obj.transcript_language or transcript_obj.language or 'en'
    quality_before = None
    assembled_units = []
    display_units = []
    internal_evidence_units = []
    display_readable_transcript = cleaned_text
    if normalize_language_code(transcript_language, default='en', allow_auto=False) == 'ml':
        from . import utils as videos_utils

        raw_text = _draft_transcript_text(transcript_payload)
        raw_script = detect_script_type(raw_text)
        cleaned_script = detect_script_type(cleaned_text)
        raw_garble_snapshot = videos_utils._garble_debug_snapshot(raw_text, language_hint='ml')  # pylint: disable=protected-access
        cleaned_garble_snapshot = videos_utils._garble_debug_snapshot(cleaned_text, language_hint='ml')  # pylint: disable=protected-access
        raw_garble = float(raw_garble_snapshot.get('garbled_score', 0.0) or 0.0)
        cleaned_garble = float(cleaned_garble_snapshot.get('garbled_score', 0.0) or 0.0)
        logger.info(
            "[MALAYALAM_QA_DEBUG] stage=pre_state_eval raw_preview=%r cleaned_preview=%r "
            "raw_script=%s cleaned_script=%s raw_garble=%.3f cleaned_garble=%.3f "
            "raw_chars=%s cleaned_chars=%s",
            _safe_debug_preview(raw_text),
            _safe_debug_preview(cleaned_text),
            raw_script,
            cleaned_script,
            raw_garble,
            cleaned_garble,
            len(raw_text or ''),
            len(cleaned_text or ''),
        )
        logger.info(
            "[ML_GARBLE] stage=pre_repair score=%.3f script_penalty=%.3f odd_token_penalty=%.3f replacement_penalty=%.3f "
            "dominant_script=%s dominant_ratio=%.3f active_scripts=%s raw_active_scripts=%s odd_token_ratio=%.3f malayalam_adjustment=%.3f",
            raw_garble,
            float(raw_garble_snapshot.get('script_penalty', 0.0) or 0.0),
            float(raw_garble_snapshot.get('odd_token_penalty', 0.0) or 0.0),
            float(raw_garble_snapshot.get('replacement_penalty', 0.0) or 0.0),
            str(raw_garble_snapshot.get('dominant_script', '') or 'unknown'),
            float(raw_garble_snapshot.get('dominant_script_ratio', 0.0) or 0.0),
            ",".join(raw_garble_snapshot.get('active_scripts', []) or []),
            ",".join(raw_garble_snapshot.get('raw_active_scripts', []) or []),
            float(raw_garble_snapshot.get('odd_token_ratio', 0.0) or 0.0),
            float(raw_garble_snapshot.get('malayalam_adjustment', 0.0) or 0.0),
        )
        logger.info(
            "[ML_GARBLE_UNICODE] stage=pre_repair replacement_chars=%s suspicious_count=%s suspicious_codepoints=%s "
            "normalization_changed=%s escaped_preview=%r",
            int(raw_garble_snapshot.get('replacement_chars', 0) or 0),
            int(raw_garble_snapshot.get('suspicious_count', 0) or 0),
            ",".join(raw_garble_snapshot.get('suspicious_codepoints', []) or []),
            bool(raw_garble_snapshot.get('normalization_changed', False)),
            str(raw_garble_snapshot.get('escaped_preview', '') or ''),
        )
        logger.info(
            "[ML_GARBLE] stage=post_cleanup score=%.3f script_penalty=%.3f odd_token_penalty=%.3f replacement_penalty=%.3f "
            "dominant_script=%s dominant_ratio=%.3f active_scripts=%s raw_active_scripts=%s odd_token_ratio=%.3f malayalam_adjustment=%.3f",
            cleaned_garble,
            float(cleaned_garble_snapshot.get('script_penalty', 0.0) or 0.0),
            float(cleaned_garble_snapshot.get('odd_token_penalty', 0.0) or 0.0),
            float(cleaned_garble_snapshot.get('replacement_penalty', 0.0) or 0.0),
            str(cleaned_garble_snapshot.get('dominant_script', '') or 'unknown'),
            float(cleaned_garble_snapshot.get('dominant_script_ratio', 0.0) or 0.0),
            ",".join(cleaned_garble_snapshot.get('active_scripts', []) or []),
            ",".join(cleaned_garble_snapshot.get('raw_active_scripts', []) or []),
            float(cleaned_garble_snapshot.get('odd_token_ratio', 0.0) or 0.0),
            float(cleaned_garble_snapshot.get('malayalam_adjustment', 0.0) or 0.0),
        )
        logger.info(
            "[ML_GARBLE_UNICODE] stage=post_cleanup replacement_chars=%s suspicious_count=%s suspicious_codepoints=%s "
            "normalization_changed=%s escaped_preview=%r",
            int(cleaned_garble_snapshot.get('replacement_chars', 0) or 0),
            int(cleaned_garble_snapshot.get('suspicious_count', 0) or 0),
            ",".join(cleaned_garble_snapshot.get('suspicious_codepoints', []) or []),
            bool(cleaned_garble_snapshot.get('normalization_changed', False)),
            str(cleaned_garble_snapshot.get('escaped_preview', '') or ''),
        )
        if cleaned_garble > raw_garble or (raw_script == 'malayalam' and cleaned_script != 'malayalam'):
            logger.warning(
                "[MALAYALAM_QA_DEBUG] stage=cleanup_penalty_detected raw_script=%s cleaned_script=%s "
                "raw_garble=%.3f cleaned_garble=%.3f raw_preview=%r cleaned_preview=%r",
                raw_script,
                cleaned_script,
                raw_garble,
                cleaned_garble,
                _safe_debug_preview(raw_text),
                _safe_debug_preview(cleaned_text),
            )
        quality_before = _compute_transcript_state(
            cleaned_text=cleaned_text,
            cleaned_segments=cleaned_segments,
            transcript_payload=transcript_payload,
            audio_duration_seconds=audio_duration_seconds,
            transcript_language=transcript_language,
        )
        qa_before = quality_before.get('qa_metrics', {}) if isinstance(quality_before, dict) else {}
        initial_state = str(quality_before.get('state') or '')
        if initial_state in {'degraded', 'failed', 'low_confidence'} or cleaned_script == 'mixed':
            repaired = videos_utils.repair_malayalam_degraded_transcript(
                cleaned_text,
                cleaned_segments,
            )
            repaired_text = str(repaired.get('text') or cleaned_text)
            repaired_segments = repaired.get('segments') or cleaned_segments
            repaired_meta = repaired.get('metadata') if isinstance(repaired.get('metadata'), dict) else {}
            quality_after = _compute_transcript_state(
                cleaned_text=repaired_text,
                cleaned_segments=repaired_segments,
                transcript_payload=transcript_payload,
                audio_duration_seconds=audio_duration_seconds,
                transcript_language=transcript_language,
            )
            use_repaired, repair_reason = _should_use_malayalam_repair_candidate(
                cleaned_text,
                repaired_text,
                quality_before,
                quality_after,
            )
            logger.info(
                "[ML_CLEAN] qa_before=%s qa_after=%s selected=%s reason=%s script_before=%s script_after=%s",
                initial_state or 'unknown',
                str(quality_after.get('state') or 'unknown'),
                use_repaired,
                repair_reason,
                cleaned_script,
                detect_script_type(repaired_text),
            )
            logger.info(
                "[ML_LEX] qa_before=%s qa_after=%s garble_before=%.3f garble_after=%.3f selected=%s selected_reason=%s",
                initial_state or 'unknown',
                str(quality_after.get('state') or 'unknown'),
                float((qa_before or {}).get('garbled_detector_score', 0.0) or 0.0),
                float((quality_after.get('qa_metrics', {}) or {}).get('garbled_detector_score', 0.0) or 0.0),
                use_repaired,
                repair_reason,
            )
            if use_repaired:
                cleaned_text = repaired_text
                cleaned_segments = repaired_segments
                cleaned['metadata'] = {
                    **(cleaned.get('metadata') if isinstance(cleaned.get('metadata'), dict) else {}),
                    'malayalam_cleaning': repaired_meta,
                }
                quality_before = quality_after
        if audio_path and cleaned_segments:
            pre_rescue_trust = videos_utils.build_malayalam_transcript_trust(cleaned_segments)
            skip_rescue, skip_rescue_reason = videos_utils.should_skip_malayalam_segment_rescue(
                transcript_trust=pre_rescue_trust,
                quality=quality_before,
            )
            rescue_meta = {
                'attempted_segments': 0,
                'rescued_segments': 0,
                'selected_candidates': [],
                'fast_local_wins': 0,
                'quality_local_wins': 0,
                'medium_context_wins': 0,
                'wide_context_wins': 0,
                'planned_audio_seconds': 0.0,
                'skip_reason': skip_rescue_reason,
            }
            bad_segments_all = [] if skip_rescue else videos_utils.detect_bad_malayalam_segments(cleaned_segments)
            bad_segments_all = sorted(
                bad_segments_all,
                key=lambda item: (
                    -float(((item or {}).get('recoverability', {}) or {}).get('score', 0.0) or 0.0),
                    int((item or {}).get('idx', 0) or 0),
                ),
            )
            clustered_bad_segments = 0
            for i in range(len(bad_segments_all) - 1):
                if int(bad_segments_all[i + 1].get('idx', -10)) - int(bad_segments_all[i].get('idx', -100)) == 1:
                    clustered_bad_segments += 1
            retry_limit = int(getattr(settings, 'ASR_MALAYALAM_RESCUE_MAX_SEGMENTS', 2))
            rescue_audio_limit = float(getattr(settings, 'ASR_MALAYALAM_RESCUE_MAX_TOTAL_AUDIO_SECONDS', 30.0))
            bad_segments: list[dict] = []
            planned_audio_seconds = 0.0
            for bad in bad_segments_all:
                idx = int(bad.get('idx', -1))
                if idx < 0 or idx >= len(cleaned_segments):
                    continue
                candidate_windows = videos_utils._build_malayalam_rescue_windows(cleaned_segments[idx], segments=cleaned_segments)  # pylint: disable=protected-access
                candidate_duration = float((candidate_windows[0] or {}).get('duration', 0.0) or 0.0) if candidate_windows else 0.0
                if len(bad_segments) >= retry_limit:
                    break
                if planned_audio_seconds + candidate_duration > rescue_audio_limit and bad_segments:
                    break
                planned_audio_seconds += candidate_duration
                bad['rescue_candidate_rank'] = len(bad_segments) + 1
                bad_segments.append(bad)
            rescue_meta['planned_audio_seconds'] = round(planned_audio_seconds, 3)
            logger.info(
                "[ML_RESCUE_RANKING] ranked=%s",
                [
                    {
                        'idx': int(item.get('idx', -1)),
                        'recoverability': round(float(((item.get('recoverability') or {}).get('score', 0.0) or 0.0)), 3),
                        'reason': str(((item.get('recoverability') or {}).get('reason', '')) or 'none'),
                    }
                    for item in bad_segments_all[:6]
                ],
            )
            logger.info(
                "[ML_RESCUE_PLAN] transcript_state=%s eligible_segments=%s selected_for_retry=%s clustered_bad_segments=%s audio_path_present=%s retry_limit=%s planned_audio_seconds=%.3f skipped=%s skip_reason=%s",
                initial_state or 'unknown',
                len(bad_segments_all),
                len(bad_segments),
                clustered_bad_segments,
                bool(audio_path),
                retry_limit,
                planned_audio_seconds,
                skip_rescue,
                skip_rescue_reason or 'none',
            )
            if bad_segments:
                updated_segments = list(cleaned_segments)
                baseline_kept = 0
                quality_local_wins = 0
                repaired_baseline_wins = 0
                for bad in bad_segments:
                    idx = int(bad.get('idx', -1))
                    if idx < 0 or idx >= len(updated_segments):
                        continue
                    base_segment = dict(updated_segments[idx])
                    repaired_single = videos_utils.repair_malayalam_degraded_transcript(
                        str(base_segment.get('text') or ''),
                        [base_segment],
                    )
                    repaired_single_text = str(
                        ((repaired_single.get('segments') or [base_segment])[0].get('text'))
                        if (repaired_single.get('segments') or [base_segment]) else base_segment.get('text', '')
                    )
                    allow_local_segment_rescue, local_segment_rescue_reason = videos_utils.should_attempt_malayalam_local_segment_rescue()
                    if allow_local_segment_rescue:
                        local_rescue = videos_utils.rescue_malayalam_segment_with_local_large_v3(
                            audio_path,
                            base_segment,
                            segments=updated_segments,
                        )
                    else:
                        logger.info(
                            "[ML_SEGMENT_LOCAL_RESCUE_SKIPPED] idx=%s reason=%s",
                            idx,
                            local_segment_rescue_reason or 'none',
                        )
                        local_rescue = {
                            'text': str(base_segment.get('text') or ''),
                            'segments': [base_segment],
                            'rescue_available': False,
                            'candidates': {},
                        }
                    selected_name, selected_reason, scored = videos_utils.choose_best_malayalam_segment_candidate({
                        'baseline': str(base_segment.get('text') or ''),
                        'repaired_baseline': repaired_single_text,
                        'fast_local': local_rescue.get('candidates', {}).get('fast_local') or local_rescue.get('candidates', {}).get('tight_fast') or str(local_rescue.get('fast_text') or ''),
                        'quality_local': local_rescue.get('candidates', {}).get('quality_local') or local_rescue.get('candidates', {}).get('tight_quality') or str(local_rescue.get('quality_text') or ''),
                    })
                    before_snapshot = scored.get('baseline', {})
                    after_snapshot = scored.get(selected_name, {})
                    logger.info(
                        "[ML_SEGMENT_RESCUE] idx=%s selected=%s reason=%s readability_before=%.3f readability_after=%.3f malformed_before=%.3f malformed_after=%.3f",
                        idx,
                        selected_name,
                        selected_reason,
                        float(before_snapshot.get('score', 0.0) or 0.0),
                        float(after_snapshot.get('score', 0.0) or 0.0),
                        float(before_snapshot.get('malformed_density', 0.0) or 0.0),
                        float(after_snapshot.get('malformed_density', 0.0) or 0.0),
                    )
                    logger.info(
                        "[ML_SEGMENT_DECISION] idx=%s candidates=baseline,repaired_baseline,fast_local,quality_local selected=%s reason=%s readability_baseline=%.3f readability_repaired=%.3f readability_fast_local=%.3f readability_quality_local=%.3f english_baseline=%.3f english_repaired=%.3f english_fast_local=%.3f english_quality_local=%.3f wrong_script_baseline=%.3f wrong_script_repaired=%.3f wrong_script_fast_local=%.3f wrong_script_quality_local=%.3f overrun_penalty_quality_local=%.3f neighbor_drift_quality_local=%.3f mal_ratio_before=%.3f mal_ratio_after=%.3f",
                        idx,
                        selected_name,
                        selected_reason,
                        float((scored.get('baseline', {}) or {}).get('score', 0.0) or 0.0),
                        float((scored.get('repaired_baseline', {}) or {}).get('score', 0.0) or 0.0),
                        float((scored.get('fast_local', {}) or scored.get('tight_fast', {}) or {}).get('score', 0.0) or 0.0),
                        float((scored.get('quality_local', {}) or scored.get('tight_quality', {}) or {}).get('score', 0.0) or 0.0),
                        float((scored.get('baseline', {}) or {}).get('english_preserved_ratio', 0.0) or 0.0),
                        float((scored.get('repaired_baseline', {}) or {}).get('english_preserved_ratio', 0.0) or 0.0),
                        float((scored.get('fast_local', {}) or scored.get('tight_fast', {}) or {}).get('english_preserved_ratio', 0.0) or 0.0),
                        float((scored.get('quality_local', {}) or scored.get('tight_quality', {}) or {}).get('english_preserved_ratio', 0.0) or 0.0),
                        float((scored.get('baseline', {}) or {}).get('wrong_script_ratio', 0.0) or 0.0),
                        float((scored.get('repaired_baseline', {}) or {}).get('wrong_script_ratio', 0.0) or 0.0),
                        float((scored.get('fast_local', {}) or scored.get('tight_fast', {}) or {}).get('wrong_script_ratio', 0.0) or 0.0),
                        float((scored.get('quality_local', {}) or scored.get('tight_quality', {}) or {}).get('wrong_script_ratio', 0.0) or 0.0),
                        float((scored.get('quality_local', {}) or scored.get('tight_quality', {}) or {}).get('overrun_penalty', 0.0) or 0.0),
                        float((scored.get('quality_local', {}) or scored.get('tight_quality', {}) or {}).get('neighbor_drift_penalty', 0.0) or 0.0),
                        float((scored.get('baseline', {}) or {}).get('malayalam_ratio', 0.0) or 0.0),
                        float((scored.get(selected_name, {}) or {}).get('malayalam_ratio', 0.0) or 0.0),
                    )
                    rescue_meta['attempted_segments'] += 1
                    rescue_meta['selected_candidates'].append({
                        'idx': idx,
                        'selected': selected_name,
                        'reason': selected_reason,
                        'recoverability': float(((bad.get('recoverability') or {}).get('score', 0.0) or 0.0)),
                        'rank': int(bad.get('rescue_candidate_rank', 0) or 0),
                    })
                    if selected_name == 'baseline':
                        baseline_kept += 1
                    elif selected_name in {'tight_quality', 'medium_quality', 'wide_quality', 'quality_local'}:
                        quality_local_wins += 1
                        rescue_meta['quality_local_wins'] = int(rescue_meta.get('quality_local_wins', 0) or 0) + 1
                        if selected_name == 'medium_quality':
                            rescue_meta['medium_context_wins'] = int(rescue_meta.get('medium_context_wins', 0) or 0) + 1
                        elif selected_name == 'wide_quality':
                            rescue_meta['wide_context_wins'] = int(rescue_meta.get('wide_context_wins', 0) or 0) + 1
                    elif selected_name in {'fast_local', 'tight_fast'}:
                        rescue_meta['fast_local_wins'] = int(rescue_meta.get('fast_local_wins', 0) or 0) + 1
                    elif selected_name == 'repaired_baseline':
                        repaired_baseline_wins += 1
                    selected_text = {
                        'baseline': str(base_segment.get('text') or ''),
                        'repaired_baseline': repaired_single_text,
                        'fast_local': str(local_rescue.get('tight_fast_text') or local_rescue.get('fast_text') or ''),
                        'quality_local': str(local_rescue.get('tight_quality_text') or local_rescue.get('quality_text') or ''),
                        'tight_fast': str(local_rescue.get('tight_fast_text') or local_rescue.get('fast_text') or ''),
                        'tight_quality': str(local_rescue.get('tight_quality_text') or local_rescue.get('quality_text') or ''),
                    }.get(selected_name, str(base_segment.get('text') or ''))
                    if selected_name != 'baseline' and selected_text.strip():
                        updated = dict(base_segment)
                        updated['text'] = selected_text.strip()
                        updated_segments[idx] = updated
                        rescue_meta['rescued_segments'] += 1
                logger.info(
                    "[ML_RESCUE_RESULT] retried=%s replaced=%s baseline_kept=%s quality_local_wins=%s fast_local_wins=%s medium_context_wins=%s wide_context_wins=%s repaired_baseline_wins=%s",
                    rescue_meta['attempted_segments'],
                    rescue_meta['rescued_segments'],
                    baseline_kept,
                    quality_local_wins,
                    int(rescue_meta.get('fast_local_wins', 0) or 0),
                    int(rescue_meta.get('medium_context_wins', 0) or 0),
                    int(rescue_meta.get('wide_context_wins', 0) or 0),
                    repaired_baseline_wins,
                )
                if rescue_meta['rescued_segments'] > 0:
                    cleaned_segments = updated_segments
                    cleaned_text = " ".join(str(seg.get('text') or '').strip() for seg in cleaned_segments if isinstance(seg, dict)).strip()
                    cleaned['metadata'] = {
                        **(cleaned.get('metadata') if isinstance(cleaned.get('metadata'), dict) else {}),
                        'malayalam_segment_rescue': rescue_meta,
                    }
                    quality_before = _compute_transcript_state(
                        cleaned_text=cleaned_text,
                        cleaned_segments=cleaned_segments,
                        transcript_payload=transcript_payload,
                        audio_duration_seconds=audio_duration_seconds,
                        transcript_language=transcript_language,
                    )
            else:
                logger.info(
                    "[ML_RESCUE_RESULT] retried=0 replaced=0 baseline_kept=0 quality_local_wins=0 fast_local_wins=0 medium_context_wins=0 wide_context_wins=0 repaired_baseline_wins=0"
                )
        assembly_payload = videos_utils.assemble_malayalam_transcript_units(cleaned_segments)
        assembled_units = assembly_payload.get('units') or []
        transcript_trust = videos_utils.build_malayalam_transcript_trust(cleaned_segments)
        rescue_summary = (
            (cleaned.get('metadata', {}) if isinstance(cleaned.get('metadata', {}), dict) else {})
            .get('malayalam_segment_rescue', {}) or {}
        )
        transcript_trust['rescued_segments'] = int(rescue_summary.get('rescued_segments', 0) or 0)
        transcript_trust['quality_local_wins'] = int(rescue_summary.get('quality_local_wins', 0) or 0)
        transcript_trust['medium_context_wins'] = int(rescue_summary.get('medium_context_wins', 0) or 0)
        transcript_trust['wide_context_wins'] = int(rescue_summary.get('wide_context_wins', 0) or 0)
        cleaned['metadata'] = {
            **(cleaned.get('metadata', {}) if isinstance(cleaned.get('metadata', {}), dict) else {}),
            'malayalam_transcript_assembly': assembly_payload.get('metadata', {}),
        }
        display_payload = videos_utils.build_malayalam_display_transcript_units(
            cleaned_segments,
            assembled_units,
            transcript_state=str((quality_before or {}).get('state') or 'degraded'),
            allow_accepted_first_pass_visible=bool(
                ((quality_before or {}).get('qa_metrics', {}) or {}).get('malayalam_post_asr_accepted', False)
                and not ((quality_before or {}).get('qa_metrics', {}) or {}).get('source_language_fidelity_failed', False)
            ),
        )
        display_units = display_payload.get('units') or []
        internal_evidence_units = display_payload.get('internal_evidence_units') or []
        display_readable_transcript = str(display_payload.get('readable_transcript') or '')
        if (
            normalize_language_code(transcript_language, default='en', allow_auto=False) == 'ml'
            and str((quality_before or {}).get('state') or '') == 'degraded'
            and not display_units
        ):
            logger.info(
                "[TRANSCRIPT_DISPLAY_SUPPRESSED] transcript_state=degraded visible_display=False reason=no_trusted_display_units"
            )
        cleaned['metadata'] = {
            **(cleaned.get('metadata', {}) if isinstance(cleaned.get('metadata', {}), dict) else {}),
            'malayalam_display_refinement': display_payload.get('metadata', {}),
        }
        logger.info(
            "[ML_TRANSCRIPT_TRUST] total_segments=%s malayalam_dominant_segments=%s wrong_script_segments=%s rescued_segments=%s quality_local_wins=%s medium_context_wins=%s wide_context_wins=%s english_only_segments=%s mixed_lang_segments=%s overall_readability=%.3f malayalam_token_coverage=%.3f trusted_english_preservation=%.3f dominant_script_final=%s",
            transcript_trust.get('total_segments', 0),
            transcript_trust.get('malayalam_dominant_segments', 0),
            transcript_trust.get('wrong_script_segments', 0),
            transcript_trust.get('rescued_segments', 0),
            transcript_trust.get('quality_local_wins', 0),
            transcript_trust.get('medium_context_wins', 0),
            transcript_trust.get('wide_context_wins', 0),
            transcript_trust.get('english_only_segments', 0),
            transcript_trust.get('mixed_lang_segments', 0),
            float(transcript_trust.get('overall_readability', 0.0) or 0.0),
            float(transcript_trust.get('malayalam_token_coverage', 0.0) or 0.0),
            float(transcript_trust.get('trusted_english_preservation', 0.0) or 0.0),
            transcript_trust.get('dominant_script_final', 'unknown'),
        )
    quality = quality_before or _compute_transcript_state(
        cleaned_text=cleaned_text,
        cleaned_segments=cleaned_segments,
        transcript_payload=transcript_payload,
        audio_duration_seconds=audio_duration_seconds,
        transcript_language=transcript_language,
    )
    if normalize_language_code(transcript_language, default='en', allow_auto=False) == 'ml':
        final_source_fidelity = videos_utils.analyze_malayalam_source_fidelity(
            cleaned_segments,
            detected_language='ml',
            detected_language_confidence=float(
                (transcript_payload or {}).get('confidence', 0.0) or ((transcript_payload or {}).get('metadata', {}) or {}).get('language_detection_confidence', 0.0) or 0.0
            ),
            dominant_script_final=str(transcript_trust.get('dominant_script_final', '') or (quality.get('qa_metrics', {}) or {}).get('dominant_script', '') or ''),
            script_detector_result=str((quality.get('qa_metrics', {}) or {}).get('script_detector_result', '') or ''),
            lexical_trust_score=float(transcript_trust.get('lexical_trust_score', (quality.get('qa_metrics', {}) or {}).get('lexical_trust_score', 0.0)) or 0.0),
            overall_readability=float(transcript_trust.get('overall_readability', (quality.get('qa_metrics', {}) or {}).get('overall_readability', 0.0)) or 0.0),
            malayalam_token_coverage=float(transcript_trust.get('malayalam_token_coverage', (quality.get('qa_metrics', {}) or {}).get('malayalam_token_coverage', 0.0)) or 0.0),
            malayalam_dominant_segments=int(transcript_trust.get('malayalam_dominant_segments', 0) or 0),
            english_only_segments=int(transcript_trust.get('english_only_segments', 0) or 0),
            trusted_visible_word_count=int(display_payload.get('metadata', {}).get('final_visible_word_count', 0) or 0) if isinstance(display_payload.get('metadata', {}), dict) else None,
            trusted_display_unit_count=len(display_units),
            display_units=display_units,
        )
        if bool(final_source_fidelity.get('source_language_fidelity_failed', False)):
            qa_metrics = dict((quality or {}).get('qa_metrics', {}) if isinstance((quality or {}).get('qa_metrics', {}), dict) else {})
            qa_metrics.update({
                'source_language_fidelity_failed': True,
                'transcript_fidelity_state': str(final_source_fidelity.get('transcript_fidelity_state', '') or 'source_language_fidelity_failed'),
                'final_malayalam_fidelity_decision': 'source_language_fidelity_failed',
                'catastrophic_latin_substitution_failure': bool(final_source_fidelity.get('catastrophic_latin_substitution_failure', False)),
                'trusted_visible_word_count': int(final_source_fidelity.get('trusted_visible_word_count', 0) or 0),
                'trusted_display_unit_count': int(final_source_fidelity.get('trusted_display_unit_count', 0) or 0),
                'malayalam_token_coverage': round(float(final_source_fidelity.get('malayalam_token_coverage', qa_metrics.get('malayalam_token_coverage', 0.0)) or 0.0), 4),
                'summary_blocked_reason': 'malayalam_source_fidelity_failed',
                'chatbot_blocked_reason': 'malayalam_source_fidelity_failed',
                'dominant_non_malayalam_visible_ratio': round(float(final_source_fidelity.get('dominant_non_malayalam_visible_ratio', 0.0) or 0.0), 4),
                'suspicious_substitution_burden': round(float(final_source_fidelity.get('suspicious_substitution_burden', 0.0) or 0.0), 4),
                'visible_malayalam_char_ratio': round(float(final_source_fidelity.get('visible_malayalam_char_ratio', 0.0) or 0.0), 4),
                'english_only_segments': int(final_source_fidelity.get('english_only_segments', 0) or 0),
                'malayalam_dominant_segments': int(final_source_fidelity.get('malayalam_dominant_segments', 0) or 0),
            })
            quality = {
                **quality,
                'state': 'source_language_fidelity_failed',
                'qa_metrics': qa_metrics,
                'summary_blocked_reason': 'malayalam_source_fidelity_failed',
                'chatbot_blocked_reason': 'malayalam_source_fidelity_failed',
                'malayalam_post_asr_mode': 'source_language_fidelity_failed',
                'malayalam_post_asr_reason': 'malayalam_source_fidelity_failed',
                'transcript_warning_message': 'Malayalam speech could not be transcribed faithfully enough for safe display.',
            }
            display_units = []
            display_readable_transcript = ''
            logger.info(
                "[ML_FINALIZED_AS_SOURCE_FIDELITY_FAILED] dominant_script_final=%s malayalam_token_coverage=%.3f visible_malayalam_char_ratio=%.3f trusted_visible_word_count=%s trusted_display_unit_count=%s english_only_segments=%s malayalam_dominant_segments=%s",
                str(final_source_fidelity.get('effective_dominant_script_final', '') or transcript_trust.get('dominant_script_final', '') or 'unknown'),
                float(final_source_fidelity.get('malayalam_token_coverage', 0.0) or 0.0),
                float(final_source_fidelity.get('visible_malayalam_char_ratio', 0.0) or 0.0),
                int(final_source_fidelity.get('trusted_visible_word_count', 0) or 0),
                int(final_source_fidelity.get('trusted_display_unit_count', 0) or 0),
                int(final_source_fidelity.get('english_only_segments', 0) or 0),
                int(final_source_fidelity.get('malayalam_dominant_segments', 0) or 0),
            )
    if (
        normalize_language_code(transcript_language, default='en', allow_auto=False) == 'ml'
        and bool((quality.get('qa_metrics', {}) or {}).get('source_language_fidelity_failed', False))
    ):
        display_units = []
        display_readable_transcript = ''
    processing_seconds = 0.0
    try:
        processing_seconds = max(0.0, float((timezone.now() - clean_start).total_seconds()))
    except Exception:
        processing_seconds = 0.0

    try:
        finalized_payload = {
            **(transcript_payload or {}),
            'text': cleaned_text,
            'segments': cleaned_segments,
            'assembled_transcript_units': assembled_units if normalize_language_code(transcript_language, default='en', allow_auto=False) == 'ml' else [],
            'display_transcript_units': display_units if normalize_language_code(transcript_language, default='en', allow_auto=False) == 'ml' else [],
            'internal_evidence_units': internal_evidence_units if normalize_language_code(transcript_language, default='en', allow_auto=False) == 'ml' else [],
            'transcript_quality_score': quality['quality_score'],
        }
        finalized_payload['metadata'] = {
            **((transcript_payload or {}).get('metadata', {}) if isinstance((transcript_payload or {}).get('metadata', {}), dict) else {}),
            **(cleaned.get('metadata', {}) if isinstance(cleaned.get('metadata', {}), dict) else {}),
            'transcript_trust': transcript_trust if normalize_language_code(transcript_language, default='en', allow_auto=False) == 'ml' else {},
        }
        transcript_obj.full_text = cleaned_text
        transcript_obj.transcript_original_text = cleaned_text
        transcript_obj.transcript_canonical_text = canonical_payload.get('canonical_text', cleaned_text)
        transcript_obj.transcript_canonical_en_text = canonical_payload.get('canonical_text', cleaned_text)
        transcript_obj.canonical_language = canonical_payload.get('canonical_language', 'en')
        transcript_obj.transcript_quality_score = quality['quality_score']
        existing_json = transcript_obj.json_data if isinstance(transcript_obj.json_data, dict) else {}
        transcript_obj.json_data = _build_transcript_json_payload(
            finalized_payload,
            canonical_payload,
            draft_payload=draft_payload,
            assembled_units=assembled_units if normalize_language_code(transcript_language, default='en', allow_auto=False) == 'ml' else [],
            display_units=display_units if normalize_language_code(transcript_language, default='en', allow_auto=False) == 'ml' else [],
            internal_evidence_units=internal_evidence_units if normalize_language_code(transcript_language, default='en', allow_auto=False) == 'ml' else [],
            transcript_state=quality['state'],
            transcript_warnings=quality['warnings'],
            qa_metrics=quality['qa_metrics'],
            readable_transcript=cleaned_text,
            display_readable_transcript=display_readable_transcript if normalize_language_code(transcript_language, default='en', allow_auto=False) == 'ml' else cleaned_text,
            processing_time_seconds=processing_seconds,
            stage_metrics=existing_json.get('processing_metrics', {}),
            transcript_warning_message=quality.get('transcript_warning_message', ''),
            malayalam_post_asr_mode=quality.get('malayalam_post_asr_mode', ''),
            malayalam_post_asr_reason=quality.get('malayalam_post_asr_reason', ''),
        )
        transcript_obj.word_timestamps = (transcript_payload or {}).get('word_timestamps', [])
        transcript_obj.save(update_fields=[
            'full_text',
            'transcript_original_text',
            'transcript_canonical_text',
            'transcript_canonical_en_text',
            'canonical_language',
            'transcript_quality_score',
            'json_data',
            'word_timestamps',
        ])
    except Exception as exc:
        is_ml = normalize_language_code(transcript_language, default='en', allow_auto=False) == 'ml'
        if is_ml and _is_degraded_low_trust_malayalam_quality(quality, transcript_language) and _is_low_trust_malayalam_exception(exc):
            _persist_minimal_low_trust_malayalam_checkpoint(
                transcript_obj,
                transcript_payload=transcript_payload,
                canonical_payload=canonical_payload,
                cleaned_text=cleaned_text,
                cleaned_segments=cleaned_segments,
                processing_seconds=processing_seconds,
                quality=quality,
            )
            logger.warning(
                "[ML_LOW_TRUST_CONTINUE] transcript_state=%s reason=%s exception=%s decision=continue_degraded source=finalize_checkpoint",
                quality.get('state', ''),
                quality.get('malayalam_post_asr_reason', '') or quality.get('qa_metrics', {}).get('malayalam_post_asr_accept_reason', ''),
                str(exc),
            )
        else:
            raise
    return transcript_obj, quality


def _upsert_summary(video, summary_type: str, summary_result: dict, default_title: str):
    summary_data = _summary_fields(summary_result, default_title)
    Summary.objects.update_or_create(
        video=video,
        summary_type=summary_type,
        defaults={
            'title': summary_data['title'],
            'content': summary_data['content'],
            'key_topics': summary_data['key_topics'],
            'summary_language': summary_data['summary_language'],
            'summary_source_language': summary_data['summary_source_language'],
            'translation_used': summary_data['translation_used'],
            'model_used': summary_data['model_used'],
            'generation_time': summary_data['generation_time']
        }
    )


def _upsert_all_summaries(
    video,
    transcript_obj,
    summary_types=None,
    output_language: str = 'auto',
    source_language: str = 'en',
    summary_language_mode: str = 'same_as_transcript'
):
    """Create/update full, bullet, and short summaries with consistent logic."""
    transcript_text = (
        transcript_obj.transcript_original_text
        or transcript_obj.full_text
        or ''
    )
    canonical_text = transcript_obj.transcript_canonical_text or ''
    canonical_language = transcript_obj.canonical_language or 'en'
    transcript_language = transcript_obj.transcript_language or source_language or 'en'
    transcript_json = transcript_obj.json_data if isinstance(getattr(transcript_obj, "json_data", None), dict) else {}
    has_fidelity_gaps = bool(transcript_json.get("has_fidelity_gaps", False))

    active_summary_types = list(summary_types or ['full', 'bullet', 'short'])
    summary_runtime_rows = []
    for summary_type in active_summary_types:
        summary_text = summarize_text(
            transcript_text,
            summary_type=summary_type,
            output_language=output_language,
            source_language=transcript_language,
            canonical_text=canonical_text,
            canonical_language=canonical_language,
            summary_language_mode=summary_language_mode,
            has_fidelity_gaps=has_fidelity_gaps,
        )
        _upsert_summary(video, summary_type, summary_text, f'{summary_type.capitalize()} Summary')
        summary_runtime_rows.append({
            'summary_type': summary_type,
            'summary_model_requested': summary_text.get('summary_model_requested', ''),
            'summary_model_used': summary_text.get('summary_model_used', summary_text.get('model_used', '')),
            'summary_model_fallback_used': bool(summary_text.get('summary_model_fallback_used', False)),
            'summary_generation_mode': summary_text.get('summary_generation_mode', ''),
            'summary_runtime_error': summary_text.get('summary_runtime_error', ''),
        })
    return summary_runtime_rows


def _run_audio_pipeline(
    video,
    *,
    chunks=None,
    audio_path: str,
    source_type: str,
    transcription_language: str = 'auto',
    output_language: str = 'auto',
    summary_language_mode: str = 'same_as_transcript',
):
    """Shared staged production pipeline for local and YouTube sources."""
    pipeline_started = timezone.now()
    transcription_started = timezone.now()
    if not video.duration:
        try:
            video.duration = get_video_duration(audio_path)
            video.save(update_fields=['duration', 'updated_at'])
        except Exception:
            pass

    _update_video_stage(video, 'transcribing', 30)
    transcript_payload = transcribe_video(
        audio_path=audio_path,
        source_type=source_type,
        requested_language=transcription_language,
        chunks=chunks,
    )
    transcription_seconds = max(0.0, float((timezone.now() - transcription_started).total_seconds()))
    transcript_payload = apply_entity_corrections(transcript_payload, video_title=video.title)

    source_language = normalize_language_code(transcript_payload.get('language', 'en'), default='en', allow_auto=False)
    resolved_output_language = resolve_output_language(output_language, source_language)
    script_type = detect_script_type(transcript_payload.get('text', ''))
    asr_engine = (transcript_payload.get('metadata', {}) or {}).get('asr_provider_used', 'faster_whisper')
    detection_confidence = float(transcript_payload.get('language_probability', 0.0) or 0.0)

    draft_transcript = _create_draft_transcript_record(
        video,
        transcript_payload,
        source_language=source_language,
        script_type=script_type,
        asr_engine=asr_engine,
        detection_confidence=detection_confidence,
    )

    _update_video_stage(video, 'cleaning_transcript', 52)
    cleanup_started = timezone.now()
    canonical_payload = build_canonical_text(
        transcript_text=transcript_payload.get('text', ''),
        transcript_segments=_normalized_segments(transcript_payload),
        transcript_language=source_language,
        canonical_language='en',
    )
    transcript_payload['canonical_segments'] = canonical_payload.get('canonical_segments', [])

    transcript_obj, quality = _finalize_transcript_record(
        draft_transcript,
        transcript_payload,
        canonical_payload,
        audio_duration_seconds=float(video.duration or 0.0),
        audio_path=audio_path,
    )
    cleanup_seconds = max(0.0, float((timezone.now() - cleanup_started).total_seconds()))

    logger.info(
        "[LANG] detected=%s inferred_script=%s chosen=%s engine=%s conf=%.2f state=%s",
        transcript_payload.get('language'),
        script_type,
        source_language,
        asr_engine,
        detection_confidence,
        quality['state'],
    )
    logger.info(
        "[TRANSCRIPT_QA] malayalam_post_asr_accepted=%s malayalam_post_asr_mode=%s "
        "malayalam_post_asr_accept_reason=%s garble_detector_score=%.3f script_detector_result=%s "
        "summary_blocked_reason=%s chatbot_blocked_reason=%s transcript_warning_message=%s",
        quality['qa_metrics'].get('malayalam_post_asr_accepted', False),
        quality['qa_metrics'].get('malayalam_post_asr_mode', ''),
        quality['qa_metrics'].get('malayalam_post_asr_accept_reason', ''),
        float(quality['qa_metrics'].get('garbled_detector_score', 0.0) or 0.0),
        quality['qa_metrics'].get('script_detector_result', ''),
        quality.get('summary_blocked_reason', ''),
        quality.get('chatbot_blocked_reason', ''),
        quality.get('transcript_warning_message', ''),
    )
    if source_language == 'ml':
        transcript_meta = (transcript_payload.get('metadata') or {}) if isinstance(transcript_payload, dict) else {}
        logger.info(
            "[ML_ASR_FINAL_ROUTE] engine=%s model=%s forced_language=%s task=%s prompt_bias=%s",
            str(transcript_meta.get('asr_provider_used', asr_engine) or asr_engine),
            str(
                transcript_meta.get('actual_local_model_name')
                or transcript_meta.get('configured_model_name')
                or transcript_meta.get('resolved_local_model_name')
                or ''
            ),
            str(transcript_meta.get('forced_transcription_language', '') or ''),
            str(transcript_meta.get('asr_task_used', '') or ''),
            bool(transcript_meta.get('malayalam_local_prompt_bias_used', False)),
        )

    if quality['state'] in {'low_confidence', 'failed'}:
        _update_video_stage(video, 'failed', 62, error_message=quality['message'])
        return {
            'status': quality['state'],
            'warning': quality['message'],
            'video_id': str(video.id),
            'transcript_id': transcript_obj.id,
        }

    summary_seconds = 0.0
    indexing_seconds = 0.0
    structured_quality_score = 0.0
    continue_low_trust_degraded = _is_degraded_low_trust_malayalam_quality(quality, source_language)
    summary_runtime_rows = []
    index_runtime = {
        'built': False,
        'embedding_model_requested': '',
        'embedding_model_used': '',
        'embedding_model_fallback_used': False,
        'embedding_blocked_reason': '',
        'embedding_runtime_error': '',
    }
    downstream_gate = _evaluate_malayalam_low_evidence_downstream_gate(transcript_obj)
    _persist_malayalam_downstream_gate_metadata(transcript_obj, downstream_gate)
    transcript_obj.save(update_fields=['json_data'])
    lite_mode = _sync_mode_lite_enabled()
    transcript_json = transcript_obj.json_data if isinstance(transcript_obj.json_data, dict) else {}
    finalized_transcript_state = str(transcript_json.get('transcript_state', '') or '').strip().lower()
    quality_transcript_state = str((quality or {}).get('state', '') or '').strip().lower()
    effective_transcript_state = finalized_transcript_state
    if (
        source_language == 'ml'
        and finalized_transcript_state in {'draft', 'processing', 'pending'}
        and quality_transcript_state in {'cleaned', 'degraded', 'failed', 'source_language_fidelity_failed'}
    ):
        # Prefer the final QA decision when the persisted row is briefly stale.
        effective_transcript_state = quality_transcript_state
    pending_malayalam_final_state = bool(
        source_language == 'ml'
        and effective_transcript_state in {'draft', 'processing', 'pending'}
    )

    try:
        _update_video_stage(video, 'transcript_ready', 65)
        summary_started = timezone.now()
        if getattr(settings, 'RENDER_TRANSCRIPT_ONLY_MODE', False):
            render_safe_summary_mode = bool(getattr(settings, 'RENDER_SAFE_SUMMARY_MODE', False))
            if render_safe_summary_mode:
                requested_summary_types = list(getattr(settings, 'RENDER_SAFE_SUMMARY_TYPES', ['short', 'bullet']) or ['short', 'bullet'])
                safe_summary_types = [summary_type for summary_type in requested_summary_types if summary_type in {'short', 'bullet', 'full'}]
                if not safe_summary_types:
                    safe_summary_types = ['short', 'bullet']
                if float(video.duration or 0.0) <= 180.0 and 'full' not in safe_summary_types:
                    safe_summary_types.append('full')
                logger.warning(
                    "[RENDER_SAFE_SUMMARY_MODE] video_id=%s transcript_id=%s summary_types=%s building_structured_summary_with_groq=true skipping_highlights_indexing=true",
                    getattr(video, 'id', ''),
                    getattr(transcript_obj, 'id', ''),
                    ",".join(safe_summary_types),
                )
                try:
                    summary_runtime_rows.extend(_upsert_all_summaries(
                        video,
                        transcript_obj,
                        summary_types=safe_summary_types,
                        output_language=resolved_output_language,
                        source_language=source_language,
                        summary_language_mode=summary_language_mode,
                    ) or [])
                    from .serializers import get_or_build_structured_summary
                    get_or_build_structured_summary(video, transcript_obj)
                    if isinstance(transcript_obj.json_data, dict):
                        structured_quality_score = float(
                            (
                                transcript_obj.json_data.get('structured_summary_cache', {})
                                or {}
                            ).get('quality_score', 0.0) or 0.0
                        )
                except Exception as exc:
                    logger.warning("Render safe structured summary build failed: %s", exc)
            else:
                logger.warning(
                    "[RENDER_TRANSCRIPT_ONLY_MODE] video_id=%s transcript_id=%s skipping_summary_highlights_indexing=true",
                    getattr(video, 'id', ''),
                    getattr(transcript_obj, 'id', ''),
                )
            summary_seconds = max(0.0, float((timezone.now() - summary_started).total_seconds()))
        elif pending_malayalam_final_state:
            logger.info(
                "[ML_SUMMARY_DEFERRED_PENDING_FINAL_STATE] video_id=%s transcript_id=%s transcript_state=%s",
                getattr(video, 'id', ''),
                getattr(transcript_obj, 'id', ''),
                effective_transcript_state or 'pending',
            )
            summary_seconds = max(0.0, float((timezone.now() - summary_started).total_seconds()))
        elif downstream_gate.get('suppress', False):
            _suppress_low_evidence_malayalam_downstream_outputs(video, transcript_obj, downstream_gate)
            if (
                source_language == 'ml'
                and str(downstream_gate.get('reason', '') or '') == 'malayalam_source_fidelity_failed'
            ):
                logger.info(
                    "[ML_FAST_EXIT_FIDELITY_FAILED] video_id=%s transcript_id=%s final_state=%s",
                    getattr(video, 'id', ''),
                    getattr(transcript_obj, 'id', ''),
                    effective_transcript_state or 'unknown',
                )
                logger.info(
                    "[ML_SUMMARY_SKIPPED_FIDELITY_FAILED_EARLY] video_id=%s transcript_id=%s",
                    getattr(video, 'id', ''),
                    getattr(transcript_obj, 'id', ''),
                )
                logger.info(
                    "[ML_EN_VIEW_SKIPPED_FIDELITY_FAILED_EARLY] video_id=%s transcript_id=%s",
                    getattr(video, 'id', ''),
                    getattr(transcript_obj, 'id', ''),
                )
            else:
                try:
                    from .serializers import get_or_build_structured_summary
                    get_or_build_structured_summary(video, transcript_obj)
                    if isinstance(transcript_obj.json_data, dict):
                        structured_quality_score = float(
                            (
                                transcript_obj.json_data.get('structured_summary_cache', {})
                                or {}
                            ).get('quality_score', 0.0) or 0.0
                        )
                except Exception as exc:
                    logger.warning("Structured summary quality metrics skipped: %s", exc)
            summary_seconds = max(0.0, float((timezone.now() - summary_started).total_seconds()))
        else:
            _update_video_stage(video, 'summarizing_quick', 74)
            summary_runtime_rows.extend(_upsert_all_summaries(
                video,
                transcript_obj,
                summary_types=['short'],
                output_language=resolved_output_language,
                source_language=source_language,
                summary_language_mode=summary_language_mode,
            ) or [])

            if lite_mode:
                logger.info(
                    "[DEV_SYNC_LITE_MODE] video_id=%s skipping_full_summaries=%s skipping_highlights=%s skipping_chat_index=%s",
                    str(video.id),
                    True,
                    True,
                    True,
                )
            else:
                _update_video_stage(video, 'summarizing_final', 84)
                summary_runtime_rows.extend(_upsert_all_summaries(
                    video,
                    transcript_obj,
                    summary_types=['full', 'bullet'],
                    output_language=resolved_output_language,
                    source_language=source_language,
                    summary_language_mode=summary_language_mode,
                ) or [])
                _rebuild_highlights(video, transcript_obj)

                _update_video_stage(video, 'indexing_chat', 94)
                indexing_started = timezone.now()
                index_runtime = _rebuild_chatbot_index(video, transcript_obj.json_data or {}) or index_runtime
                indexing_seconds = max(0.0, float((timezone.now() - indexing_started).total_seconds()))

            summary_seconds = max(0.0, float((timezone.now() - summary_started).total_seconds()))

            try:
                from .serializers import get_or_build_structured_summary
                get_or_build_structured_summary(video, transcript_obj)
                if isinstance(transcript_obj.json_data, dict):
                    structured_quality_score = float(
                        (
                            transcript_obj.json_data.get('structured_summary_cache', {})
                            or {}
                        ).get('quality_score', 0.0) or 0.0
                    )
            except Exception as exc:
                logger.warning("Structured summary quality metrics skipped: %s", exc)
    except Exception as exc:
        continued = _continue_degraded_low_trust_malayalam(
            video,
            exc,
            source='run_audio_pipeline',
            quality=quality,
            source_language=source_language,
        )
        if not continued:
            raise

    processing_metrics = {
        'asr_engine': transcript_obj.asr_engine_used or transcript_obj.asr_engine,
        'asr_model': ((transcript_payload.get('metadata') or {}).get('selected_model') or (transcript_payload.get('metadata') or {}).get('actual_local_model_name') or ''),
        'language': source_language,
        'duration': float(video.duration or 0.0),
        'transcription_seconds': round(transcription_seconds, 4),
        'cleanup_seconds': round(cleanup_seconds, 4),
        'summary_seconds': round(summary_seconds, 4),
        'indexing_seconds': round(indexing_seconds, 4),
        'total_seconds': round(max(0.0, float((timezone.now() - pipeline_started).total_seconds())), 4),
        'transcript_quality_score': float(quality['quality_score'] or 0.0),
        'summary_quality_score': round(structured_quality_score, 4),
        'fallback_triggered': bool((transcript_payload.get('metadata') or {}).get('fallback_triggered', False)),
        'fallback_reason': str((transcript_payload.get('metadata') or {}).get('fallback_reason', '') or ''),
        'downstream_suppressed': bool(downstream_gate.get('suppress', False)),
        'downstream_suppression_reason': str(downstream_gate.get('reason', '') or ''),
        'trusted_visible_word_count': int(downstream_gate.get('trusted_visible_word_count', 0) or 0),
        'trusted_display_unit_count': int(downstream_gate.get('trusted_display_unit_count', 0) or 0),
        'summary_blocked_reason': str(quality.get('summary_blocked_reason', '') or ''),
        'chatbot_blocked_reason': str(quality.get('chatbot_blocked_reason', '') or ''),
        'lexical_trust_score': round(float((quality.get('qa_metrics', {}) or {}).get('lexical_trust_score', 0.0) or 0.0), 4),
        'overall_readability': round(float((quality.get('qa_metrics', {}) or {}).get('overall_readability', 0.0) or 0.0), 4),
        'source_language_fidelity_failed': bool((quality.get('qa_metrics', {}) or {}).get('source_language_fidelity_failed', False)),
        'transcript_fidelity_state': str((quality.get('qa_metrics', {}) or {}).get('transcript_fidelity_state', '') or ''),
        'low_evidence_malayalam': bool(downstream_gate.get('low_evidence_malayalam', False)),
        'summary_runtime': summary_runtime_rows,
        'embedding_model_requested': str(index_runtime.get('embedding_model_requested', '') or ''),
        'embedding_model_used': str(index_runtime.get('embedding_model_used', '') or ''),
        'embedding_model_fallback_used': bool(index_runtime.get('embedding_model_fallback_used', False)),
        'embedding_blocked_reason': str(index_runtime.get('embedding_blocked_reason', '') or ''),
        'embedding_runtime_error': str(index_runtime.get('embedding_runtime_error', '') or ''),
        'dev_sync_lite_mode': bool(lite_mode),
    }
    if source_language == 'ml':
        processing_metrics['malayalam_observability'] = _build_malayalam_observability(
            transcript_payload=transcript_payload,
            transcript_state=quality['state'],
            processing_metrics=processing_metrics,
        )
    if isinstance(transcript_obj.json_data, dict):
        transcript_obj.json_data['processing_metrics'] = processing_metrics
        transcript_obj.save(update_fields=['json_data'])

    if source_language == 'ml':
        malayalam_obs = processing_metrics.get('malayalam_observability', {})
        logger.info(
            "[MALAYALAM_OBSERVABILITY] video_id=%s first_pass_accepted=%s first_pass_accept_reason=%s "
            "retry_executed=%s retry_skipped_reason=%s transcript_state=%s total_asr_seconds=%.3f "
            "model_path=%s selected_model=%s resolved_model=%s fallback_triggered=%s fallback_reason=%s",
            str(video.id),
            bool(malayalam_obs.get('first_pass_accepted', False)),
            malayalam_obs.get('first_pass_accept_reason', '') or 'none',
            bool(malayalam_obs.get('retry_executed', False)),
            malayalam_obs.get('retry_skipped_reason', '') or 'none',
            malayalam_obs.get('transcript_state', '') or 'unknown',
            float(malayalam_obs.get('total_asr_seconds', 0.0) or 0.0),
            malayalam_obs.get('model_path', '') or 'unknown',
            malayalam_obs.get('selected_model', '') or 'unknown',
            malayalam_obs.get('resolved_model', '') or 'unknown',
            bool(malayalam_obs.get('fallback_triggered', False)),
            malayalam_obs.get('fallback_reason', '') or 'none',
        )
        logger.info(
            "[ML_FINAL_STATUS] transcript_state=%s video_status=completed continue_low_trust_degraded=%s",
            quality.get('state', ''),
            continue_low_trust_degraded,
        )
        logger.info(
            "[ML_FINAL_FIDELITY_STATUS] transcript_state=%s fidelity_state=%s fidelity_failed=%s summary_blocked_reason=%s chatbot_blocked_reason=%s",
            quality.get('state', ''),
            str((quality.get('qa_metrics', {}) or {}).get('transcript_fidelity_state', '') or 'none'),
            bool((quality.get('qa_metrics', {}) or {}).get('source_language_fidelity_failed', False)),
            str(processing_metrics.get('summary_blocked_reason', '') or 'none'),
            str(processing_metrics.get('chatbot_blocked_reason', '') or 'none'),
        )

    _update_video_stage(video, 'completed', 100, processed=True, error_message='')
    return {
        'status': 'success',
        'video_id': str(video.id),
        'transcript_id': transcript_obj.id,
    }


def _rebuild_highlights(video, transcript_obj):
    """Rebuild highlight segments so outputs stay in sync after reprocessing."""
    try:
        suppress, reason = _should_suppress_low_trust_malayalam_outputs(transcript_obj)
        if suppress:
            video.highlight_segments.all().delete()
            logger.info(
                "[CHAPTERS_SUPPRESSED_LOW_TRUST] source=highlight_rebuild video_id=%s transcript_id=%s reason=%s",
                getattr(video, 'id', ''),
                getattr(transcript_obj, 'id', ''),
                reason,
            )
            return
        highlights = detect_highlights(transcript_obj)
        # Prevent stale/duplicate highlights when transcript is regenerated.
        video.highlight_segments.all().delete()
        for highlight in highlights:
            highlight_data = _highlight_fields(highlight)
            HighlightSegment.objects.create(
                video=video,
                start_time=highlight_data['start_time'],
                end_time=highlight_data['end_time'],
                importance_score=highlight_data['importance_score'],
                reason=highlight_data['reason'],
                transcript_snippet=highlight_data['transcript_snippet']
            )
    except Exception as e:
        logger.warning(f"Highlight detection failed: {str(e)}")


def _rebuild_chatbot_index(video, transcript_payload):
    """Build chatbot index from the latest transcript segments."""
    try:
        from chatbot.rag_engine import ChatbotEngine
        chatbot = ChatbotEngine(str(video.id))
        segments = []
        if isinstance(transcript_payload, dict):
            payload_segments = transcript_payload.get('segments', [])
            canonical_segments = transcript_payload.get('canonical_segments', [])
            if isinstance(canonical_segments, list) and canonical_segments:
                segments = canonical_segments
            elif isinstance(payload_segments, list):
                segments = payload_segments
        if not segments:
            segments = _normalized_segments(transcript_payload)
        built = chatbot.build_from_transcript(segments)
        rag_engine = getattr(chatbot, 'rag_engine', None)
        return {
            'built': bool(built),
            'embedding_model_requested': getattr(rag_engine, 'embedding_model_requested', getattr(rag_engine, 'embedding_model', '')) if rag_engine else '',
            'embedding_model_used': getattr(rag_engine, 'embedding_model_used', getattr(rag_engine, 'embedding_model', '')) if rag_engine else '',
            'embedding_model_fallback_used': bool(getattr(rag_engine, 'embedding_model_fallback_used', False)) if rag_engine else False,
            'embedding_blocked_reason': getattr(rag_engine, 'index_error_reason', '') if rag_engine else '',
            'embedding_runtime_error': getattr(rag_engine, 'embedding_runtime_error', '') if rag_engine else '',
        }
    except Exception as e:
        logger.warning(f"RAG index build failed: {str(e)}")
        return {
            'built': False,
            'embedding_model_requested': '',
            'embedding_model_used': '',
            'embedding_model_fallback_used': False,
            'embedding_blocked_reason': 'rag_index_build_exception',
            'embedding_runtime_error': str(e),
        }


def process_video_transcription_sync(
    video_id,
    transcription_language: str = 'auto',
    output_language: str = 'auto',
    summary_language_mode: str = 'same_as_transcript'
):
    """
    Synchronous version of video transcription for direct calls.
    Optimized: runs summaries and embeddings in parallel.
    """
    try:
        video, claimed = _claim_video_processing(video_id)
        if video is None:
            raise Video.DoesNotExist()
        if not claimed:
            logger.info("Skipping duplicate video processing run for %s; status=%s", video_id, video.status)
            return {'status': 'already_processing', 'message': 'Video is already being processed', 'video_id': str(video_id)}

        try:
            _update_video_stage(video, 'extracting_audio', 12, error_message='')
            staged_video_dir = ''
            audio_path = ''
            detected_duration = 0.0
            source_video_path, staged_video_dir = _stage_uploaded_video_to_local_temp(video, purpose='transcription_sync')
            try:
                audio_path = extract_audio(source_video_path)
                detected_duration = float(get_video_duration(audio_path) or 0.0)
                if detected_duration > 0:
                    video.duration = detected_duration
                    video.save(update_fields=['duration', 'updated_at'])
                _ensure_sync_mode_duration_allowed(
                    video,
                    float(video.duration or detected_duration or 0.0),
                    source_type='upload',
                )
                prepared_audio_path, prep_meta = _prepare_audio_for_pipeline(
                    audio_path,
                    transcription_language=transcription_language,
                )
                try:
                    result = _run_audio_pipeline(
                        video,
                        chunks=prep_meta.get('chunks', []),
                        audio_path=prepared_audio_path,
                        source_type='file',
                        transcription_language=transcription_language,
                        output_language=output_language,
                        summary_language_mode=summary_language_mode,
                    )
                finally:
                    shutil.rmtree(prep_meta.get('temp_dir', ''), ignore_errors=True)
                    if audio_path and os.path.exists(audio_path):
                        os.remove(audio_path)
            finally:
                _cleanup_local_staged_video(staged_video_dir, video_id=str(video_id), purpose='transcription_sync')
            logger.info(f"Video processing completed for {video_id}")
            return result
            
        except Exception as e:
            try:
                video = Video.objects.get(id=video_id)
                continued = _continue_degraded_low_trust_malayalam(video, e, source='file_sync_wrapper')
                if continued:
                    return continued
            except Exception:
                pass
            logger.error(f"Video processing failed: {str(e)}")
            try:
                video = Video.objects.get(id=video_id)
                _fail_video_with_logged_status(video, e, source='file_sync_wrapper')
            except:
                pass
            return {'status': 'error', 'message': str(e), 'video_id': str(video_id)}
    except Exception as e:
        logger.error(f"Video processing failed before setup for {video_id}: {str(e)}")
        return {'status': 'error', 'message': str(e), 'video_id': str(video_id)}


@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def process_video_transcription(
    self,
    video_id,
    transcription_language: str = 'auto',
    output_language: str = 'auto',
    summary_language_mode: str = 'same_as_transcript'
):
    """
    Main task for processing video transcription.
    Steps:
    1. Extract audio from video
    2. Transcribe audio using Whisper
    3. Update video status
    """
    task = None
    staged_video_dir = ''
    audio_path = ''
    try:
        video, claimed = _claim_video_processing(video_id)
        if video is None:
            raise Video.DoesNotExist()
        if not claimed:
            logger.info("Skipping duplicate video processing task for %s; status=%s", video_id, video.status)
            return {'status': 'already_processing', 'message': 'Video is already being processed', 'video_id': str(video_id)}
        _update_video_stage(video, 'extracting_audio', 12, error_message='')

        task = ProcessingTask.objects.filter(task_id=self.request.id).first()
        if task:
            task.mark_started()
            task.progress = 12
            task.message = 'Extracting audio from video'
            task.save(update_fields=['status', 'progress', 'message'])

        prep_meta = None
        try:
            source_video_path, staged_video_dir = _stage_uploaded_video_to_local_temp(video, purpose='transcription_async')
            audio_path = extract_audio(source_video_path)

            if task:
                task.progress = 35
                task.message = 'Running transcript, summary, and chatbot stages'
                task.save(update_fields=['progress', 'message'])

            prepared_audio_path, prep_meta = _prepare_audio_for_pipeline(
                audio_path,
                transcription_language=transcription_language,
            )
            try:
                result = _run_audio_pipeline(
                    video,
                    chunks=prep_meta.get('chunks', []),
                    audio_path=prepared_audio_path,
                    source_type='file',
                    transcription_language=transcription_language,
                    output_language=output_language,
                    summary_language_mode=summary_language_mode,
                )
            finally:
                if prep_meta:
                    shutil.rmtree(prep_meta.get('temp_dir', ''), ignore_errors=True)
                if audio_path and os.path.exists(audio_path):
                    os.remove(audio_path)
        finally:
            _cleanup_local_staged_video(staged_video_dir, video_id=str(video_id), purpose='transcription_async')

        if task:
            if result.get('status') == 'success':
                task.mark_completed()
            else:
                task.mark_failed(result.get('warning', 'Processing completed with warnings'))

        return result
        
    except Video.DoesNotExist:
        logger.error(f"Video {video_id} not found")
        return {'status': 'error', 'message': 'Video not found'}
    
    except Exception as e:
        try:
            video = Video.objects.get(id=video_id)
            continued = _continue_degraded_low_trust_malayalam(video, e, source='file_wrapper')
            if continued:
                if task:
                    task.mark_completed()
                return continued
        except Exception:
            pass
        logger.error(f"Transcription failed for video {video_id}: {str(e)}")
        
        # Update video status
        try:
            video = Video.objects.get(id=video_id)
            _fail_video_with_logged_status(video, e, source='file_wrapper')
            
            # Update task status
            if task:
                task.mark_failed(str(e))
                
        except Exception:
            pass
        
        # Retry if retries available
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)
        
        return {'status': 'error', 'message': str(e)}


@shared_task(bind=True, max_retries=2, default_retry_delay=120)
def generate_summary(
    self,
    video_id,
    summary_type,
    max_length=None,
    min_length=None,
    output_language: str = 'auto',
    summary_language_mode: str = 'same_as_transcript'
):
    """
    Generate summary for a video transcript.
    
    summary_type: 'full', 'bullet', 'short', 'timestamps'
    """
    try:
        video = Video.objects.get(id=video_id)
        
        # Get latest transcript
        transcript = video.transcripts.order_by('-created_at').first()
        if not transcript:
            raise ValueError("No transcript found for this video")
        
        # Update task status
        task = ProcessingTask.objects.filter(task_id=self.request.id).first()
        if task:
            task.mark_started()
            task.message = f'Generating {summary_type} summary'
            task.save(update_fields=['status', 'message'])
        
        # Generate summary
        resolved_output_language = resolve_output_language(
            output_language,
            transcript.language
        )
        summary_result = summarize_text(
            transcript.transcript_original_text or transcript.full_text,
            summary_type=summary_type,
            max_length=max_length,
            min_length=min_length,
            output_language=resolved_output_language,
            source_language=transcript.transcript_language or transcript.language,
            canonical_text=transcript.transcript_canonical_text or '',
            canonical_language=transcript.canonical_language or 'en',
            summary_language_mode=summary_language_mode,
            has_fidelity_gaps=bool((transcript.json_data or {}).get("has_fidelity_gaps", False)) if isinstance(getattr(transcript, "json_data", None), dict) else False,
        )
        summary_data = _summary_fields(summary_result, f'{summary_type.capitalize()} Summary')
        
        # Save summary
        with transaction.atomic():
            summary, _ = Summary.objects.update_or_create(
                video=video,
                summary_type=summary_type,
                defaults={
                    'title': summary_data['title'],
                    'content': summary_data['content'],
                    'key_topics': summary_data['key_topics'],
                    'summary_language': summary_data['summary_language'],
                    'summary_source_language': summary_data['summary_source_language'],
                    'translation_used': summary_data['translation_used'],
                    'model_used': summary_data['model_used'],
                    'generation_time': summary_data['generation_time']
                }
            )
        
        if task:
            task.mark_completed()
        
        logger.info(f"Summary generated for video {video_id}, type: {summary_type}")
        return {
            'status': 'completed', 
            'video_id': str(video_id), 
            'summary_id': str(summary.id)
        }
        
    except Video.DoesNotExist:
        logger.error(f"Video {video_id} not found")
        return {'status': 'error', 'message': 'Video not found'}
    
    except Exception as e:
        logger.error(f"Summary generation failed for video {video_id}: {str(e)}")
        
        try:
            if task:
                task.mark_failed(str(e))
        except Exception:
            pass
        
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)
        
        return {'status': 'error', 'message': str(e)}


@shared_task(bind=True, max_retries=2, default_retry_delay=120)
def detect_video_highlights(self, video_id):
    """
    Detect highlight segments in video using transcript analysis.
    """
    try:
        video = Video.objects.get(id=video_id)
        
        # Get transcript
        transcript = video.transcripts.order_by('-created_at').first()
        if not transcript:
            raise ValueError("No transcript found for this video")
        
        # Update task status
        task = ProcessingTask.objects.filter(task_id=self.request.id).first()
        if task:
            task.mark_started()
            task.message = 'Detecting highlight segments'
            task.save(update_fields=['status', 'message'])
        
        # Detect highlights
        highlights = detect_highlights(transcript)
        
        # Save highlights
        with transaction.atomic():
            for highlight in highlights:
                HighlightSegment.objects.create(
                    video=video,
                    start_time=highlight['start_time'],
                    end_time=highlight['end_time'],
                    importance_score=highlight['importance_score'],
                    reason=highlight['reason'],
                    transcript_snippet=highlight.get('transcript_snippet', '')
                )
        
        if task:
            task.mark_completed()
        
        logger.info(f"Highlights detected for video {video_id}")
        return {
            'status': 'completed',
            'video_id': str(video_id),
            'highlights_count': len(highlights)
        }
        
    except Exception as e:
        logger.error(f"Highlight detection failed for video {video_id}: {str(e)}")
        
        try:
            if task:
                task.mark_failed(str(e))
        except Exception:
            pass
        
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)
        
        return {'status': 'error', 'message': str(e)}


@shared_task(bind=True, max_retries=2, default_retry_delay=180)
def generate_short_video(
    self, video_id, max_duration=60.0, style='default',
    include_music=False, caption_style='default', font_size=24
):
    """
    Generate a short video from highlights.
    """
    try:
        video = Video.objects.get(id=video_id)
        
        # Get highlights sorted by importance
        highlights = video.highlight_segments.filter(
            used_in_short=False
        ).order_by('-importance_score')[:10]
        
        if not highlights.exists():
            raise ValueError("No highlights found for this video")
        
        # Update task status
        task = ProcessingTask.objects.filter(task_id=self.request.id).first()
        if task:
            task.mark_started()
            task.message = 'Creating short video'
            task.save(update_fields=['status', 'message'])
        
        # Calculate total duration and select segments
        selected_segments = []
        total_duration = 0
        
        for highlight in highlights:
            segment_duration = highlight.end_time - highlight.start_time
            if total_duration + segment_duration <= max_duration:
                selected_segments.append(highlight)
                total_duration += segment_duration
        
        if not selected_segments:
            raise ValueError("No segments fit within max_duration")
        
        # Update task progress
        if task:
            task.progress = 30
            task.message = 'Processing video segments'
            task.save(update_fields=['progress', 'message'])
        
        # Create short video
        source_video_path, staged_video_dir = _stage_uploaded_video_to_local_temp(video, purpose='short_video')
        try:
            short_video_path = create_short_video(
                source_video_path,
                selected_segments,
                style=style,
                caption_style=caption_style,
                font_size=font_size
            )
        finally:
            _cleanup_local_staged_video(staged_video_dir, video_id=str(video_id), purpose='short_video')
        
        if task:
            task.progress = 80
            task.message = 'Saving short video'
            task.save(update_fields=['progress', 'message'])
        
        # Save short video record
        from django.core.files import File
        
        with open(short_video_path, 'rb') as f:
            short_video = ShortVideo.objects.create(
                video=video,
                file=File(f, name=os.path.basename(short_video_path)),
                duration=total_duration,
                style=style,
                include_music=include_music,
                caption_style=caption_style,
                font_size=font_size,
                status='completed'
            )
        
        # Mark highlights as used
        for highlight in selected_segments:
            highlight.used_in_short = True
            highlight.save(update_fields=['used_in_short'])
        
        if task:
            task.mark_completed()
        
        logger.info(f"Short video generated for video {video_id}")
        return {
            'status': 'completed',
            'video_id': str(video_id),
            'short_video_id': str(short_video.id),
            'duration': total_duration
        }
        
    except Exception as e:
        logger.error(f"Short video generation failed for video {video_id}: {str(e)}")
        
        try:
            if task:
                task.mark_failed(str(e))
        except Exception:
            pass
        
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)
        
        return {'status': 'error', 'message': str(e)}


@shared_task
def cleanup_old_files():
    """
    Cleanup task to remove old files and database records.
    Run daily via Celery beat.
    """
    from datetime import timedelta
    from django.core.files.storage import default_storage
    
    # Delete files older than 7 days
    cutoff_date = timezone.now() - timedelta(days=7)
    
    old_videos = Video.objects.filter(created_at__lt=cutoff_date)
    
    for video in old_videos:
        try:
            # Delete files
            if video.original_file:
                video.original_file.delete(save=False)
            
            # Delete related files
            for transcript in video.transcripts.all():
                # Cleanup if there are file references
                pass
            
            for short in video.short_videos.all():
                if short.file:
                    short.file.delete(save=False)
                if short.thumbnail:
                    short.thumbnail.delete(save=False)
            
            logger.info(f"Cleaned up video {video.id}")
        except Exception as e:
            logger.error(f"Cleanup failed for video {video.id}: {str(e)}")
    
    # Delete old task records
    ProcessingTask.objects.filter(
        created_at__lt=cutoff_date,
        status='completed'
    ).delete()
    
    return {'status': 'completed', 'cleaned_videos': old_videos.count()}


@shared_task(bind=True, max_retries=2, default_retry_delay=120)
def build_video_chatbot_index(self, video_id):
    """
    Build RAG index for video chatbot from transcript segments.
    This enables the chatbot to answer questions about the video content.
    """
    try:
        from chatbot.rag_engine import VideoRAGEngine
        
        video = Video.objects.get(id=video_id)
        
        # Get latest transcript
        transcript = video.transcripts.order_by('-created_at').first()
        if not transcript:
            raise ValueError("No transcript found for this video")
        
        # Get transcript segments from json_data (prefer canonical segments).
        segments = []
        if isinstance(transcript.json_data, dict):
            canonical_segments = transcript.json_data.get('canonical_segments', [])
            if isinstance(canonical_segments, list) and canonical_segments:
                segments = canonical_segments
            else:
                segments = transcript.json_data.get('segments', [])
        elif isinstance(transcript.json_data, list):
            segments = transcript.json_data

        if not isinstance(segments, list):
            segments = []

        if not segments:
            # If json_data is not available, create segments from full_text
            # Split by sentences
            import re
            text = transcript.transcript_canonical_en_text or transcript.transcript_canonical_text or transcript.full_text
            sentences = re.split(r'(?<=[.!?])\s+', text)
            segments = []
            start_time = 0
            for i, sent in enumerate(sentences):
                if sent.strip():
                    # Estimate 5 seconds per sentence for timestamps
                    duration = len(sent.split()) * 0.5  # ~0.5s per word
                    segments.append({
                        'id': i,
                        'start': start_time,
                        'end': start_time + duration,
                        'text': sent.strip()
                    })
                    start_time += duration
        
        # Build RAG index
        rag_engine = VideoRAGEngine(str(video_id))
        success = rag_engine.build_index(segments)
        
        if success:
            logger.info(f"RAG index built for video {video_id}")
            return {'status': 'completed', 'video_id': str(video_id), 'segments_count': len(segments)}
        else:
            raise ValueError("Failed to build RAG index")
            
    except Video.DoesNotExist:
        logger.error(f"Video {video_id} not found")
        return {'status': 'error', 'message': 'Video not found'}
    
    except Exception as e:
        logger.error(f"RAG index build failed for video {video_id}: {str(e)}")
        
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)
        
        return {'status': 'error', 'message': str(e)}


@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def process_youtube_video(
    self,
    video_id,
    transcription_language: str = 'auto',
    output_language: str = 'auto',
    summary_language_mode: str = 'same_as_transcript'
):
    """
    Process a YouTube video: download audio and transcribe.
    """
    try:
        video, claimed = _claim_video_processing(video_id)
        if video is None:
            raise Video.DoesNotExist()
        if not claimed:
            logger.info("Skipping duplicate YouTube video processing task for %s; status=%s", video_id, video.status)
            return {'status': 'already_processing', 'message': 'Video is already being processed', 'video_id': str(video_id)}
        
        if not video.youtube_url:
            raise ValueError("No YouTube URL associated with this video")

        # Create temp directory for download
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, f'{uuid.uuid4()}.wav')
        
        try:
            logger.info(f"Downloading YouTube video: {video.youtube_url}")
            _download_youtube_audio(video.youtube_url, audio_path)
            
            # Validate download
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 50_000:
                raise Exception("Downloaded audio is missing or too small")

            _update_video_stage(video, 'extracting_audio', 20, error_message='')
            video.duration = get_video_duration(audio_path)
            video.save(update_fields=['duration', 'updated_at'])
            _ensure_sync_mode_duration_allowed(
                video,
                float(video.duration or 0.0),
                source_type='youtube',
            )

            prepared_audio_path, prep_meta = _prepare_audio_for_pipeline(
                audio_path,
                transcription_language=transcription_language,
            )
            try:
                result = _run_audio_pipeline(
                    video,
                    chunks=prep_meta.get('chunks', []),
                    audio_path=prepared_audio_path,
                    source_type='youtube',
                    transcription_language=transcription_language,
                    output_language=output_language,
                    summary_language_mode=summary_language_mode,
                )
            finally:
                _retain_malayalam_debug_audio(
                    video_id=video_id,
                    transcription_language=transcription_language,
                    prep_meta=prep_meta,
                    result=locals().get('result'),
                )
                shutil.rmtree(prep_meta.get('temp_dir', ''), ignore_errors=True)
            if result.get('status') == 'success':
                logger.info(f"YouTube video {video_id} processed successfully")
            else:
                logger.warning(
                    "YouTube video %s finished without completion: status=%s warning=%s",
                    video_id,
                    result.get('status'),
                    result.get('warning') or result.get('message', ''),
                )
            return result
            
        finally:
            # Cleanup temp files
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp files: {str(e)}")
    
    except Video.DoesNotExist:
        logger.error(f"Video {video_id} not found")
        return {'status': 'error', 'message': 'Video not found'}
    
    except Exception as e:
        try:
            video = Video.objects.get(id=video_id)
            continued = _continue_degraded_low_trust_malayalam(video, e, source='youtube_wrapper')
            if continued:
                return continued
        except Exception:
            pass
        logger.error(f"YouTube video processing failed for {video_id}: {str(e)}")
        
        try:
            video = Video.objects.get(id=video_id)
            _fail_video_with_logged_status(video, e, source='youtube_wrapper')
        except:
            pass
        
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)
        
        return {'status': 'error', 'message': str(e)}


# Synchronous version for YouTube processing (without Celery)
def process_youtube_video_sync(
    video_id,
    transcription_language: str = 'auto',
    output_language: str = 'auto',
    summary_language_mode: str = 'same_as_transcript'
):
    """
    Synchronous version of YouTube video processing.
    Used when Redis/Celery is not available.
    """
    try:
        video, claimed = _claim_video_processing(video_id)
        if video is None:
            raise Video.DoesNotExist()
        if not claimed:
            logger.info("Skipping duplicate sync YouTube processing run for %s; status=%s", video_id, video.status)
            return {'status': 'already_processing', 'message': 'Video is already being processed', 'video_id': str(video_id)}
        
        if not video.youtube_url:
            raise ValueError("No YouTube URL associated with this video")
        
        # Create temp directory for download
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, f'{uuid.uuid4()}.wav')
        
        try:
            logger.info(f"Downloading YouTube video: {video.youtube_url}")
            _download_youtube_audio(video.youtube_url, audio_path)
            
            # Validate download
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 50_000:
                raise Exception("Downloaded audio is missing or too small")

            _update_video_stage(video, 'extracting_audio', 20, error_message='')
            video.duration = get_video_duration(audio_path)
            video.save(update_fields=['duration', 'updated_at'])

            prepared_audio_path, prep_meta = _prepare_audio_for_pipeline(
                audio_path,
                transcription_language=transcription_language,
            )
            try:
                result = _run_audio_pipeline(
                    video,
                    chunks=prep_meta.get('chunks', []),
                    audio_path=prepared_audio_path,
                    source_type='youtube',
                    transcription_language=transcription_language,
                    output_language=output_language,
                    summary_language_mode=summary_language_mode,
                )
            finally:
                _retain_malayalam_debug_audio(
                    video_id=video_id,
                    transcription_language=transcription_language,
                    prep_meta=prep_meta,
                    result=locals().get('result'),
                )
                shutil.rmtree(prep_meta.get('temp_dir', ''), ignore_errors=True)
            if result.get('status') == 'success':
                logger.info(f"YouTube video {video_id} processed successfully")
            else:
                logger.warning(
                    "YouTube video %s finished without completion: status=%s warning=%s",
                    video_id,
                    result.get('status'),
                    result.get('warning') or result.get('message', ''),
                )
            return result
            
        finally:
            # Cleanup temp files
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp files: {str(e)}")
    
    except Video.DoesNotExist:
        logger.error(f"Video {video_id} not found")
        return {'status': 'error', 'message': 'Video not found'}
    except IntegrityError as e:
        logger.warning(f"YouTube processing cancelled for {video_id}: {str(e)}")
        return {'status': 'cancelled', 'message': 'Video data changed during processing'}
    except IntegrityError as e:
        logger.warning(f"YouTube processing cancelled for {video_id}: {str(e)}")
        return {'status': 'cancelled', 'message': 'Video data changed during processing'}
    
    except Exception as e:
        try:
            video = Video.objects.get(id=video_id)
            continued = _continue_degraded_low_trust_malayalam(video, e, source='youtube_sync_wrapper')
            if continued:
                return continued
        except Exception:
            pass
        logger.error(f"YouTube video processing failed for {video_id}: {str(e)}")
        
        try:
            video = Video.objects.get(id=video_id)
            _fail_video_with_logged_status(video, e, source='youtube_sync_wrapper')
        except:
            pass
        
        return {'status': 'error', 'message': str(e)}
