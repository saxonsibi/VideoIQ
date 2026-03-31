"""
API Views for chatbot app
"""

import logging
import re
import uuid
from django.conf import settings
from django.db import transaction
from django.utils import timezone
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import ChatSession, ChatMessage, VideoIndex
from .serializers import (
    ChatSessionSerializer, ChatMessageSerializer, ChatMessageCreateSerializer,
    ChatResponseSerializer, SuggestedQuestionsSerializer, VideoIndexSerializer
)
from .rag_engine import ChatbotEngine
from videos.utils import normalize_language_code
from videos.language import detect_text_language
from videos.translation import (
    translate_text,
    build_safe_english_view_text,
    build_english_view_source_hash,
    evaluate_english_view_policy,
    build_english_view_cache_entry,
)
from videos.processing_metadata import build_processing_metadata

logger = logging.getLogger(__name__)


def _render_transcript_only_chat_disabled_response(*, session_id=None, transcript=None, video=None, output_language='en'):
    english_view = {
        'english_view_text': '',
        'english_view_available': False,
        'translation_state': 'blocked',
        'translation_warning': '',
        'translation_blocked_reason': 'render_transcript_only_mode',
    }
    payload = {
        'answer': '',
        'sources': [],
        'session_id': str(session_id) if session_id else None,
        'user_language': output_language,
        'output_language': output_language,
        'retrieval_language': 'en',
        'processing_metadata': build_processing_metadata(video, transcript) if video else {},
        'error': 'Chatbot is disabled on the live demo to keep the server responsive after transcription.',
        'english_view_answer': '',
        'english_view_available': False,
        'chatbot_answer_language': output_language,
        'chatbot_english_view_available': False,
        'chatbot_translation_state': 'blocked',
        'chatbot_translation_warning': '',
        'chatbot_translation_blocked_reason': 'render_transcript_only_mode',
        'chatbot_blocked_reason': 'render_transcript_only_mode',
    }
    payload.update(english_view)
    return payload


def _pack_chat_sources_with_english_view_cache(sources, cache_entry=None):
    payload = {
        "sources": list(sources or []),
    }
    if isinstance(cache_entry, dict) and cache_entry:
        payload["_english_view_cache"] = dict(cache_entry)
    return payload


def _build_chat_english_view(answer_text: str, *, answer_language: str, transcript, translated_english_hint: str = "") -> dict:
    json_data = transcript.json_data if isinstance(getattr(transcript, "json_data", None), dict) else {}
    transcript_state = str(json_data.get("transcript_state", "") or "").strip().lower()
    low_evidence_malayalam = bool(json_data.get("low_evidence_malayalam", False))
    source_language_fidelity_failed = bool(json_data.get("source_language_fidelity_failed", False))
    source_language = normalize_language_code(answer_language, default="en", allow_auto=False)
    transcript_language = normalize_language_code(getattr(transcript, "transcript_language", "") or getattr(transcript, "language", ""), default="en", allow_auto=False)
    policy = evaluate_english_view_policy(
        content_kind="chat",
        source_language=source_language,
        source_state=transcript_state,
        has_grounded_text=bool(str(answer_text or "").strip()),
        low_evidence_malayalam=bool(transcript_language == "ml" and transcript_state == "degraded" and (low_evidence_malayalam or source_language_fidelity_failed)),
        degraded_low_evidence_reason="source_language_fidelity_failed" if source_language_fidelity_failed else "degraded_safe_translation_blocked",
    )
    if transcript_language == "ml" and source_language_fidelity_failed:
        policy = {
            **policy,
            "allow_translation": False,
            "blocked_reason": "source_language_fidelity_failed",
            "translation_state": "blocked",
            "policy_reason": "malayalam_source_fidelity_failed",
        }
    logger.info(
        "[CHAT_EN_VIEW_REQUEST] transcript_id=%s answer_language=%s state=%s allow=%s blocked_reason=%s",
        getattr(transcript, "id", ""),
        answer_language,
        transcript_state,
        policy["allow_translation"],
        policy["blocked_reason"],
    )
    payload = build_safe_english_view_text(
        answer_text,
        answer_language,
        allow_translation=bool(policy["allow_translation"]),
        blocked_reason=str(policy["blocked_reason"] or ""),
        warning=str(json_data.get("transcript_warning_message", "") or ""),
        preserve_format=True,
        translated_text=translated_english_hint,
    )
    if payload.get("english_view_available"):
        logger.info(
            "[CHAT_EN_VIEW_RESULT] transcript_id=%s mode=%s",
            getattr(transcript, "id", ""),
            payload.get("translation_state", ""),
        )
    else:
        logger.info(
            "[CHAT_EN_VIEW_BLOCKED] transcript_id=%s reason=%s",
            getattr(transcript, "id", ""),
            payload.get("translation_blocked_reason", ""),
        )
    payload["_english_view_policy_mode"] = str(policy.get("translation_state", "") or "")
    payload["_english_view_policy_reason"] = str(policy.get("policy_reason", "") or "")
    return payload


def _normalize_transcript_segments(transcript):
    """Ensure chatbot index always receives segment dictionaries with text/start/end."""
    json_data = transcript.json_data
    if isinstance(json_data, dict):
        segments = json_data.get('canonical_segments') or json_data.get('segments', [])
        if isinstance(segments, list):
            return segments
    elif isinstance(json_data, list):
        return json_data

    # Fallback: derive simple segments from transcript text
    text = transcript.transcript_canonical_en_text or transcript.transcript_canonical_text or transcript.full_text or ''
    if not text:
        return []

    import re
    chunks = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    segments = []
    for i, seg_text in enumerate(chunks):
        segments.append({
            'id': i,
            'start': i * 5,
            'end': (i + 1) * 5,
            'text': seg_text
        })
    return segments


def _format_timestamp_label(seconds: float) -> str:
    total_seconds = max(0, int(float(seconds or 0)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _parse_answer_sections(answer_text: str):
    """Split formatted chatbot output into explanation and key points."""
    normalized = str(answer_text or '').replace('\r\n', '\n').replace('\r', '\n').strip()
    if not normalized:
        return '', []

    answer_match = re.search(r'Answer:\s*([\s\S]*?)(?:\n\s*Key Points:|$)', normalized, flags=re.IGNORECASE)
    key_match = re.search(r'Key Points:\s*([\s\S]*)$', normalized, flags=re.IGNORECASE)

    explanation = (answer_match.group(1) if answer_match else normalized).strip()
    explanation = re.sub(r'\s+', ' ', explanation).strip()

    key_points = []
    if key_match:
        for line in key_match.group(1).splitlines():
            cleaned = re.sub(r'^[\u2022*\-\s]+', '', line).strip()
            if cleaned:
                key_points.append(cleaned)

    return explanation, key_points


def _clean_voice_sentence(text: str) -> str:
    sentence = str(text or '').strip()
    if not sentence:
        return ''
    sentence = re.sub(r'^(?:Answer|Key Points?)\s*:?\s*', '', sentence, flags=re.IGNORECASE)
    sentence = re.sub(r'^[\u2022*\-\s]+', '', sentence).strip()
    sentence = re.sub(r'^\s*(?:so|well|okay|ok|right)\b[:,\-\s]*', '', sentence, flags=re.IGNORECASE)
    sentence = re.sub(r'^\s*(?:he|she|they)\s+say[s]?\s+', '', sentence, flags=re.IGNORECASE)
    sentence = re.sub(r'\s+', ' ', sentence).strip(' .,;:-')
    if sentence and sentence[-1] not in '.!?':
        sentence += '.'
    return sentence


def _select_voice_speaker(sources):
    counts = {}
    for source in sources or []:
        speaker = str(source.get('speaker') or '').strip()
        if not speaker:
            continue
        counts[speaker] = counts.get(speaker, 0) + 1
    if not counts:
        return ''
    return max(counts, key=counts.get)


def _limit_voice_narration(text: str, max_words: int = 60) -> str:
    words = str(text or '').split()
    if len(words) <= max_words:
        return str(text or '').strip()
    clipped = ' '.join(words[:max_words]).rstrip(' ,;:')
    if clipped and clipped[-1] not in '.!?':
        clipped += '.'
    return clipped


def _build_voice_narration(answer_text: str, sources) -> str:
    """Convert formatted chat output into a short natural TTS narration."""
    explanation, key_points = _parse_answer_sections(answer_text)
    explanation = _clean_voice_sentence(explanation)
    takeaway = _clean_voice_sentence(key_points[0]) if key_points else ''
    speaker = _select_voice_speaker(sources)

    if speaker:
        lead = f"Here's what {speaker} says in the interview."
    else:
        lead = "Here's what the video says."

    parts = [lead]
    if explanation:
        parts.append(explanation)
    if takeaway:
        parts.append(f"Key takeaway: {takeaway}")

    narration = ' '.join(part.strip() for part in parts if part.strip())
    return _limit_voice_narration(narration, max_words=60)


def _select_tts_language(preferred: str) -> str:
    """Use the requested language when gTTS supports it, otherwise fall back to English."""
    language = normalize_language_code(preferred, default='en', allow_auto=False)
    try:
        from gtts.lang import tts_langs
        supported = tts_langs()
        if language in supported:
            return language
    except Exception:
        logger.debug("Unable to inspect gTTS language registry; defaulting to English.")
    return 'en'


def _get_context_segments_around_timestamp(segments, timestamp: float, window: float = None):
    """Return a local moment window around a timestamp, then fall back to nearest segments."""
    try:
        target = float(timestamp)
    except (TypeError, ValueError):
        return []

    default_before = float(getattr(settings, 'CHAT_MOMENT_CONTEXT_BEFORE_SECONDS', 20))
    default_after = float(getattr(settings, 'CHAT_MOMENT_CONTEXT_AFTER_SECONDS', 40))
    max_chunks = max(1, int(getattr(settings, 'CHAT_MOMENT_MAX_CHUNKS', 4)))

    if window is not None:
        try:
            radius = max(1.0, float(window))
        except (TypeError, ValueError):
            radius = default_after
        before_seconds = radius
        after_seconds = radius
    else:
        before_seconds = max(1.0, default_before)
        after_seconds = max(1.0, default_after)

    nearby = []
    for segment in segments or []:
        start = float(segment.get('start', 0) or 0)
        end = float(segment.get('end', start) or start)
        overlaps_window = (start <= target + after_seconds) and (end >= target - before_seconds)
        center_distance = abs(((start + end) / 2.0) - target)
        if overlaps_window or center_distance <= max(before_seconds, after_seconds):
            nearby.append(segment)

    if nearby:
        ranked_nearby = sorted(
            nearby,
            key=lambda seg: (
                abs((((float(seg.get('start', 0) or 0) + float(seg.get('end', 0) or seg.get('start', 0) or 0)) / 2.0) - target)),
                float(seg.get('start', 0) or 0),
            )
        )
        return sorted(ranked_nearby[:max_chunks], key=lambda seg: float(seg.get('start', 0) or 0))

    ranked = sorted(
        segments or [],
        key=lambda seg: abs(float(seg.get('start', 0) or 0) - target)
    )
    return sorted(ranked[:max_chunks], key=lambda seg: float(seg.get('start', 0) or 0))


class ChatSessionViewSet(viewsets.ModelViewSet):
    """ViewSet for ChatSession CRUD operations."""
    queryset = ChatSession.objects.all()
    serializer_class = ChatSessionSerializer
    
    def get_queryset(self):
        """Filter by video_id if provided."""
        queryset = super().get_queryset()
        video_id = self.request.query_params.get('video_id')
        if video_id:
            queryset = queryset.filter(video_id=video_id)
        return queryset
    
    @action(detail=True, methods=['get'])
    def messages(self, request, pk=None):
        """Get all messages in a session."""
        session = self.get_object()
        messages = session.messages.all()
        serializer = ChatMessageSerializer(messages, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['delete'])
    def clear(self, request, pk=None):
        """Clear all messages in a session."""
        session = self.get_object()
        session.messages.all().delete()
        return Response({'status': 'cleared'})


class ChatbotView(APIView):
    """
    Main chatbot API endpoint for asking questions about videos.
    """
    
    def post(self, request):
        """Handle chatbot question."""
        serializer = ChatMessageCreateSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        video_id = serializer.validated_data['video_id']
        message = serializer.validated_data['message']
        session_id = serializer.validated_data.get('session_id')
        strict_mode = serializer.validated_data.get('strict_mode', False)
        context_timestamp = serializer.validated_data.get('context_timestamp')
        context_window_seconds = serializer.validated_data.get('context_window_seconds')
        requested_english_view = serializer.validated_data.get('english_view', False)
        requested_output_language = normalize_language_code(
            serializer.validated_data.get('output_language') or serializer.validated_data.get('response_language'),
            default='auto',
            allow_auto=True
        )
        user_language, user_confidence, _, _detector = detect_text_language(message, default='en')
        
        try:
            # Get or create session
            if session_id:
                session = ChatSession.objects.filter(id=session_id).first()
                if session and str(session.video_id) != str(video_id):
                    logger.warning(
                        "stale_state_blocked=True session_id=%s session_video_id=%s requested_video_id=%s reason=session_video_mismatch",
                        session_id,
                        session.video_id,
                        video_id,
                    )
                    session = ChatSession.objects.create(
                        video_id=video_id,
                        title=f"Chat about video {video_id}"
                    )
                elif not session:
                    session = ChatSession.objects.create(
                        id=session_id,
                        video_id=video_id,
                        title=f"Chat about video {video_id}"
                    )
            else:
                session = ChatSession.objects.create(
                    video_id=video_id,
                    title=f"Chat about video {video_id}"
                )
            
            # Save user message with default retrieval language
            # When English View is requested, it will be overridden later
            user_msg = ChatMessage.objects.create(
                session=session,
                sender='user',
                message=message,
                user_language=user_language,
                output_language=requested_output_language if requested_output_language != 'auto' else user_language,
                retrieval_language='en',  # Default, will be updated after transcript is fetched
            )
            
            # Get transcript and initialize chatbot
            from videos.models import Video, Transcript
            
            try:
                video = Video.objects.get(id=video_id)
                transcript = Transcript.objects.filter(video=video).order_by('-created_at').first()
                
                if not transcript:
                    return Response(
                        {
                            'error': 'No transcript available for this video',
                            'session_id': str(session.id)
                        },
                        status=status.HTTP_400_BAD_REQUEST
                    )
                transcript_language = normalize_language_code(
                    transcript.transcript_language or transcript.language,
                    default='en',
                    allow_auto=False
                )
                retrieval_language = normalize_language_code(
                    transcript.canonical_language or 'en',
                    default='en',
                    allow_auto=False
                )
                output_language = requested_output_language
                if output_language == 'auto':
                    output_language = transcript_language
                # When English View is requested, force retrieval and output to English
                if requested_english_view:
                    retrieval_language = 'en'
                    output_language = 'en'
                user_msg.output_language = output_language
                user_msg.retrieval_language = retrieval_language
                user_msg.save(update_fields=['output_language', 'retrieval_language'])

                transcript_state = ''
                transcript_fidelity_failed = False
                if isinstance(transcript.json_data, dict):
                    transcript_state = str(transcript.json_data.get('transcript_state', '') or '').strip()
                    transcript_fidelity_failed = bool(
                        transcript.json_data.get('source_language_fidelity_failed', False)
                        or str(transcript.json_data.get('transcript_state', '') or '').strip().lower() == 'source_language_fidelity_failed'
                        or str(transcript.json_data.get('final_malayalam_fidelity_decision', '') or '').strip().lower() == 'source_language_fidelity_failed'
                        or bool(transcript.json_data.get('catastrophic_latin_substitution_failure', False))
                    )
                if transcript_state in {'low_confidence', 'failed'} or transcript_fidelity_failed:
                    warning_answer = "The transcript does not contain enough evidence to answer this confidently."
                    error_message = (
                        'Malayalam source transcript was not faithful enough for grounded chat'
                        if transcript_fidelity_failed else
                        'Transcript quality is too low for grounded chat'
                    )
                    chat_en_view = _build_chat_english_view(
                        warning_answer,
                        answer_language=output_language,
                        transcript=transcript,
                    )
                    chat_source_hash = build_english_view_source_hash(
                        "chat",
                        {
                            "answer_text": warning_answer,
                            "answer_language": output_language,
                            "sources": [],
                        },
                    )
                    chat_cache_entry = build_english_view_cache_entry(
                        chat_en_view,
                        source_hash=chat_source_hash,
                        build_reason="chat_warning_answer",
                        source_language=output_language,
                        policy={
                            "translation_state": chat_en_view.get("_english_view_policy_mode", chat_en_view.get("translation_state", "")),
                            "policy_reason": chat_en_view.get("_english_view_policy_reason", ""),
                            "current_available_views": chat_en_view.get("current_available_views", ["original"]),
                        },
                    )
                    logger.info(
                        "[EN_VIEW_PERSIST_RESULT] kind=chat transcript_id=%s source_hash=%s available=%s",
                        getattr(transcript, "id", ""),
                        chat_source_hash,
                        bool(chat_en_view.get("english_view_available", False)),
                    )
                    bot_msg = ChatMessage.objects.create(
                        session=session,
                        sender='bot',
                        message=warning_answer,
                        referenced_segments=_pack_chat_sources_with_english_view_cache([], chat_cache_entry),
                        user_language=user_language,
                        output_language=output_language,
                        retrieval_language=retrieval_language,
                    )
                    return Response({
                        'answer': warning_answer,
                        'sources': [],
                        'session_id': str(session.id),
                        'user_language': user_language,
                        'output_language': output_language,
                        'retrieval_language': retrieval_language,
                        'timestamp_context': _format_timestamp_label(context_timestamp) if context_timestamp is not None else None,
                        'processing_metadata': build_processing_metadata(video, transcript),
                        'error': error_message,
                        'english_view_answer': chat_en_view.get('english_view_text', ''),
                        'english_view_available': bool(chat_en_view.get('english_view_available', False)),
                        'chatbot_answer_language': output_language,
                        'chatbot_english_view_available': bool(chat_en_view.get('english_view_available', False)),
                        'chatbot_translation_state': str(chat_en_view.get('translation_state', '') or ''),
                        'chatbot_translation_warning': str(chat_en_view.get('translation_warning', '') or ''),
                        'chatbot_translation_blocked_reason': str(chat_en_view.get('translation_blocked_reason', '') or ''),
                        'chatbot_blocked_reason': 'malayalam_source_fidelity_failed' if transcript_fidelity_failed else 'transcript_quality_too_low',
                    })

                # Initialize chatbot engine
                chatbot = ChatbotEngine(str(video_id))

                if bool(getattr(settings, 'RENDER_TRANSCRIPT_ONLY_MODE', False)):
                    if bool(getattr(settings, 'RENDER_SAFE_CHATBOT_MODE', False)):
                        result = chatbot.ask_safe_summary_only(
                            message,
                            response_language=output_language,
                        )
                        localized_sources = result.get('sources', []) or []
                        final_answer = str(result.get('answer', '') or '')
                        raw_answer = final_answer
                        chat_en_view = _build_chat_english_view(
                            final_answer,
                            answer_language=output_language,
                            transcript=transcript,
                            translated_english_hint=raw_answer if output_language == 'en' else "",
                        )
                        chat_source_hash = build_english_view_source_hash(
                            "chat",
                            {
                                "answer_text": final_answer,
                                "answer_language": output_language,
                                "sources": localized_sources,
                            },
                        )
                        chat_cache_entry = build_english_view_cache_entry(
                            chat_en_view,
                            source_hash=chat_source_hash,
                            build_reason="chat_response_render_safe",
                            source_language=output_language,
                            policy={
                                "translation_state": chat_en_view.get("_english_view_policy_mode", chat_en_view.get("translation_state", "")),
                                "policy_reason": chat_en_view.get("_english_view_policy_reason", ""),
                                "current_available_views": chat_en_view.get("current_available_views", ["original"]),
                            },
                        )
                        ChatMessage.objects.create(
                            session=session,
                            sender='bot',
                            message=final_answer,
                            referenced_segments=_pack_chat_sources_with_english_view_cache(localized_sources, chat_cache_entry),
                            user_language=user_language,
                            output_language=output_language,
                            retrieval_language='en',
                        )
                        response_data = {
                            'answer': final_answer,
                            'sources': localized_sources,
                            'session_id': str(session.id),
                            'user_language': user_language,
                            'output_language': output_language,
                            'retrieval_language': 'en',
                            'timestamp_context': _format_timestamp_label(context_timestamp) if context_timestamp is not None else None,
                            'processing_metadata': build_processing_metadata(video, transcript),
                            'english_view_answer': chat_en_view.get('english_view_text', ''),
                            'english_view_available': bool(chat_en_view.get('english_view_available', False)),
                            'chatbot_answer_language': output_language,
                            'chatbot_english_view_available': bool(chat_en_view.get('english_view_available', False)),
                            'chatbot_translation_state': str(chat_en_view.get('translation_state', '') or ''),
                            'chatbot_translation_warning': str(chat_en_view.get('translation_warning', '') or ''),
                            'chatbot_translation_blocked_reason': str(chat_en_view.get('translation_blocked_reason', '') or ''),
                        }
                        if result.get('error'):
                            response_data['error'] = result['error']
                            response_data['chatbot_blocked_reason'] = 'render_safe_summary_only'
                        return Response(response_data)

                    return Response(
                        _render_transcript_only_chat_disabled_response(
                            session_id=session.id,
                            transcript=transcript,
                            video=video,
                            output_language=output_language,
                        ),
                        status=status.HTTP_503_SERVICE_UNAVAILABLE,
                    )
                
                # Build index if needed
                index_exists = chatbot.initialize()
                if chatbot.index_blocked_reason:
                    warning_answer = "The transcript does not contain enough faithful Malayalam evidence to answer this safely."
                    chat_en_view = _build_chat_english_view(
                        warning_answer,
                        answer_language=output_language,
                        transcript=transcript,
                    )
                    bot_msg = ChatMessage.objects.create(
                        session=session,
                        sender='bot',
                        message=warning_answer,
                        referenced_segments=_pack_chat_sources_with_english_view_cache([], None),
                        user_language=user_language,
                        output_language=output_language,
                        retrieval_language=retrieval_language,
                    )
                    return Response({
                        'answer': warning_answer,
                        'sources': [],
                        'session_id': str(session.id),
                        'user_language': user_language,
                        'output_language': output_language,
                        'retrieval_language': retrieval_language,
                        'timestamp_context': _format_timestamp_label(context_timestamp) if context_timestamp is not None else None,
                        'processing_metadata': build_processing_metadata(video, transcript),
                        'error': 'Malayalam source transcript was not faithful enough for grounded chat',
                        'english_view_answer': chat_en_view.get('english_view_text', ''),
                        'english_view_available': bool(chat_en_view.get('english_view_available', False)),
                        'chatbot_answer_language': output_language,
                        'chatbot_english_view_available': bool(chat_en_view.get('english_view_available', False)),
                        'chatbot_translation_state': str(chat_en_view.get('translation_state', '') or ''),
                        'chatbot_translation_warning': str(chat_en_view.get('translation_warning', '') or ''),
                        'chatbot_translation_blocked_reason': str(chat_en_view.get('translation_blocked_reason', '') or ''),
                        'chatbot_blocked_reason': chatbot.index_blocked_reason,
                    })
                if not index_exists:
                    chatbot.build_from_transcript(_normalize_transcript_segments(transcript))
                
                retrieval_query = message
                translated_q = False
                if user_language != retrieval_language:
                    retrieval_query = translate_text(
                        message,
                        source_language=user_language,
                        target_language=retrieval_language,
                        preserve_format=False
                    )
                    translated_q = retrieval_query.strip() != message.strip()
                logger.info(
                    "[CHAT] q_lang=%s q_conf=%.2f out_lang=%s retrieval_lang=%s translated_q=%s context_ts=%s",
                    user_language,
                    float(user_confidence or 0.0),
                    output_language,
                    retrieval_language,
                    translated_q,
                    context_timestamp,
                )

                moment_segments = []
                if context_timestamp is not None:
                    moment_segments = _get_context_segments_around_timestamp(
                        _normalize_transcript_segments(transcript),
                        context_timestamp,
                        context_window_seconds,
                    )

                # Get answer
                result = chatbot.ask(
                    retrieval_query,
                    use_llm=bool(getattr(settings, 'CHATBOT_USE_GROQ_LLM', False)),
                    strict_mode=strict_mode,
                    response_language=retrieval_language,
                    moment_segments=moment_segments,
                    context_timestamp=context_timestamp,
                    context_window_seconds=context_window_seconds,
                )

                final_answer = result.get('answer', '')
                raw_answer = final_answer
                if retrieval_language != output_language:
                    final_answer = translate_text(
                        final_answer,
                        source_language=retrieval_language,
                        target_language=output_language,
                        preserve_format=True
                    )

                localized_sources = []
                for src in result.get('sources', []):
                    src_copy = dict(src)
                    src_text = str(src_copy.get('text', '')).strip()
                    if retrieval_language != output_language and src_text:
                        src_text = translate_text(
                            src_text,
                            source_language=retrieval_language,
                            target_language=output_language,
                            preserve_format=False
                        )
                    src_copy['text'] = src_text
                    localized_sources.append(src_copy)
                english_view_hint = raw_answer if retrieval_language == 'en' else ""
                chat_en_view = _build_chat_english_view(
                    final_answer,
                    answer_language=output_language,
                    transcript=transcript,
                    translated_english_hint=english_view_hint,
                )
                chat_source_hash = build_english_view_source_hash(
                    "chat",
                    {
                        "answer_text": final_answer,
                        "answer_language": output_language,
                        "sources": localized_sources,
                    },
                )
                chat_cache_entry = build_english_view_cache_entry(
                    chat_en_view,
                    source_hash=chat_source_hash,
                    build_reason="chat_response",
                    source_language=output_language,
                    policy={
                        "translation_state": chat_en_view.get("_english_view_policy_mode", chat_en_view.get("translation_state", "")),
                        "policy_reason": chat_en_view.get("_english_view_policy_reason", ""),
                        "current_available_views": chat_en_view.get("current_available_views", ["original"]),
                    },
                )
                logger.info(
                    "[EN_VIEW_PERSIST_RESULT] kind=chat transcript_id=%s source_hash=%s available=%s",
                    getattr(transcript, "id", ""),
                    chat_source_hash,
                    bool(chat_en_view.get("english_view_available", False)),
                )
                
                # Save bot message with sources
                bot_msg = ChatMessage.objects.create(
                    session=session,
                    sender='bot',
                    message=final_answer,
                    referenced_segments=_pack_chat_sources_with_english_view_cache(localized_sources, chat_cache_entry),
                    user_language=user_language,
                    output_language=output_language,
                    retrieval_language=retrieval_language,
                )
                
                response_data = {
                    'answer': final_answer,
                    'sources': localized_sources,
                    'session_id': str(session.id),
                    'user_language': user_language,
                    'output_language': output_language,
                    'retrieval_language': retrieval_language,
                    'timestamp_context': result.get('timestamp_context') or (_format_timestamp_label(context_timestamp) if context_timestamp is not None else None),
                    'processing_metadata': build_processing_metadata(video, transcript),
                    'english_view_answer': chat_en_view.get('english_view_text', ''),
                    'english_view_available': bool(chat_en_view.get('english_view_available', False)),
                    'chatbot_answer_language': output_language,
                    'chatbot_english_view_available': bool(chat_en_view.get('english_view_available', False)),
                    'chatbot_translation_state': str(chat_en_view.get('translation_state', '') or ''),
                    'chatbot_translation_warning': str(chat_en_view.get('translation_warning', '') or ''),
                    'chatbot_translation_blocked_reason': str(chat_en_view.get('translation_blocked_reason', '') or ''),
                }
                
                # Generate TTS for the answer if requested
                generate_tts = request.data.get('generate_tts', False)
                if generate_tts:
                    try:
                        from videos.tts_utils import text_to_speech
                        voice_narration = _build_voice_narration(final_answer, localized_sources)
                        tts_path = text_to_speech(
                            voice_narration,
                            lang=_select_tts_language(output_language),
                        )
                        audio_url = f"/media/{tts_path}"
                        bot_msg.audio_url = audio_url
                        bot_msg.voice_narration = voice_narration
                        bot_msg.save(update_fields=['audio_url', 'voice_narration'])
                        response_data['audio_url'] = audio_url
                    except Exception as e:
                        logger.warning(f"TTS generation failed: {e}")
                
                if result.get('error'):
                    response_data['error'] = result['error']
                
                return Response(response_data)
                
            except Video.DoesNotExist:
                return Response(
                    {'error': 'Video not found'},
                    status=status.HTTP_404_NOT_FOUND
                )
            
        except Exception as e:
            logger.error(f"Chatbot error: {e}")
            return Response(
                {
                    'error': 'Failed to process question',
                    'detail': str(e),
                    'session_id': str(session.id) if 'session' in locals() else None
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def get(self, request):
        """Get suggested questions for a video."""
        video_id = request.query_params.get('video_id')
        
        if not video_id:
            return Response(
                {'error': 'video_id is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if bool(getattr(settings, 'RENDER_TRANSCRIPT_ONLY_MODE', False)):
            return Response({'questions': []})
        
        chatbot = ChatbotEngine(str(video_id))
        suggested_questions = chatbot.get_suggested_questions()
        
        return Response({'questions': suggested_questions})


class VideoIndexViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for viewing video indices (read-only)."""
    queryset = VideoIndex.objects.all()
    serializer_class = VideoIndexSerializer
    
    def get_queryset(self):
        """Filter by video_id if provided."""
        queryset = super().get_queryset()
        video_id = self.request.query_params.get('video_id')
        if video_id:
            queryset = queryset.filter(video_id=video_id)
        return queryset
    
    @action(detail=False, methods=['post'])
    def build(self, request):
        """Build index for a video."""
        if bool(getattr(settings, 'RENDER_TRANSCRIPT_ONLY_MODE', False)):
            return Response(
                {'error': 'Chatbot index build is disabled on the live demo.'},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        video_id = request.data.get('video_id')
        
        if not video_id:
            return Response(
                {'error': 'video_id is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            from videos.models import Video, Transcript
            
            video = Video.objects.get(id=video_id)
            transcript = Transcript.objects.filter(video=video).order_by('-created_at').first()
            
            if not transcript:
                return Response(
                    {'error': 'No transcript available'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Initialize and build index
            chatbot = ChatbotEngine(str(video_id))
            
            segments = _normalize_transcript_segments(transcript)
            success = chatbot.build_from_transcript(segments)
            if chatbot.index_blocked_reason:
                return Response(
                    {'error': chatbot.index_blocked_reason},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            if success:
                # Update VideoIndex record
                VideoIndex.objects.update_or_create(
                    video_id=video_id,
                    defaults={
                        'is_indexed': True,
                        'index_created_at': timezone.now(),
                        'num_documents': len(chatbot.rag_engine.documents)
                    }
                )
                return Response({'status': 'index built successfully'})
            else:
                return Response(
                    {'error': 'Failed to build index'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            
        except Video.DoesNotExist:
            return Response(
                {'error': 'Video not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.error(f"Index build error: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
