"""
API Views for video processing - Synchronous version (no Celery)
"""

import os
import logging
import threading
from django.conf import settings
from django.shortcuts import get_object_or_404
from rest_framework import viewsets, status, views
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from django.db import transaction

from .models import Video, Transcript, Summary, HighlightSegment, ShortVideo, ProcessingTask
from .serializers import (
    VideoSerializer, VideoUploadSerializer, VideoDetailSerializer,
    TranscriptSerializer, SummarySerializer, SummaryGenerateSerializer,
    HighlightSegmentSerializer, ShortVideoSerializer, ShortVideoGenerateSerializer,
    ProcessingTaskSerializer, get_or_build_structured_summary
)
from .utils import (
    extract_audio, transcribe_video, summarize_text,
    detect_highlights, create_short_video, get_video_duration,
    normalize_language_code
)
from .summary_schema import default_structured_summary
from .processing_metadata import build_processing_metadata

logger = logging.getLogger(__name__)


def _storage_backend_label(field_file) -> str:
    try:
        return field_file.storage.__class__.__name__
    except Exception:
        return 'unknown'


class TranscriptViewSet(viewsets.ModelViewSet):
    """ViewSet for managing transcripts with edit capability."""
    serializer_class = TranscriptSerializer
    
    def get_queryset(self):
        return Transcript.objects.all()
    
    def update(self, request, *args, **kwargs):
        """Update transcript text (for manual corrections)."""
        partial = kwargs.pop('partial', False)
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)
        return Response(serializer.data)


def process_video_sync(
    video_id,
    transcription_language: str = 'auto',
    output_language: str = 'auto',
    summary_language_mode: str = 'same_as_transcript'
):
    """Delegate to shared sync pipeline used by manual uploads."""
    from videos.tasks import process_video_transcription_sync
    process_video_transcription_sync(
        video_id,
        transcription_language=transcription_language,
        output_language=output_language,
        summary_language_mode=summary_language_mode
    )


def _launch_manual_processing(
    video_id: str,
    transcription_language: str = 'auto',
    output_language: str = 'auto',
    summary_language_mode: str = 'same_as_transcript'
):
    """Run manual upload processing according to DEV_SYNC_MODE toggle."""
    if getattr(settings, 'DEV_SYNC_MODE', False):
        # Run in background thread so video appears as queued first
        def _run_manual_processing(v_id, t_lang, o_lang, s_mode):
            from django.db import close_old_connections
            from videos.tasks import process_video_transcription_sync
            close_old_connections()
            try:
                process_video_transcription_sync(
                    v_id,
                    transcription_language=t_lang,
                    output_language=o_lang,
                    summary_language_mode=s_mode
                )
            except Exception:
                logger.exception(f"DEV_SYNC_MODE manual processing failed for video {v_id}")
            finally:
                close_old_connections()

        threading.Thread(
            target=_run_manual_processing,
            args=(video_id, transcription_language, output_language, summary_language_mode),
            daemon=True
        ).start()
        return

    from videos.tasks import process_video_transcription
    process_video_transcription.delay(
        video_id,
        transcription_language=transcription_language,
        output_language=output_language,
        summary_language_mode=summary_language_mode
    )


def _launch_youtube_processing(
    video_id: str,
    transcription_language: str = 'auto',
    output_language: str = 'auto',
    summary_language_mode: str = 'same_as_transcript'
):
    """Run YouTube processing according to DEV_SYNC_MODE toggle."""
    if getattr(settings, 'DEV_SYNC_MODE', False):
        # Run in background thread so video appears as queued first
        def _run_youtube_processing(v_id, t_lang, o_lang, s_mode):
            from django.db import close_old_connections
            from videos.tasks import process_youtube_video_sync
            close_old_connections()
            try:
                process_youtube_video_sync(
                    v_id,
                    transcription_language=t_lang,
                    output_language=o_lang,
                    summary_language_mode=s_mode
                )
            except Exception:
                logger.exception(f"DEV_SYNC_MODE YouTube processing failed for video {v_id}")
            finally:
                close_old_connections()

        threading.Thread(
            target=_run_youtube_processing,
            args=(video_id, transcription_language, output_language, summary_language_mode),
            daemon=True
        ).start()
        return

    from videos.tasks import process_youtube_video
    process_youtube_video.delay(
        video_id,
        transcription_language=transcription_language,
        output_language=output_language,
        summary_language_mode=summary_language_mode
    )


class VideoUploadView(views.APIView):
    """Handle video uploads."""
    parser_classes = [MultiPartParser, FormParser]
    
    def post(self, request):
        """Upload a new video for processing."""
        serializer = VideoUploadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            video_file = serializer.validated_data['file']
            title = serializer.validated_data.get('title', video_file.name)
            description = serializer.validated_data.get('description', '')
            transcription_language = normalize_language_code(
                serializer.validated_data.get('transcription_language'),
                default='auto',
                allow_auto=True
            )
            output_language = normalize_language_code(
                serializer.validated_data.get('output_language'),
                default='auto',
                allow_auto=True
            )
            summary_language_mode = (
                str(request.data.get('summary_language_mode', 'same_as_transcript')).strip().lower()
                or 'same_as_transcript'
            )
            
            # Create video instance
            video = Video.objects.create(
                title=title,
                description=description,
                original_file=video_file,
                file_size=video_file.size,
                file_format=os.path.splitext(video_file.name)[1].lower()[1:],
                status='uploaded'
            )

            logger.warning(
                "[UPLOAD_SAVED] video_id=%s storage_backend=%s file_name=%s file_size=%s",
                str(video.id),
                _storage_backend_label(video.original_file),
                str(getattr(video.original_file, 'name', '') or ''),
                int(video.file_size or 0),
            )
            
            _launch_manual_processing(
                str(video.id),
                transcription_language=transcription_language,
                output_language=output_language,
                summary_language_mode=summary_language_mode
            )
            
            response_serializer = VideoSerializer(video)
            return Response(response_serializer.data, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            logger.error(f"Video upload failed: {str(e)}")
            return Response(
                {'error': 'Video upload failed', 'detail': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class YouTubeURLUploadView(views.APIView):
    """Handle YouTube URL uploads."""
    parser_classes = [JSONParser]
    
    def post(self, request):
        """Upload a YouTube video for processing."""
        url = request.data.get('url')
        title = request.data.get('title', '')
        transcription_language = normalize_language_code(
            request.data.get('transcription_language'),
            default='auto',
            allow_auto=True
        )
        output_language = normalize_language_code(
            request.data.get('output_language'),
            default='auto',
            allow_auto=True
        )
        summary_language_mode = (
            str(request.data.get('summary_language_mode', 'same_as_transcript')).strip().lower()
            or 'same_as_transcript'
        )
        
        if not url:
            return Response(
                {'error': 'YouTube URL is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Validate YouTube URL
        youtube_patterns = [
            r'(?:https?://)?(?:www\.|m\.)?youtube\.com/watch\?v=[\w-]+',
            r'(?:https?://)?(?:www\.)?youtu\.be/[\w-]+',
            r'(?:https?://)?(?:www\.|m\.)?youtube\.com/shorts/[\w-]+',
            r'(?:https?://)?(?:www\.|m\.)?youtube\.com/embed/[\w-]+',
            r'(?:https?://)?(?:www\.|m\.)?youtube\.com/v/[\w-]+',
        ]
        
        import re
        is_valid = any(re.match(pattern, url) for pattern in youtube_patterns)
        
        if not is_valid:
            return Response(
                {'error': 'Invalid YouTube URL format'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Use provided title or default
            if not title:
                title = 'YouTube Video'
            
            # Create video instance with YouTube URL
            video = Video.objects.create(
                title=title,
                description=f'YouTube URL: {url}',
                youtube_url=url,
                status='uploaded'
            )
            
            _launch_youtube_processing(
                str(video.id),
                transcription_language=transcription_language,
                output_language=output_language,
                summary_language_mode=summary_language_mode
            )

            response_serializer = VideoSerializer(video)
            return Response(response_serializer.data, status=status.HTTP_202_ACCEPTED)
            
        except Exception as e:
            logger.error(f"YouTube URL upload failed: {str(e)}")
            return Response(
                {'error': 'YouTube URL processing failed', 'detail': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class TranscriptViewSet(viewsets.ModelViewSet):
    """ViewSet for managing transcripts."""
    serializer_class = TranscriptSerializer
    
    def get_queryset(self):
        return Transcript.objects.all()
    
    def update(self, request, *args, **kwargs):
        """Update transcript text (for manual corrections)."""
        partial = kwargs.pop('partial', False)
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)
        return Response(serializer.data)


class VideoViewSet(viewsets.ModelViewSet):
    """ViewSet for Video CRUD operations."""
    queryset = Video.objects.all()
    serializer_class = VideoSerializer
    
    def get_serializer_class(self):
        if self.action == 'retrieve':
            return VideoDetailSerializer
        return VideoSerializer
    
    def get_queryset(self):
        """Filter videos by user if provided."""
        queryset = Video.objects.all()
        return queryset.prefetch_related(
            'transcripts', 'summaries', 'highlight_segments', 'short_videos'
        )
    
    def perform_destroy(self, instance):
        """Delete video and associated files."""
        instance.delete()
    
    @action(detail=True, methods=['get'])
    def transcripts(self, request, pk=None):
        """Get all transcripts for a video."""
        video = self.get_object()
        transcripts = video.transcripts.all()
        serializer = TranscriptSerializer(transcripts, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['put', 'patch'])
    def update_transcript(self, request, pk=None):
        """Update transcript text (for manual corrections)."""
        video = self.get_object()
        transcript = video.transcripts.first()
        
        if not transcript:
            return Response(
                {'error': 'No transcript found for this video'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Update the full_text field
        full_text = request.data.get('full_text')
        if full_text:
            transcript.full_text = full_text
            transcript.save()
        
        serializer = TranscriptSerializer(transcript)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'])
    def generate_transcript(self, request, pk=None):
        """Generate transcript for a video."""
        video = self.get_object()
        
        in_progress_statuses = {
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
        if video.status in in_progress_statuses:
            return Response(
                {'error': 'Video is already being processed'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Process synchronously
        try:
            raw_t_lang = request.data.get('transcription_language')
            default_t_lang = 'auto' if video.youtube_url else 'en'
            transcription_language = normalize_language_code(
                raw_t_lang,
                default=default_t_lang,
                allow_auto=True
            )
            output_language = normalize_language_code(
                request.data.get('output_language'),
                default='auto',
                allow_auto=True
            )
            summary_language_mode = (
                str(request.data.get('summary_language_mode', 'same_as_transcript')).strip().lower()
                or 'same_as_transcript'
            )
            # English View: Request translation to English
            english_view = request.data.get('english_view', False)
            if video.original_file:
                process_video_sync(
                    str(video.id),
                    transcription_language=transcription_language,
                    output_language=output_language,
                    summary_language_mode=summary_language_mode,
                    english_view=english_view
                )
            elif video.youtube_url:
                from videos.tasks import process_youtube_video_sync
                process_youtube_video_sync(
                    str(video.id),
                    transcription_language=transcription_language,
                    output_language=output_language,
                    summary_language_mode=summary_language_mode,
                    english_view=english_view
                )
            else:
                return Response(
                    {'error': 'No valid video source found for transcript generation.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            video.refresh_from_db()
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        serializer = VideoSerializer(video)
        return Response(serializer.data)
    
    @action(detail=True, methods=['get'])
    def summaries(self, request, pk=None):
        """Get all summaries for a video."""
        video = self.get_object()
        summaries = video.summaries.all()
        serializer = SummarySerializer(summaries, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['get'])
    def structured_summary(self, request, pk=None):
        """Get bounded structured summary object without changing existing summary APIs."""
        video = self.get_object()
        transcript = video.transcripts.order_by('-created_at').first()
        if not transcript:
            return Response({
                **default_structured_summary(),
                "processing_metadata": build_processing_metadata(video, None),
            })

        payload = get_or_build_structured_summary(video, transcript)
        payload["processing_metadata"] = build_processing_metadata(video, transcript)
        return Response(payload)
    
    @action(detail=True, methods=['post'])
    def generate_summary(self, request, pk=None):
        """Generate summary for a video."""
        video = self.get_object()
        serializer = SummaryGenerateSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        summary_type = serializer.validated_data.get('summary_type', 'full')
        max_length = serializer.validated_data.get('max_length')
        min_length = serializer.validated_data.get('min_length')
        summary_language_mode = serializer.validated_data.get('summary_language_mode', 'same_as_transcript')
        output_language = normalize_language_code(
            serializer.validated_data.get('output_language'),
            default='auto',
            allow_auto=True
        )
        
        # Delete existing summary and regenerate
        video.summaries.filter(summary_type=summary_type).delete()
        
        # Generate summary synchronously
        try:
            transcript = video.transcripts.first()
            if not transcript:
                return Response(
                    {'error': 'No transcript found. Generate transcript first.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            summary_text = summarize_text(
                transcript.transcript_original_text or transcript.full_text,
                summary_type=summary_type,
                max_length=max_length,
                min_length=min_length,
                output_language=output_language,
                source_language=transcript.transcript_language or transcript.language,
                canonical_text=transcript.transcript_canonical_text or '',
                canonical_language=transcript.canonical_language or 'en',
                summary_language_mode=summary_language_mode
            )
            summary = Summary.objects.create(
                video=video,
                summary_type=summary_type,
                title=summary_text.get('title', f'{summary_type.capitalize()} Summary'),
                content=summary_text.get('content') or summary_text.get('summary', ''),
                key_topics=summary_text.get('key_topics', []),
                summary_language=summary_text.get('summary_language', 'en'),
                summary_source_language=summary_text.get('summary_source_language', transcript.transcript_language or transcript.language or 'en'),
                translation_used=bool(summary_text.get('translation_used', False)),
                model_used=summary_text.get('model_used') or summary_text.get('model', 'facebook/bart-large-cnn'),
                generation_time=summary_text.get('generation_time', 0)
            )
            
            # Also detect highlights if none exist
            if not video.highlight_segments.exists():
                try:
                    highlights = detect_highlights(transcript)
                    for highlight in highlights:
                        HighlightSegment.objects.create(
                            video=video,
                            start_time=highlight.get('start_time', highlight.get('start', 0)),
                            end_time=highlight.get('end_time', highlight.get('end', 0)),
                            importance_score=highlight.get('importance_score', highlight.get('score', 0.5)),
                            reason=highlight.get('reason', ''),
                            transcript_snippet=highlight.get('transcript_snippet', highlight.get('text', ''))
                        )
                    logger.info(f"Created {len(highlights)} highlight segments for video {video.id}")
                except Exception as e:
                    logger.warning(f"Highlight detection failed: {str(e)}")
            
            serializer = SummarySerializer(summary)
            payload = dict(serializer.data)
            payload['summary_en'] = summary_text.get('summary_en', payload.get('content', ''))
            payload['summary_out'] = summary_text.get('summary_out', payload.get('content', ''))
            payload['structured_summary'] = get_or_build_structured_summary(video, transcript)
            payload['processing_metadata'] = build_processing_metadata(video, transcript)
            return Response(payload)
            
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['get'])
    def highlights(self, request, pk=None):
        """Get highlight segments for a video."""
        video = self.get_object()
        segments = video.highlight_segments.all()
        serializer = HighlightSegmentSerializer(segments, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'])
    def generate_audio_summary(self, request, pk=None):
        """
        Generate an audio summary (podcast-style) for a video.
        Returns an audio file that can be downloaded.
        """
        video = self.get_object()
        
        try:
            # Get transcript
            transcript = video.transcripts.first()
            if not transcript:
                return Response(
                    {'error': 'No transcript found. Generate transcript first.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Get full text
            full_text = transcript.full_text
            if not full_text:
                return Response(
                    {'error': 'Transcript is empty.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Generate audio summary using TTS
            from .tts_utils import generate_podcast_summary
            audio_path = generate_podcast_summary(full_text)
            
            return Response({
                'audio_url': f"/media/{audio_path}",
                'message': 'Audio summary generated successfully'
            })
            
        except Exception as e:
            logger.error(f"Audio summary generation failed: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['get'])
    def tasks(self, request, pk=None):
        """Get processing tasks for a video."""
        video = self.get_object()
        tasks = video.tasks.all()
        serializer = ProcessingTaskSerializer(tasks, many=True)
        return Response(serializer.data)
