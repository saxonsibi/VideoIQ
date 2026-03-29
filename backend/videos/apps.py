import os
import sys
import threading
import time
import logging
from datetime import timedelta

from django.apps import AppConfig
from django.conf import settings


class VideosConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'videos'
    verbose_name = 'Video Processing'
    
    def ready(self):
        # Import signals when app is ready
        import videos.signals  # noqa

        command = next((arg for arg in sys.argv[1:] if not arg.startswith('-')), '')
        if command in {'test', 'migrate', 'makemigrations', 'collectstatic', 'shell', 'dbshell'}:
            return

        if settings.DEBUG and os.environ.get('RUN_MAIN') not in {'true', '1'}:
            return

        if getattr(settings, 'ON_RENDER', False):
            return

        if getattr(settings, 'DEV_SYNC_MODE', False) and getattr(settings, 'DEV_SYNC_RECOVERY_ENABLED', False):
            self._start_sync_recovery_thread()

        if not getattr(settings, 'ASR_MALAYALAM_WARMUP', False):
            return

        def _prewarm():
            try:
                from . import utils as videos_utils
                videos_utils.prewarm_malayalam_asr()
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning("Malayalam ASR prewarm skipped: %s", exc)

        threading.Thread(target=_prewarm, daemon=True, name='malayalam-asr-prewarm').start()

    def _start_sync_recovery_thread(self):
        logger = logging.getLogger(__name__)

        def _recover():
            from django.db import close_old_connections
            from django.utils import timezone
            from .models import Video
            from .tasks import resume_interrupted_video_processing_sync

            delay_seconds = int(getattr(settings, 'DEV_SYNC_RECOVERY_STARTUP_DELAY_SECONDS', 20) or 20)
            stale_after = int(getattr(settings, 'DEV_SYNC_RECOVERY_MAX_AGE_SECONDS', 45) or 45)
            interrupted_statuses = [
                'processing',
                'extracting_audio',
                'transcribing',
                'cleaning_transcript',
                'transcript_ready',
                'summarizing_quick',
                'summarizing_final',
                'summarizing',
                'indexing_chat',
            ]

            time.sleep(max(0, delay_seconds))
            close_old_connections()
            try:
                cutoff = timezone.now() - timedelta(seconds=max(1, stale_after))
                stranded = list(
                    Video.objects.filter(status__in=interrupted_statuses, updated_at__lt=cutoff)
                    .order_by('updated_at')[:3]
                )
                if stranded:
                    logger.warning(
                        "DEV_SYNC recovery scanning %d stranded videos after startup",
                        len(stranded),
                    )
                for video in stranded:
                    try:
                        resume_interrupted_video_processing_sync(str(video.id))
                    except Exception as exc:
                        logger.warning("DEV_SYNC recovery failed for %s: %s", video.id, exc)
            finally:
                close_old_connections()

        threading.Thread(
            target=_recover,
            daemon=True,
            name='dev-sync-recovery',
        ).start()
