import os
import sys
import threading

from django.apps import AppConfig
from django.conf import settings


class ChatbotConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'chatbot'
    verbose_name = 'Video Chatbot'
    
    def ready(self):
        if getattr(settings, 'ON_RENDER', False):
            return

        if not getattr(settings, 'RAG_PREWARM_EMBEDDING_MODEL', True):
            return

        command = next((arg for arg in sys.argv[1:] if not arg.startswith('-')), '')
        if command in {'test', 'migrate', 'makemigrations', 'collectstatic', 'shell', 'dbshell'}:
            return

        if settings.DEBUG and os.environ.get('RUN_MAIN') not in {'true', '1'}:
            return

        def _prewarm():
            try:
                from .rag_engine import prewarm_embedding_model, prewarm_embedding_skip_reason
                skip_reason = prewarm_embedding_skip_reason()
                if skip_reason:
                    import logging
                    logging.getLogger(__name__).warning("Embedding prewarm skipped: %s", skip_reason)
                    return
                prewarm_embedding_model()
            except Exception as exc:
                # Warmup is opportunistic; never block app startup.
                import logging
                logging.getLogger(__name__).warning("Embedding prewarm skipped: %s", exc)

        threading.Thread(target=_prewarm, daemon=True, name='rag-embedding-prewarm').start()
