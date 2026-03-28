"""
Django settings for VideoIQ AI Video Intelligence System project.
"""

import os
from pathlib import Path
from django.core.exceptions import ImproperlyConfigured
import dj_database_url
from dotenv import load_dotenv, dotenv_values

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIST_DIR = BASE_DIR / 'frontend_dist'
DOTENV_PATH = os.path.join(BASE_DIR, '.env')

# Load environment variables from .env file
load_dotenv(DOTENV_PATH)
DOTENV_VALUES = dotenv_values(DOTENV_PATH)


def _env_bool(name: str, default: str = 'False', *, prefer_dotenv: bool = False) -> bool:
    raw = None
    if prefer_dotenv:
        raw = DOTENV_VALUES.get(name, None)
    if raw is None:
        raw = os.environ.get(name, default)
    return str(raw).strip().lower() in ('true', '1', 'yes', 'on')

# Create logs directory if it doesn't exist
LOG_DIR = BASE_DIR / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'django-insecure-dev-key-change-in-production')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.environ.get('DJANGO_DEBUG', 'True').lower() in ('true', '1', 'yes')

_allowed_hosts = os.environ.get('DJANGO_ALLOWED_HOSTS', '*')
ALLOWED_HOSTS = [host.strip() for host in _allowed_hosts.split(',') if host.strip()] or ['*']

_csrf_trusted_origins = os.environ.get('DJANGO_CSRF_TRUSTED_ORIGINS', '')
CSRF_TRUSTED_ORIGINS = [
    origin.strip() for origin in _csrf_trusted_origins.split(',') if origin.strip()
]

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    
    # Third party apps
    'rest_framework',
    'corsheaders',
    'drf_yasg',
    
    # Local apps
    'videos',
    'chatbot',
    'summarizer',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# YouTube embedded player may fail with Error 153 if no Referer/origin is sent.
# Allow origin-level referrer on cross-origin requests.
SECURE_REFERRER_POLICY = os.environ.get('SECURE_REFERRER_POLICY', 'strict-origin-when-cross-origin')

# Allow embedding the app inside a Chrome extension side panel iframe.
# Keep disabled by default in production unless explicitly enabled.
ALLOW_EXTENSION_IFRAME = os.environ.get(
    'ALLOW_EXTENSION_IFRAME',
    'True' if DEBUG else 'False'
).lower() in ('true', '1', 'yes')
if ALLOW_EXTENSION_IFRAME:
    X_FRAME_OPTIONS = 'ALLOWALL'

ROOT_URLCONF = 'videoiq.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [FRONTEND_DIST_DIR],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'videoiq.wsgi.application'

# Database
DATABASE_URL = os.environ.get('DATABASE_URL', '').strip()
if DATABASE_URL:
    DATABASES = {
        'default': dj_database_url.parse(
            DATABASE_URL,
            conn_max_age=600,
            ssl_require=not DEBUG,
        )
    }
else:
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': BASE_DIR / 'db.sqlite3',
        }
    }

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [FRONTEND_DIST_DIR]
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# Media files (Videos, audio, etc.)
MEDIA_URL = 'media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# REST Framework configuration
REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny',
    ],
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.JSONParser',
        'rest_framework.parsers.MultiPartParser',
        'rest_framework.parsers.FormParser',
    ],
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
    'EXCEPTION_HANDLER': 'videos.exceptions.custom_exception_handler',
}

# CORS settings
if DEBUG:
    CORS_ALLOW_ALL_ORIGINS = True
    CORS_ALLOWED_ORIGINS = [
        "http://localhost:5173",
        "http://localhost:3000",
    ]
else:
    CORS_ALLOW_ALL_ORIGINS = False
    CORS_ALLOWED_ORIGINS = [
        origin.strip()
        for origin in os.environ.get('CORS_ALLOWED_ORIGINS', '').split(',')
        if origin.strip()
    ]

# Celery Configuration
CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = 'UTC'
# Local development switch: run processing without Celery worker.
# When True, videos are processed in background threads within Django runserver.
# No separate Celery worker terminal needed!
DEV_SYNC_MODE = os.environ.get('DEV_SYNC_MODE', 'False').lower() in ('true', '1', 'yes')
DEV_SYNC_RECOVERY_ENABLED = os.environ.get(
    'DEV_SYNC_RECOVERY_ENABLED',
    'True' if DEV_SYNC_MODE else 'False'
).lower() in ('true', '1', 'yes')
DEV_SYNC_RECOVERY_MAX_AGE_SECONDS = int(os.environ.get('DEV_SYNC_RECOVERY_MAX_AGE_SECONDS', '45'))
DEV_SYNC_RECOVERY_STARTUP_DELAY_SECONDS = int(os.environ.get('DEV_SYNC_RECOVERY_STARTUP_DELAY_SECONDS', '20'))
DEV_SYNC_MAX_VIDEO_SECONDS = int(os.environ.get('DEV_SYNC_MAX_VIDEO_SECONDS', '480'))
DEV_SYNC_LITE_MODE = os.environ.get(
    'DEV_SYNC_LITE_MODE',
    'True' if DEV_SYNC_MODE else 'False'
).lower() in ('true', '1', 'yes')

# File Upload Settings
DATA_UPLOAD_MAX_MEMORY_SIZE = 500 * 1024 * 1024  # 500MB
FILE_UPLOAD_MAX_MEMORY_SIZE = 500 * 1024 * 1024  # 500MB

# AI Models Configuration
# Prefer accuracy defaults; can still be overridden in .env
WHISPER_MODEL_SIZE = os.environ.get('WHISPER_MODEL_SIZE', 'large-v3')  # tiny, small, medium, large, large-v2, large-v3
WHISPER_FORCE_LARGE_V3 = os.environ.get('WHISPER_FORCE_LARGE_V3', 'True').lower() in ('true', '1', 'yes')
ASR_MALAYALAM_PRIMARY_MODEL = os.environ.get(
    'ASR_MALAYALAM_PRIMARY_MODEL',
    os.environ.get('MALAYALAM_ASR_MODEL', 'large-v2')
)
ASR_MALAYALAM_STRATEGY = os.environ.get(
    'ASR_MALAYALAM_STRATEGY',
    'quality_first'
).strip().lower()
ASR_MALAYALAM_FAST_PRIMARY_MODEL = os.environ.get(
    'ASR_MALAYALAM_FAST_PRIMARY_MODEL',
    'medium'
).strip()
ASR_MALAYALAM_SECOND_PASS_MODEL = os.environ.get(
    'ASR_MALAYALAM_SECOND_PASS_MODEL',
    'large-v2'  # Use large-v2 to avoid OOM on RTX 3050 (4GB VRAM)
).strip()
ASR_MALAYALAM_DEVICE = os.environ.get(
    'ASR_MALAYALAM_DEVICE',
    os.environ.get('MALAYALAM_ASR_DEVICE', 'auto')
).strip().lower()
ASR_MALAYALAM_MODEL_FAMILY = os.environ.get(
    'ASR_MALAYALAM_MODEL_FAMILY',
    'auto'
).strip().lower()
ASR_MALAYALAM_COMPUTE_TYPE = os.environ.get(
    'ASR_MALAYALAM_COMPUTE_TYPE',
    os.environ.get('MALAYALAM_ASR_COMPUTE_TYPE', 'float16')
).strip().lower()
ASR_MALAYALAM_SEGMENT_LOCAL_RESCUE_ALLOW_CPU = os.environ.get(
    'ASR_MALAYALAM_SEGMENT_LOCAL_RESCUE_ALLOW_CPU',
    'False'
).lower() in ('true', '1', 'yes')
ASR_MALAYALAM_PROMPT_ENABLED = os.environ.get(
    'ASR_MALAYALAM_PROMPT_ENABLED',
    'True'
).lower() in ('true', '1', 'yes')
ASR_MALAYALAM_DEBUG_RETAIN_AUDIO = os.environ.get(
    'ASR_MALAYALAM_DEBUG_RETAIN_AUDIO',
    'False'
).lower() in ('true', '1', 'yes')
ASR_MALAYALAM_DEBUG_OUTPUT_DIR = os.environ.get(
    'ASR_MALAYALAM_DEBUG_OUTPUT_DIR',
    'backend/videos/debug_audio'
)
ASR_MALAYALAM_CHUNK_MAX_DURATION_SECONDS = float(
    os.environ.get('ASR_MALAYALAM_CHUNK_MAX_DURATION_SECONDS', '18')
)
# English ASR fast path - skip preprocessing for speed
ASR_ENGLISH_USE_FAST_PATH = os.environ.get('ASR_ENGLISH_USE_FAST_PATH', 'true').lower() == 'true'
ASR_ENGLISH_CHUNK_MAX_DURATION_SECONDS = float(
    os.environ.get('ASR_ENGLISH_CHUNK_MAX_DURATION_SECONDS', '120')
)
WHISPER_MODEL_MALAYALAM_PRIMARY = os.environ.get(
    'WHISPER_MODEL_MALAYALAM_PRIMARY',
    ASR_MALAYALAM_PRIMARY_MODEL
)
WHISPER_MODEL_MALAYALAM_SECONDARY = os.environ.get(
    'WHISPER_MODEL_MALAYALAM_SECONDARY',
    ''
)
WHISPER_MODEL_MALAYALAM_FALLBACK = os.environ.get(
    'WHISPER_MODEL_MALAYALAM_FALLBACK',
    ASR_MALAYALAM_PRIMARY_MODEL
)

# Groq Whisper - MUCH FASTER (5-20x faster than CPU Faster-Whisper)
# 40 min video -> 1-3 min transcription vs 20-30 min
USE_GROQ_WHISPER = os.environ.get('USE_GROQ_WHISPER', 'True').lower() in ('true', '1', 'yes')
GROQ_WHISPER_MODEL = os.environ.get('GROQ_WHISPER_MODEL', 'whisper-large-v3-turbo')  # Use turbo for speed
GROQ_WHISPER_ENGLISH_ONLY = os.environ.get('GROQ_WHISPER_ENGLISH_ONLY', 'False').lower() in ('true', '1', 'yes')
ASR_LONGFORM_LOCAL_THRESHOLD_SECONDS = int(os.environ.get('ASR_LONGFORM_LOCAL_THRESHOLD_SECONDS', '900'))
ASR_USE_DEEPGRAM_FOR_NON_ENGLISH = os.environ.get('ASR_USE_DEEPGRAM_FOR_NON_ENGLISH', 'True').lower() in ('true', '1', 'yes')
GROQ_WHISPER_MAX_SINGLE_FILE_MB = float(os.environ.get('GROQ_WHISPER_MAX_SINGLE_FILE_MB', '23'))
GROQ_WHISPER_MAX_SINGLE_SECONDS = int(os.environ.get('GROQ_WHISPER_MAX_SINGLE_SECONDS', '600'))
DEEPGRAM_API_KEY = os.environ.get('DEEPGRAM_API_KEY', '')
DEEPGRAM_MODEL = os.environ.get('DEEPGRAM_MODEL', 'nova-2')
DEEPGRAM_BASE_URL = os.environ.get('DEEPGRAM_BASE_URL', 'https://api.deepgram.com/v1/listen')
DEEPGRAM_TIMEOUT_SEC = int(os.environ.get('DEEPGRAM_TIMEOUT_SEC', '180'))
DEEPGRAM_SUPPORTED_LANGUAGES = os.environ.get(
    'DEEPGRAM_SUPPORTED_LANGUAGES',
    'en,hi,ta,te,kn,es,fr,de,pt,it,nl,ru,ja,ko,zh,ar,tr,id,sv,no,pl,uk'
)
ASR_ENABLE_DEEPGRAM_MALAYALAM_EXPERIMENT = _env_bool(
    'ASR_ENABLE_DEEPGRAM_MALAYALAM_EXPERIMENT',
    'False',
    prefer_dotenv=True,
)
ASR_USE_GROQ_FALLBACK = os.environ.get('ASR_USE_GROQ_FALLBACK', 'True').lower() in ('true', '1', 'yes')
ASR_GROQ_COOLDOWN_SEC = int(os.environ.get('ASR_GROQ_COOLDOWN_SEC', '900'))
ASR_LOW_CONTENT_WPM_MIN = float(os.environ.get('ASR_LOW_CONTENT_WPM_MIN', '20'))
ASR_LOW_CONTENT_MIN_WORDS_LONG = int(os.environ.get('ASR_LOW_CONTENT_MIN_WORDS_LONG', '80'))
ASR_MAX_RETRIES = int(os.environ.get('ASR_MAX_RETRIES', '3'))
ASR_REJECT_ON_GARBLE = os.environ.get('ASR_REJECT_ON_GARBLE', 'True').lower() in ('true', '1', 'yes')
TRANSCRIPT_LLM_CORRECTION_ENABLED = os.environ.get('TRANSCRIPT_LLM_CORRECTION_ENABLED', 'False').lower() in ('true', '1', 'yes')
TRANSCRIPT_LLM_PROPER_NOUN_ENABLED = os.environ.get('TRANSCRIPT_LLM_PROPER_NOUN_ENABLED', 'False').lower() in ('true', '1', 'yes')
TRANSCRIPT_LLM_QUALITY_THRESHOLD = float(os.environ.get('TRANSCRIPT_LLM_QUALITY_THRESHOLD', '0.70'))
TRANSCRIPT_LLM_MIN_WORDS = int(os.environ.get('TRANSCRIPT_LLM_MIN_WORDS', '80'))
TRANSCRIPT_LLM_MAX_WORDS = int(os.environ.get('TRANSCRIPT_LLM_MAX_WORDS', '1800'))
ASR_LANGUAGE_DETECT_ENABLED = os.environ.get('ASR_LANGUAGE_DETECT_ENABLED', 'True').lower() in ('true', '1', 'yes')
ASR_LANGUAGE_DETECT_MODEL = os.environ.get('ASR_LANGUAGE_DETECT_MODEL', 'small')
ASR_LANGUAGE_DETECT_SAMPLE_SECONDS = int(os.environ.get('ASR_LANGUAGE_DETECT_SAMPLE_SECONDS', '180'))
ASR_LANGUAGE_DETECT_MIN_PROB = float(os.environ.get('ASR_LANGUAGE_DETECT_MIN_PROB', '0.35'))
ASR_LATENCY_BUDGET_SECONDS = int(os.environ.get('ASR_LATENCY_BUDGET_SECONDS', '480'))
ASR_ENABLE_QUALITY_AWARE_ROUTER = os.environ.get('ASR_ENABLE_QUALITY_AWARE_ROUTER', 'True').lower() in ('true', '1', 'yes')
ASR_PROVIDER_PRIOR_DEFAULT_QUALITY = float(os.environ.get('ASR_PROVIDER_PRIOR_DEFAULT_QUALITY', '0.72'))
ASR_AUTO_FALLBACK_LANGUAGES = os.environ.get('ASR_AUTO_FALLBACK_LANGUAGES', 'ml,hi,ta,te,kn,en')
ASR_CANDIDATE_PROBE_SECONDS = int(os.environ.get('ASR_CANDIDATE_PROBE_SECONDS', '120'))
WHISPER_FORCE_ACCURACY_PROFILE = os.environ.get('WHISPER_FORCE_ACCURACY_PROFILE', 'True').lower() in ('true', '1', 'yes')
ASR_LOCAL_MODEL_REUSE = os.environ.get('ASR_LOCAL_MODEL_REUSE', 'True').lower() in ('true', '1', 'yes')
ASR_MALAYALAM_WARMUP = os.environ.get('ASR_MALAYALAM_WARMUP', 'False').lower() in ('true', '1', 'yes')
ASR_MALAYALAM_ENABLE_FULL_RETRY = os.environ.get('ASR_MALAYALAM_ENABLE_FULL_RETRY', 'False').lower() in ('true', '1', 'yes')
ASR_MALAYALAM_RETRY_MIN_QUALITY = float(os.environ.get('ASR_MALAYALAM_RETRY_MIN_QUALITY', '0.42'))
ASR_MALAYALAM_MAX_RETRY_COUNT = int(os.environ.get('ASR_MALAYALAM_MAX_RETRY_COUNT', '1'))
ASR_MALAYALAM_MAX_TOTAL_SECONDS = int(os.environ.get('ASR_MALAYALAM_MAX_TOTAL_SECONDS', '420'))
ASR_MALAYALAM_SKIP_RETRY_ABOVE_DURATION_SECONDS = int(os.environ.get('ASR_MALAYALAM_SKIP_RETRY_ABOVE_DURATION_SECONDS', '360'))
ASR_MALAYALAM_FAST_MODE = os.environ.get('ASR_MALAYALAM_FAST_MODE', 'True').lower() in ('true', '1', 'yes')
ASR_MALAYALAM_LOCAL_INITIAL_PROMPT_ENABLED = os.environ.get('ASR_MALAYALAM_LOCAL_INITIAL_PROMPT_ENABLED', 'True').lower() in ('true', '1', 'yes')
ASR_MALAYALAM_ENABLE_LARGE_FALLBACK = os.environ.get('ASR_MALAYALAM_ENABLE_LARGE_FALLBACK', 'True').lower() in ('true', '1', 'yes')
ASR_MALAYALAM_LARGE_FALLBACK_MIN_QUALITY = float(os.environ.get('ASR_MALAYALAM_LARGE_FALLBACK_MIN_QUALITY', '0.34'))
ASR_MALAYALAM_ACCEPT_FIRST_PASS_MIN_QUALITY = float(os.environ.get('ASR_MALAYALAM_ACCEPT_FIRST_PASS_MIN_QUALITY', '0.5'))
ASR_MALAYALAM_ALLOW_USABLE_FIRST_PASS = os.environ.get('ASR_MALAYALAM_ALLOW_USABLE_FIRST_PASS', 'True').lower() in ('true', '1', 'yes')
ASR_MALAYALAM_GARBLED_MAX_RATIO = float(os.environ.get('ASR_MALAYALAM_GARBLED_MAX_RATIO', '0.34'))
ASR_MALAYALAM_MIN_SCRIPT_PURITY = float(os.environ.get('ASR_MALAYALAM_MIN_SCRIPT_PURITY', '0.85'))
ASR_MALAYALAM_MIXED_SCRIPT_OVERRIDE_MIN_QUALITY = float(os.environ.get('ASR_MALAYALAM_MIXED_SCRIPT_OVERRIDE_MIN_QUALITY', '0.88'))
ASR_MALAYALAM_MIXED_SCRIPT_OVERRIDE_MIN_SCRIPT_RATIO = float(os.environ.get('ASR_MALAYALAM_MIXED_SCRIPT_OVERRIDE_MIN_SCRIPT_RATIO', '0.45'))
ASR_MALAYALAM_MIXED_SCRIPT_OVERRIDE_MIN_INFO_DENSITY = float(os.environ.get('ASR_MALAYALAM_MIXED_SCRIPT_OVERRIDE_MIN_INFO_DENSITY', '0.55'))
ASR_MALAYALAM_MIXED_SCRIPT_OVERRIDE_MAX_REPEAT_RATIO = float(os.environ.get('ASR_MALAYALAM_MIXED_SCRIPT_OVERRIDE_MAX_REPEAT_RATIO', '0.12'))
ASR_MALAYALAM_MIXED_SCRIPT_OVERRIDE_MAX_GARBLE_SCORE = float(os.environ.get('ASR_MALAYALAM_MIXED_SCRIPT_OVERRIDE_MAX_GARBLE_SCORE', '0.72'))
ASR_MALAYALAM_ACCEPT_FIRST_PASS_MIN_SCRIPT_RATIO = float(os.environ.get('ASR_MALAYALAM_ACCEPT_FIRST_PASS_MIN_SCRIPT_RATIO', '0.55'))
ASR_MALAYALAM_MAX_REPEAT_TOKEN_RATIO = float(os.environ.get('ASR_MALAYALAM_MAX_REPEAT_TOKEN_RATIO', '0.28'))
ASR_MALAYALAM_MIN_INFO_DENSITY = float(os.environ.get('ASR_MALAYALAM_MIN_INFO_DENSITY', '0.45'))
ASR_MALAYALAM_FINAL_MIN_SCRIPT_RATIO = float(os.environ.get('ASR_MALAYALAM_FINAL_MIN_SCRIPT_RATIO', '0.60'))
ASR_MALAYALAM_FINAL_MAX_OTHER_INDIC_RATIO = float(os.environ.get('ASR_MALAYALAM_FINAL_MAX_OTHER_INDIC_RATIO', '0.20'))
ASR_MALAYALAM_FINAL_MIN_TOKEN_COVERAGE = float(os.environ.get('ASR_MALAYALAM_FINAL_MIN_TOKEN_COVERAGE', '0.55'))
ASR_MALAYALAM_FINAL_FAIL_OTHER_INDIC_RATIO = float(os.environ.get('ASR_MALAYALAM_FINAL_FAIL_OTHER_INDIC_RATIO', '0.45'))
ASR_MALAYALAM_FINAL_MIN_INTENDED_SEGMENT_RATIO = float(os.environ.get('ASR_MALAYALAM_FINAL_MIN_INTENDED_SEGMENT_RATIO', '0.55'))
ASR_MALAYALAM_FINAL_MAX_CORRUPTED_SEGMENT_SHARE = float(os.environ.get('ASR_MALAYALAM_FINAL_MAX_CORRUPTED_SEGMENT_SHARE', '0.34'))
ASR_MALAYALAM_FINAL_MIN_OVERALL_READABILITY = float(os.environ.get('ASR_MALAYALAM_FINAL_MIN_OVERALL_READABILITY', '0.34'))
ASR_MALAYALAM_FINAL_MAX_CLEANED_GARBLE_SCORE = float(os.environ.get('ASR_MALAYALAM_FINAL_MAX_CLEANED_GARBLE_SCORE', '0.45'))
ASR_MALAYALAM_FINAL_MIN_LEXICAL_TRUST = float(os.environ.get('ASR_MALAYALAM_FINAL_MIN_LEXICAL_TRUST', '0.48'))
ASR_MALAYALAM_FINAL_MAX_PSEUDO_PHONETIC_RATIO = float(os.environ.get('ASR_MALAYALAM_FINAL_MAX_PSEUDO_PHONETIC_RATIO', '0.30'))
ASR_MALAYALAM_FINAL_MAX_LOW_TRUST_SEGMENT_SHARE = float(os.environ.get('ASR_MALAYALAM_FINAL_MAX_LOW_TRUST_SEGMENT_SHARE', '0.34'))


def validate_malayalam_settings():
    active_settings = globals().get('settings', None)
    configured_purity = float(
        getattr(active_settings, 'ASR_MALAYALAM_MIN_SCRIPT_PURITY', ASR_MALAYALAM_MIN_SCRIPT_PURITY)
        if active_settings is not None
        else ASR_MALAYALAM_MIN_SCRIPT_PURITY
    )
    if configured_purity < 0.85:
        raise ImproperlyConfigured(
            f"ASR_MALAYALAM_MIN_SCRIPT_PURITY is set to {configured_purity}, "
            f"which is below the minimum allowed value of 0.85. "
            f"Values below 0.85 allow English leakage to pass Malayalam transcript "
            f"validation and break the Malayalam fidelity guarantee. "
            f"If you need to adjust Malayalam quality thresholds, "
            f"see ASR_MALAYALAM_STRATEGY and related settings."
        )
    configured_model = str(
        getattr(active_settings, 'ASR_MALAYALAM_PRIMARY_MODEL', ASR_MALAYALAM_PRIMARY_MODEL)
        if active_settings is not None
        else ASR_MALAYALAM_PRIMARY_MODEL
    ).strip()
    configured_device = str(
        getattr(active_settings, 'ASR_MALAYALAM_DEVICE', ASR_MALAYALAM_DEVICE)
        if active_settings is not None
        else ASR_MALAYALAM_DEVICE
    ).strip().lower()
    should_validate_gpu_vram = configured_device in {'cuda', 'gpu'} or os.environ.get(
        'ASR_VALIDATE_GPU_VRAM_ON_STARTUP', 'False'
    ).lower() in ('true', '1', 'yes')
    if not should_validate_gpu_vram:
        return
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram_gb < 6.0 and configured_model == "large-v3":
                raise ImproperlyConfigured(
                    f"ASR_MALAYALAM_PRIMARY_MODEL=large-v3 requires ~6GB VRAM minimum. "
                    f"Detected {vram_gb:.1f}GB on this GPU. "
                    f"Set ASR_MALAYALAM_PRIMARY_MODEL=large-v2 for the RTX 3050."
                )
    except ImportError:
        pass


from django.conf import settings  # noqa: E402  pylint: disable=wrong-import-position

validate_malayalam_settings()
ASR_MALAYALAM_TOKEN_PLAUSIBILITY_MIN = float(os.environ.get('ASR_MALAYALAM_TOKEN_PLAUSIBILITY_MIN', '0.58'))
ASR_MALAYALAM_FIRST_PASS_BEAM_SIZE = int(os.environ.get('ASR_MALAYALAM_FIRST_PASS_BEAM_SIZE', '2'))
ASR_MALAYALAM_FIRST_PASS_BEST_OF = int(os.environ.get('ASR_MALAYALAM_FIRST_PASS_BEST_OF', '2'))
ASR_MALAYALAM_RETRY_BEAM_SIZE = int(os.environ.get('ASR_MALAYALAM_RETRY_BEAM_SIZE', '6'))
ASR_MALAYALAM_RETRY_BEST_OF = int(os.environ.get('ASR_MALAYALAM_RETRY_BEST_OF', '6'))
ASR_MALAYALAM_RESCUE_QUALITY_BEAM_SIZE = int(os.environ.get('ASR_MALAYALAM_RESCUE_QUALITY_BEAM_SIZE', '6'))
ASR_MALAYALAM_RESCUE_QUALITY_BEST_OF = int(os.environ.get('ASR_MALAYALAM_RESCUE_QUALITY_BEST_OF', '5'))
ASR_MALAYALAM_RESCUE_QUALITY_COMPRESSION_RATIO_THRESHOLD = float(os.environ.get('ASR_MALAYALAM_RESCUE_QUALITY_COMPRESSION_RATIO_THRESHOLD', '2.4'))
ASR_MALAYALAM_RESCUE_QUALITY_LOGPROB_THRESHOLD = float(os.environ.get('ASR_MALAYALAM_RESCUE_QUALITY_LOGPROB_THRESHOLD', '-1.0'))
ASR_MALAYALAM_RESCUE_QUALITY_NO_SPEECH_THRESHOLD = float(os.environ.get('ASR_MALAYALAM_RESCUE_QUALITY_NO_SPEECH_THRESHOLD', '0.6'))
ASR_MALAYALAM_RESCUE_TIGHT_PAD_SECONDS = float(os.environ.get('ASR_MALAYALAM_RESCUE_TIGHT_PAD_SECONDS', '0.35'))
ASR_MALAYALAM_RESCUE_MEDIUM_PAD_SECONDS = float(os.environ.get('ASR_MALAYALAM_RESCUE_MEDIUM_PAD_SECONDS', '0.9'))
ASR_MALAYALAM_RESCUE_WIDE_PAD_SECONDS = float(os.environ.get('ASR_MALAYALAM_RESCUE_WIDE_PAD_SECONDS', '1.5'))
ASR_MALAYALAM_RESCUE_MAX_WINDOW_SECONDS = float(os.environ.get('ASR_MALAYALAM_RESCUE_MAX_WINDOW_SECONDS', '12.0'))
ASR_MALAYALAM_RESCUE_MAX_SEGMENTS = int(os.environ.get('ASR_MALAYALAM_RESCUE_MAX_SEGMENTS', '2'))
ASR_MALAYALAM_RESCUE_MAX_TOTAL_AUDIO_SECONDS = float(os.environ.get('ASR_MALAYALAM_RESCUE_MAX_TOTAL_AUDIO_SECONDS', '30.0'))
ASR_MALAYALAM_RESCUE_HOPELESS_MAX_LEXICAL_TRUST = float(os.environ.get('ASR_MALAYALAM_RESCUE_HOPELESS_MAX_LEXICAL_TRUST', '0.15'))
ASR_MALAYALAM_RESCUE_HOPELESS_MAX_READABILITY = float(os.environ.get('ASR_MALAYALAM_RESCUE_HOPELESS_MAX_READABILITY', '0.22'))
ASR_MALAYALAM_RESCUE_HOPELESS_MIN_GARBLE_SCORE = float(os.environ.get('ASR_MALAYALAM_RESCUE_HOPELESS_MIN_GARBLE_SCORE', '0.10'))
ASR_MALAYALAM_ASSEMBLY_MAX_GAP_SECONDS = float(os.environ.get('ASR_MALAYALAM_ASSEMBLY_MAX_GAP_SECONDS', '1.2'))
ASR_MALAYALAM_ASSEMBLY_MAX_UNIT_SECONDS = float(os.environ.get('ASR_MALAYALAM_ASSEMBLY_MAX_UNIT_SECONDS', '14.0'))
ASR_MALAYALAM_ASSEMBLY_MAX_SEGMENTS_PER_UNIT = int(os.environ.get('ASR_MALAYALAM_ASSEMBLY_MAX_SEGMENTS_PER_UNIT', '4'))
ASR_MALAYALAM_DISPLAY_ENABLE = os.environ.get('ASR_MALAYALAM_DISPLAY_ENABLE', 'True').lower() in ('true', '1', 'yes')
ASR_MALAYALAM_DISPLAY_MAX_LOW_TRUST_VISIBLE_SHARE = float(os.environ.get('ASR_MALAYALAM_DISPLAY_MAX_LOW_TRUST_VISIBLE_SHARE', '0.25'))
ASR_MALAYALAM_DISPLAY_ATTACH_MAX_GAP_SECONDS = float(os.environ.get('ASR_MALAYALAM_DISPLAY_ATTACH_MAX_GAP_SECONDS', '0.8'))
ASR_MALAYALAM_DISPLAY_MAX_UNIT_SECONDS = float(os.environ.get('ASR_MALAYALAM_DISPLAY_MAX_UNIT_SECONDS', '18.0'))
ASR_MALAYALAM_DISPLAY_SUPPRESS_MIN_TRUST = float(os.environ.get('ASR_MALAYALAM_DISPLAY_SUPPRESS_MIN_TRUST', '0.26'))
ASR_MALAYALAM_DISPLAY_COMPACT_REPETITION_THRESHOLD = int(os.environ.get('ASR_MALAYALAM_DISPLAY_COMPACT_REPETITION_THRESHOLD', '2'))
ASR_MALAYALAM_CPU_THREADS = int(os.environ.get('ASR_MALAYALAM_CPU_THREADS', str(max(4, (os.cpu_count() or 4) - 1))))
ASR_MALAYALAM_NUM_WORKERS = int(os.environ.get('ASR_MALAYALAM_NUM_WORKERS', '1'))
WHISPER_BEAM_SIZE = int(os.environ.get('WHISPER_BEAM_SIZE', '5'))
WHISPER_BEST_OF = int(os.environ.get('WHISPER_BEST_OF', '5'))
WHISPER_TEMPERATURE = float(os.environ.get('WHISPER_TEMPERATURE', '0'))
WHISPER_VAD_FILTER = os.environ.get('WHISPER_VAD_FILTER', 'True').lower() in ('true', '1', 'yes')
WHISPER_LONGFORM_SPEED_MODE = os.environ.get('WHISPER_LONGFORM_SPEED_MODE', 'False').lower() in ('true', '1', 'yes')
AUDIO_PREPROCESS_FILTER = os.environ.get(
    'AUDIO_PREPROCESS_FILTER',
    ''  # Empty = no preprocessing for faster extraction. Set to 'highpass=f=80,lowpass=f=7600,dynaudnorm=f=120:g=15,afftdn=nf=-25' for full preprocessing
)
SUMMARIZATION_MODEL = os.environ.get('SUMMARIZATION_MODEL', 'facebook/bart-large-cnn')  # BART fallback
SUMMARIZATION_PROVIDER = os.environ.get('SUMMARIZATION_PROVIDER', 'hf')  # groq | hf
SUMMARIZATION_HF_FALLBACK_TASKS = os.environ.get(
    'SUMMARIZATION_HF_FALLBACK_TASKS',
    'summarization,text2text-generation'
)
GROQ_SUMMARY_MODEL = os.environ.get('GROQ_SUMMARY_MODEL', 'llama-3.3-70b-versatile')
GROQ_SUMMARY_RETRY_ATTEMPTS = int(os.environ.get('GROQ_SUMMARY_RETRY_ATTEMPTS', '3'))
GROQ_SUMMARY_RETRY_BASE_DELAY_SEC = int(os.environ.get('GROQ_SUMMARY_RETRY_BASE_DELAY_SEC', '4'))
GROQ_RATE_LIMIT_COOLDOWN_SEC = int(os.environ.get('GROQ_RATE_LIMIT_COOLDOWN_SEC', '180'))
# Hard cap summary prompt size to avoid Groq 413/TPM request-size errors on long videos.
GROQ_SUMMARY_MAX_INPUT_WORDS = int(os.environ.get('GROQ_SUMMARY_MAX_INPUT_WORDS', '2200'))
SUMMARY_OUTLINE_RETRY_ATTEMPTS = int(os.environ.get('SUMMARY_OUTLINE_RETRY_ATTEMPTS', '1'))
SUMMARY_OUTLINE_MIN_STEPS = int(os.environ.get('SUMMARY_OUTLINE_MIN_STEPS', '4'))
SUMMARY_OUTLINE_MAX_STEPS = int(os.environ.get('SUMMARY_OUTLINE_MAX_STEPS', '7'))
SUMMARY_FULL_MIN_WORDS = int(os.environ.get('SUMMARY_FULL_MIN_WORDS', '200'))
SUMMARY_FULL_MAX_WORDS = int(os.environ.get('SUMMARY_FULL_MAX_WORDS', '260'))
SUMMARY_FULL_MIN_ENTITY_COVERAGE = int(os.environ.get('SUMMARY_FULL_MIN_ENTITY_COVERAGE', '2'))
SUMMARY_SHORT_MIN_WORDS = int(os.environ.get('SUMMARY_SHORT_MIN_WORDS', '35'))
SUMMARY_SHORT_MAX_WORDS = int(os.environ.get('SUMMARY_SHORT_MAX_WORDS', '55'))
SHORT_SUMMARY_SKIP_FULL_RECURSION_WORDS = int(os.environ.get('SHORT_SUMMARY_SKIP_FULL_RECURSION_WORDS', '4000'))
HF_SHORT_MAX_INPUT_WORDS = int(os.environ.get('HF_SHORT_MAX_INPUT_WORDS', '900'))
SUMMARY_BULLET_MIN_COUNT = int(os.environ.get('SUMMARY_BULLET_MIN_COUNT', '5'))
SUMMARY_BULLET_MAX_COUNT = int(os.environ.get('SUMMARY_BULLET_MAX_COUNT', '8'))
SUMMARY_BULLET_WORD_MIN = int(os.environ.get('SUMMARY_BULLET_WORD_MIN', '10'))
SUMMARY_BULLET_WORD_MAX = int(os.environ.get('SUMMARY_BULLET_WORD_MAX', '22'))
HF_BULLET_MAX_INPUT_WORDS = int(os.environ.get('HF_BULLET_MAX_INPUT_WORDS', '1200'))
EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'BAAI/bge-m3')
EMBEDDING_MODEL_FALLBACKS = os.environ.get(
    'EMBEDDING_MODEL_FALLBACKS',
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2,sentence-transformers/all-MiniLM-L6-v2'
)
RAG_PREWARM_EMBEDDING_MODEL = os.environ.get(
    'RAG_PREWARM_EMBEDDING_MODEL',
    'True' if DEBUG else 'False'
).lower() in ('true', '1', 'yes')
EMBED_CANONICAL_LANGUAGE = os.environ.get('EMBED_CANONICAL_LANGUAGE', 'en')
CANONICAL_TRANSLATE_SEGMENTS = os.environ.get('CANONICAL_TRANSLATE_SEGMENTS', 'False').lower() in ('true', '1', 'yes')
CANONICAL_TRANSLATION_MODEL = os.environ.get('CANONICAL_TRANSLATION_MODEL', GROQ_SUMMARY_MODEL)
TRANSLATION_PROVIDER = os.environ.get('TRANSLATION_PROVIDER', 'none').strip().lower()
TRANSLATION_MAX_WORDS_PER_CALL = int(os.environ.get('TRANSLATION_MAX_WORDS_PER_CALL', '1200'))
TRANSLATION_RETRY_ATTEMPTS = int(os.environ.get('TRANSLATION_RETRY_ATTEMPTS', '2'))
TRANSLATION_RETRY_BASE_DELAY_SEC = float(os.environ.get('TRANSLATION_RETRY_BASE_DELAY_SEC', '1.5'))
HIER_SUMMARY_CHUNK_WORDS = int(os.environ.get('HIER_SUMMARY_CHUNK_WORDS', '1000'))
HIER_SUMMARY_OVERLAP_WORDS = int(os.environ.get('HIER_SUMMARY_OVERLAP_WORDS', '150'))
HIER_SUMMARY_MIN_WORDS = int(os.environ.get('HIER_SUMMARY_MIN_WORDS', '800'))
HIER_SUMMARY_ONLY_FULL = os.environ.get('HIER_SUMMARY_ONLY_FULL', 'False').lower() in ('true', '1', 'yes')
RAG_CHUNK_SIZE_WORDS = int(os.environ.get('RAG_CHUNK_SIZE_WORDS', '140'))
RAG_CHUNK_OVERLAP_WORDS = int(os.environ.get('RAG_CHUNK_OVERLAP_WORDS', '35'))
RAG_CHUNK_TARGET_SECONDS = int(os.environ.get('RAG_CHUNK_TARGET_SECONDS', '75'))
RAG_SEMANTIC_CHUNKING = os.environ.get('RAG_SEMANTIC_CHUNKING', 'True').lower() in ('true', '1', 'yes')
RERANKER_MODEL = os.environ.get('RERANKER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
RAG_ENABLE_RERANKER = os.environ.get('RAG_ENABLE_RERANKER', 'True').lower() in ('true', '1', 'yes')
RAG_SEARCH_POOL_MULTIPLIER = int(os.environ.get('RAG_SEARCH_POOL_MULTIPLIER', '4'))
RAG_TOP_K = int(os.environ.get('RAG_TOP_K', '8'))
RAG_TOP_K_SUMMARY = int(os.environ.get('RAG_TOP_K_SUMMARY', '8'))
RAG_TOP_K_FACTUAL = int(os.environ.get('RAG_TOP_K_FACTUAL', '6'))
RAG_TOP_K_TIMELINE = int(os.environ.get('RAG_TOP_K_TIMELINE', '6'))
RAG_TOP_K_QUOTE = int(os.environ.get('RAG_TOP_K_QUOTE', '5'))
RAG_FINAL_CONTEXT_K = int(os.environ.get('RAG_FINAL_CONTEXT_K', '3'))
RAG_SIMILARITY_THRESHOLD = float(os.environ.get('RAG_SIMILARITY_THRESHOLD', '0.30'))
RAG_MAX_CONTEXT_WORDS = int(os.environ.get('RAG_MAX_CONTEXT_WORDS', '1500'))
RAG_MIN_SOURCE_PREVIEW_CHARS = int(os.environ.get('RAG_MIN_SOURCE_PREVIEW_CHARS', '18'))
RAG_MIN_LEXICAL_OVERLAP = float(os.environ.get('RAG_MIN_LEXICAL_OVERLAP', '0.05'))
RAG_MIN_SOURCE_QUALITY = float(os.environ.get('RAG_MIN_SOURCE_QUALITY', '0.16'))
RAG_RETRIEVAL_MIN_USEFUL_RESULTS = int(os.environ.get('RAG_RETRIEVAL_MIN_USEFUL_RESULTS', '1'))
CHAT_ANSWER_QA_ENABLED = os.environ.get('CHAT_ANSWER_QA_ENABLED', 'True').lower() in ('true', '1', 'yes')
CHAT_ANSWER_MIN_INFO_TOKENS = int(os.environ.get('CHAT_ANSWER_MIN_INFO_TOKENS', '6'))
CHAT_KEY_POINTS_MAX = int(os.environ.get('CHAT_KEY_POINTS_MAX', '3'))
CHAT_MOMENT_CONTEXT_BEFORE_SECONDS = int(os.environ.get('CHAT_MOMENT_CONTEXT_BEFORE_SECONDS', '20'))
CHAT_MOMENT_CONTEXT_AFTER_SECONDS = int(os.environ.get('CHAT_MOMENT_CONTEXT_AFTER_SECONDS', '40'))
CHAT_MOMENT_MAX_CHUNKS = int(os.environ.get('CHAT_MOMENT_MAX_CHUNKS', '4'))
CHAT_MOMENT_MAX_WORDS = int(os.environ.get('CHAT_MOMENT_MAX_WORDS', '90'))
CHAT_MOMENT_ALLOW_BROADENING = os.environ.get('CHAT_MOMENT_ALLOW_BROADENING', 'True').lower() in ('true', '1', 'yes')
CHATBOT_USE_GROQ_LLM = os.environ.get('CHATBOT_USE_GROQ_LLM', 'False').lower() in ('true', '1', 'yes')
ENTITY_VALIDATOR_ENABLED = os.environ.get('ENTITY_VALIDATOR_ENABLED', 'True').lower() in ('true', '1', 'yes')
ENTITY_LOW_CONFIDENCE_THRESHOLD = float(os.environ.get('ENTITY_LOW_CONFIDENCE_THRESHOLD', '0.62'))
ENTITY_FUZZY_THRESHOLD = float(os.environ.get('ENTITY_FUZZY_THRESHOLD', '0.85'))
ENTITY_MIN_TOKEN_LEN = int(os.environ.get('ENTITY_MIN_TOKEN_LEN', '4'))
ENTITY_REPLACEMENT_LOG_LIMIT = int(os.environ.get('ENTITY_REPLACEMENT_LOG_LIMIT', '30'))
TRANSCRIPT_QA_LOGGING = os.environ.get('TRANSCRIPT_QA_LOGGING', 'True').lower() in ('true', '1', 'yes')
TRANSCRIPT_QA_LONG_SENTENCE_WARN = int(os.environ.get('TRANSCRIPT_QA_LONG_SENTENCE_WARN', '8'))
TRANSCRIPT_PHRASE_BLACKLIST = os.environ.get(
    'TRANSCRIPT_PHRASE_BLACKLIST',
    'Ask AI about this moment|screen recording|click the bell icon|subscribe to the channel'
)
TRANSCRIPT_REGEX_CLEANUP_PATTERNS = os.environ.get(
    'TRANSCRIPT_REGEX_CLEANUP_PATTERNS',
    '(?i)\\bask ai about this moment\\b||(?i)\\bclick the bell icon\\b||(?i)\\bsubscribe(?: now| to the channel)?\\b'
)
SUMMARY_QUALITY_MIN_SCORE = float(os.environ.get('SUMMARY_QUALITY_MIN_SCORE', '0.62'))
SUMMARY_UNSUPPORTED_TOKEN_RATIO_MAX = float(os.environ.get('SUMMARY_UNSUPPORTED_TOKEN_RATIO_MAX', '0.38'))
SUMMARY_UNSUPPORTED_ENTITY_COUNT_MAX = int(os.environ.get('SUMMARY_UNSUPPORTED_ENTITY_COUNT_MAX', '1'))
GLOBAL_ENTITY_DICT_PATH = os.environ.get(
    'GLOBAL_ENTITY_DICT_PATH',
    str(BASE_DIR / 'videos' / 'global_entity_dictionary.json')
)

# OpenAI Configuration (for chatbot if using GPT)
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
OPENAI_PROPER_NOUN_MODEL = os.environ.get('OPENAI_PROPER_NOUN_MODEL', 'gpt-4o-mini')

# Groq Configuration (free API - https://console.groq.com/)
# Groq provides free API keys with generous free tier
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')

# Ollama Configuration (local models - https://ollama.com/)
# Run locally: ollama serve
OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')

# Logging Configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {asctime} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
        # File logging disabled on Windows due to path issues
        # Enable below for production Linux servers
        # 'file': {
        #     'class': 'logging.handlers.RotatingFileHandler',
        #     'filename': LOG_DIR / 'django.log',
        #     'maxBytes': 1024 * 1024 * 10,
        #     'backupCount': 5,
        #     'formatter': 'verbose',
        # },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
    'loggers': {
        'videos': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'chatbot': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'summarizer': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'sentence_transformers': {
            'handlers': ['console'],
            'level': 'WARNING',
            'propagate': False,
        },
        'transformers': {
            'handlers': ['console'],
            'level': 'WARNING',
            'propagate': False,
        },
        'huggingface_hub': {
            'handlers': ['console'],
            'level': 'WARNING',
            'propagate': False,
        },
        'httpx': {
            'handlers': ['console'],
            'level': 'WARNING',
            'propagate': False,
        },
        'httpcore': {
            'handlers': ['console'],
            'level': 'WARNING',
            'propagate': False,
        },
        '_client': {
            'handlers': ['console'],
            'level': 'WARNING',
            'propagate': False,
        },
        '_http': {
            'handlers': ['console'],
            'level': 'WARNING',
            'propagate': False,
        },
    },
}

if not DEBUG:
    SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_SSL_REDIRECT = os.environ.get('DJANGO_SECURE_SSL_REDIRECT', 'True').lower() in ('true', '1', 'yes')
