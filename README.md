# VideoIQ

VideoIQ is a full-stack AI video analysis workspace built around Django, React, and a Chrome side panel extension. It supports local video uploads, YouTube imports, screen recording from the UI, transcript generation, summary generation, chatbot workflows, and extension-based access to the same application surface.

This README is meant to describe the current repository state and how the system is wired today.

## Overview

The project consists of three main surfaces:

- a Django backend that owns ingestion, processing, persistence, and APIs
- a React frontend that provides the main user interface
- a Chrome extension that opens the app in compact panel mode and also exposes compatibility endpoints for extension-specific workflows

At a high level, the product flow is:

1. create a video record from a file upload, YouTube URL, or screen capture flow
2. extract and preprocess audio
3. generate and clean a transcript
4. build summaries and chat-ready representations
5. expose results in the browser UI and extension panel

## Current Capabilities

The repository currently supports:

- local video uploads
- YouTube URL ingestion
- screen recording from the frontend
- transcript generation with timestamp data
- transcript cleaning and canonicalization stages
- multiple summary representations
- chatbot interaction over processed video content
- progress tracking through backend processing states
- Chrome extension side panel access
- extension compatibility endpoints for submit, status, result, and chat flows

## Repository Layout

```text
AI Video Summarizer/
|-- backend/
|   |-- videoiq/                    # Django project config, settings, root routes
|   |-- videos/                     # Video ingestion, transcription, summaries, processing
|   |-- chatbot/                    # Chat sessions, retrieval, chatbot APIs
|   |-- summarizer/                 # Text summary API surface
|   |-- manage.py
|   `-- requirements.txt
|-- frontend/
|   |-- src/
|   |   |-- components/             # Shared UI building blocks
|   |   |-- constants/              # Frontend constants
|   |   |-- context/                # React context providers
|   |   |-- pages/                  # Page-level views
|   |   `-- services/               # API client helpers
|   |-- package.json
|   `-- vite.config.js
|-- extension/                      # Chrome Manifest V3 side panel extension
|-- README.md
`-- .gitignore
```

## Backend Architecture

### Django Project

The active Django project package is [backend/videoiq](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/backend/videoiq).

Important files:

- [backend/videoiq/settings.py](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/backend/videoiq/settings.py)
- [backend/videoiq/urls.py](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/backend/videoiq/urls.py)
- [backend/videoiq/celery.py](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/backend/videoiq/celery.py)

The backend is configured to:

- serve the built frontend through Django when `backend/frontend_dist/index.html` exists
- redirect to the Vite dev server in development when a build is not present
- expose API routes under `/api/v1/...`
- expose extension compatibility routes under `/api/extension/...`
- optionally allow iframe embedding for the extension in debug-friendly setups

### Core Apps

#### `videos`

The `videos` app is the core processing app. It owns:

- video records
- transcript records
- summary records
- highlight and short-video related data
- upload views
- YouTube processing launch flow
- transcript and summary generation pipeline
- extension API compatibility helpers

Key files:

- [backend/videos/models.py](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/backend/videos/models.py)
- [backend/videos/views.py](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/backend/videos/views.py)
- [backend/videos/tasks.py](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/backend/videos/tasks.py)
- [backend/videos/utils.py](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/backend/videos/utils.py)
- [backend/videos/asr_router.py](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/backend/videos/asr_router.py)
- [backend/videos/audio_preprocessor.py](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/backend/videos/audio_preprocessor.py)
- [backend/videos/summary_schema.py](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/backend/videos/summary_schema.py)
- [backend/videos/translation.py](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/backend/videos/translation.py)
- [backend/videos/canonical.py](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/backend/videos/canonical.py)

#### `chatbot`

The `chatbot` app owns:

- chat session storage
- chat messages
- chat APIs
- retrieval/indexing logic for video-aware chat

Key files:

- [backend/chatbot/models.py](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/backend/chatbot/models.py)
- [backend/chatbot/views.py](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/backend/chatbot/views.py)
- [backend/chatbot/rag_engine.py](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/backend/chatbot/rag_engine.py)

#### `summarizer`

The `summarizer` app provides a simpler text summarization API surface.

Key files:

- [backend/summarizer/views.py](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/backend/summarizer/views.py)
- [backend/summarizer/urls.py](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/backend/summarizer/urls.py)

## Data Model Summary

The main `videos` app models include:

- `Video`
- `Transcript`
- `Summary`
- `HighlightSegment`
- `ShortVideo`
- `ProcessingTask`

From [backend/videos/models.py](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/backend/videos/models.py), the current `Video` processing statuses include:

- `pending`
- `uploaded`
- `processing`
- `extracting_audio`
- `transcribing`
- `cleaning_transcript`
- `transcript_ready`
- `summarizing_quick`
- `summarizing_final`
- `summarizing`
- `indexing_chat`
- `completed`
- `failed`

That status model is useful when debugging why a video appears stuck in the UI.

## API Surface

### Root Routes

Configured in [backend/videoiq/urls.py](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/backend/videoiq/urls.py):

- `/`
- `/admin/`
- `/swagger/`
- `/redoc/`
- `/api/v1/videos/`
- `/api/v1/chatbot/`
- `/api/v1/summarizer/`
- `/api/extension/`

### Video Routes

Configured in [backend/videos/urls.py](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/backend/videos/urls.py):

- `POST /api/v1/videos/upload/`
- `POST /api/v1/videos/youtube/`
- `GET /api/v1/videos/`
- router-backed video routes under `/api/v1/videos/`
- router-backed transcript routes under `/api/v1/videos/transcripts/`

### Chatbot Routes

Configured in [backend/chatbot/urls.py](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/backend/chatbot/urls.py):

- `POST /api/v1/chatbot/chat/`
- session routes under `/api/v1/chatbot/sessions/`
- index routes under `/api/v1/chatbot/indices/`

### Summarizer Routes

Configured in [backend/summarizer/urls.py](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/backend/summarizer/urls.py):

- `POST /api/v1/summarizer/summarize/`
- `GET /api/v1/summarizer/health/`

### Extension Compatibility Routes

Configured in [backend/videos/extension_urls.py](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/backend/videos/extension_urls.py):

- `POST /api/extension/summarize`
- `GET /api/extension/status`
- `GET /api/extension/result`
- `POST /api/extension/chat`

These routes are implemented in [backend/videos/extension_views.py](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/backend/videos/extension_views.py) and provide a stable contract for extension clients while reusing the core pipeline.

## Frontend Architecture

The frontend is a React 18 + Vite app under [frontend](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/frontend).

Important files:

- [frontend/src/App.jsx](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/frontend/src/App.jsx)
- [frontend/src/pages/HomePage.jsx](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/frontend/src/pages/HomePage.jsx)
- [frontend/src/pages/VideoDetailPage.jsx](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/frontend/src/pages/VideoDetailPage.jsx)
- [frontend/src/services/api.js](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/frontend/src/services/api.js)

The frontend currently supports:

- homepage video list and progress polling
- file upload flow
- YouTube URL submission flow
- screen recording flow
- video detail pages
- transcript and summary presentation
- chatbot interactions
- panel mode layout via `?layout=panel`

### Panel Mode

Panel mode is enabled by query string in [frontend/src/App.jsx](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/frontend/src/App.jsx):

- `/?layout=panel`

This lets the browser extension reuse the same frontend with a more compact layout instead of maintaining a separate UI application.

## Chrome Extension

The extension lives under [extension](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/extension).

Key files:

- [extension/manifest.json](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/extension/manifest.json)
- [extension/background.js](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/extension/background.js)
- [extension/sidepanel.js](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/extension/sidepanel.js)
- [extension/options.js](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/extension/options.js)
- [extension/README.md](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/extension/README.md)

The extension is intentionally lightweight:

- it stores a configurable backend URL
- it opens the app in the Chrome side panel
- it loads the same frontend instead of maintaining a duplicate React build

Default backend URL:

- `http://127.0.0.1:8000`

## Local Development Setup

### Prerequisites

- Python 3.12+
- Node.js 18+
- FFmpeg installed and available on `PATH`
- Redis if you want Celery workers

### Backend Setup

```powershell
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python manage.py migrate
python manage.py runserver
```

Backend default local URL:

- `http://127.0.0.1:8000`

### Optional Celery Worker

If you want background processing through Celery:

```powershell
cd backend
celery -A videoiq worker -l info
```

The current Celery app points at `videoiq`, not the older project package name.

### Frontend Setup

```powershell
cd frontend
npm install
npm run dev
```

Frontend dev URL:

- `http://localhost:5173`

In development, Django redirects `/` to the Vite server when a built frontend is not present.

### Frontend Build For Django

If you want Django to serve the frontend directly:

```powershell
cd frontend
npm install
npm run build
```

The backend expects build artifacts under:

- `backend/frontend_dist/`

### Extension Setup

To load the extension:

1. open `chrome://extensions`
2. enable Developer mode
3. click Load unpacked
4. select [extension](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/extension)

Make sure the backend is running first.

## Environment And Runtime Notes

Important settings behavior from [backend/videoiq/settings.py](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/backend/videoiq/settings.py):

- the backend loads `.env` from `backend/.env`
- SQLite is used by default for local development
- media lives under `backend/media`
- static build artifacts are expected under `backend/frontend_dist`
- `ALLOW_EXTENSION_IFRAME` is enabled by default in debug-friendly development mode
- Celery broker and result backend default to Redis on `localhost:6379`
- a development sync mode flag exists for local processing without a separate worker

## Typical Development Workflow

### Web App

1. start the backend
2. start the frontend Vite server if you are working on the UI
3. upload a file or submit a YouTube URL
4. watch progress states update in the homepage list
5. open the video detail page
6. inspect transcript, summaries, and chat behavior

### Extension

1. start the backend
2. load the unpacked extension
3. open the side panel
4. confirm the panel loads `/?layout=panel`
5. test upload, detail, and chat flows in compact mode

## Git And Repo Hygiene

The repo now ignores the main categories of local-only clutter:

- `.env` files
- local databases
- uploaded media
- vector indices
- debug audio
- evaluation output
- benchmark output
- backup folders
- editor folders
- generated audio and video files
- `node_modules`

Examples already covered in [.gitignore](c:/Users/Saxon%20sibi/Downloads/AI%20Video%20Summarizer/.gitignore):

- `backend/media/`
- `backend/vector_indices/`
- `backend/videos/eval_results/`
- `backend/benchmark_reports/`
- `backend/frontend_dist/`
- `tmp_test/`
- `RESTORE_PREPROD_BACKUP/`
- `.vscode/`
- `*.mp3`
- `*.mp4`
- `*.wav`
- `STUDY_GUIDE_CURRENT_SYSTEM.md`

### What Should Be Committed

Commit:

- source code in `backend/`
- source code in `frontend/src/`
- extension code in `extension/`
- migrations
- package manifests and requirements
- shared docs you want in the repository

Do not commit:

- `.env`
- local media
- local database files
- generated benchmark or eval outputs
- temporary scratch files
- editor-local state

## Troubleshooting

### The frontend does not load from Django

Check:

- whether `backend/frontend_dist/index.html` exists
- whether Vite is running on `http://localhost:5173`
- whether Django is redirecting to the dev server in `DEBUG`

### The extension panel opens but looks wrong

Check:

- whether the backend URL in extension options is correct
- whether the extension is loading `/?layout=panel`
- whether Django allows iframe embedding in your current environment

### Video processing appears stuck

Check:

- current `Video.status`
- `processing_progress`
- Django server logs
- Celery worker logs if you are not using local dev sync mode
- whether FFmpeg and Redis are installed correctly for your chosen setup