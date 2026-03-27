# VideoIQ

VideoIQ is an AI-powered video workspace for turning long-form video into transcripts, summaries, and chat-ready knowledge. It combines a Django backend, a React frontend, and a lightweight Chrome side panel extension so the same system can be used in the browser or inside an extension workflow.

## What It Does

VideoIQ is designed to help you ingest video content, process it into structured text, and interact with the result.

Current capabilities include:

- local video upload
- YouTube URL ingestion
- screen recording from the frontend
- transcript generation with timestamped segments
- cleaned transcript and summary generation
- chatbot workflows grounded on processed video content
- progress-aware processing states in the UI
- Chrome side panel access through an extension wrapper

## Product Flow

At a high level, the system works like this:

1. a user uploads a video, submits a YouTube URL, or records the screen
2. the backend creates a video record and starts processing
3. audio is extracted and prepared for transcription
4. transcript data is generated, cleaned, and stored
5. summary and chat-ready artifacts are produced
6. the results are shown in the web app and can also be accessed from the extension

## Tech Stack

### Backend

- Python 3.12
- Django
- Django REST Framework
- Celery
- Redis
- SQLite for local development
- FFmpeg for media processing

### Frontend

- React 18
- Vite
- React Router
- Axios
- Tailwind CSS
- Framer Motion

### Extension

- Chrome Extension Manifest V3
- Side panel wrapper around the main app

## Repository Structure

```text
AI Video Summarizer/
|-- backend/
|   |-- videoiq/              # Django project config
|   |-- videos/               # Video ingestion and processing pipeline
|   |-- chatbot/              # Chat sessions, retrieval, chatbot APIs
|   |-- summarizer/           # Summary API surface
|   |-- manage.py
|   `-- requirements.txt
|-- frontend/
|   |-- src/
|   |   |-- components/
|   |   |-- constants/
|   |   |-- context/
|   |   |-- pages/
|   |   `-- services/
|   |-- package.json
|   `-- vite.config.js
|-- extension/               # Chrome side panel extension
|-- README.md
`-- .gitignore
```

## Core Backend Apps

### `videoiq`

The active Django project package. It contains:

- settings
- root URL configuration
- WSGI and ASGI entry points
- Celery app configuration

### `videos`

The main application for:

- file upload and YouTube import
- processing orchestration
- transcript persistence
- summary persistence
- processing metadata and support utilities
- extension compatibility endpoints

### `chatbot`

The chat system for:

- chat sessions
- chat messages
- retrieval and ranking logic
- chatbot endpoints tied to video content

### `summarizer`

A smaller API surface dedicated to text summarization endpoints.

## Main Data Objects

The core objects in the current system include:

- `Video`
- `Transcript`
- `Summary`
- `HighlightSegment`
- `ShortVideo`
- `ProcessingTask`

The `Video` model tracks the current processing stage, including statuses such as:

- `pending`
- `processing`
- `extracting_audio`
- `transcribing`
- `cleaning_transcript`
- `transcript_ready`
- `summarizing`
- `indexing_chat`
- `completed`
- `failed`

These states are reflected in the frontend so users can follow processing progress.

## API Overview

The main route groups exposed by the backend are:

- `/api/v1/videos/`
- `/api/v1/chatbot/`
- `/api/v1/summarizer/`
- `/api/extension/`
- `/swagger/`
- `/redoc/`

Examples of current endpoints include:

- `POST /api/v1/videos/upload/`
- `POST /api/v1/videos/youtube/`
- `GET /api/v1/videos/`
- `POST /api/v1/chatbot/chat/`
- `POST /api/v1/summarizer/summarize/`
- `POST /api/extension/summarize`
- `GET /api/extension/status`
- `GET /api/extension/result`
- `POST /api/extension/chat`

## Frontend

The frontend is a React + Vite application that provides:

- the main dashboard
- upload and import flows
- processing progress views
- transcript and summary presentation
- chatbot interaction
- panel-mode rendering for the extension

The app supports a compact layout mode through:

- `/?layout=panel`

That allows the extension to load the same application with a side-panel-friendly presentation instead of maintaining a separate UI codebase.

## Chrome Extension

The extension is intentionally lightweight. It does not reimplement the product. Instead, it wraps the existing application and opens it in Chrome’s side panel.

The extension provides:

- backend URL configuration
- side panel entry point
- a compact way to use the existing frontend inside Chrome

By default, it points to:

- `http://127.0.0.1:8000`

## Local Setup

### Prerequisites

- Python 3.12+
- Node.js 18+
- FFmpeg installed and available on `PATH`
- Redis if you want Celery-based background processing

## Backend Setup

```powershell
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python manage.py migrate
python manage.py runserver
```

Default backend URL:

- `http://127.0.0.1:8000`

## Optional Celery Worker

If you want worker-based background processing:

```powershell
cd backend
celery -A videoiq worker -l info
```

Make sure Redis is running first.

## Frontend Setup

```powershell
cd frontend
npm install
npm run dev
```

Default frontend development URL:

- `http://localhost:5173`

In development, Django redirects to the Vite dev server when a built frontend is not available.

## Frontend Build For Django

To build the frontend for Django-served usage:

```powershell
cd frontend
npm install
npm run build
```

The backend expects built assets under:

- `backend/frontend_dist/`

## Extension Setup

To load the extension in Chrome:

1. open `chrome://extensions`
2. enable Developer mode
3. click Load unpacked
4. select the `extension/` folder

Make sure the backend is already running before opening the panel.

## Deploying To Render

This repository is now prepared for a Docker-based Render deployment.

Deployment files:

- `Dockerfile`
- `.dockerignore`
- `render.yaml`

Recommended Render layout:

- one web service for Django
- one worker service for Celery
- one Postgres database
- one Redis-compatible Key Value instance

The safest path for this repo is Docker-based deployment because the app depends on FFmpeg, Python native dependencies, frontend build output, and background processing.

Before deploying on Render, make sure you set or verify:

- `DJANGO_SECRET_KEY`
- `DJANGO_DEBUG=False`
- `DJANGO_ALLOWED_HOSTS`
- `DJANGO_CSRF_TRUSTED_ORIGINS`
- `CORS_ALLOWED_ORIGINS`
- `DATABASE_URL`
- `CELERY_BROKER_URL`
- `CELERY_RESULT_BACKEND`
- any API provider keys you want enabled in production

Important production note:

- uploaded media stored on the web service filesystem is ephemeral on Render unless you add external object storage or another persistent media strategy

## Development Notes

Current runtime behavior includes:

- Django serves the built frontend when present
- Django falls back to the Vite dev server in development
- media files are stored under `backend/media`
- the extension can embed the app through iframe-friendly development settings
- Celery and Redis can be used for background processing
- local development can also be configured to run processing without a separate worker

## Git Hygiene

The repository is configured to ignore local-only and generated files such as:

- `.env`
- local database files
- uploaded media
- vector indices
- generated benchmark and evaluation output
- temporary files
- editor-local folders
- generated audio and video files
- `node_modules`

In practice, commit:

- backend source code
- frontend source code
- extension source code
- migrations
- dependency manifest changes
- shared project documentation you want public

Do not commit:

- secrets
- local media and debug artifacts
- local databases
- cache directories
- scratch files

## Troubleshooting

### The frontend does not load through Django

Check:

- whether `backend/frontend_dist/index.html` exists
- whether the Vite dev server is running
- whether Django is in development mode and redirecting correctly

### The extension opens but the UI looks wrong

Check:

- whether the extension backend URL is correct
- whether the app is loading with `?layout=panel`
- whether the backend is allowing extension iframe embedding in your current environment

### Video processing appears stuck

Check:

- the current video status in the UI or API
- backend logs
- Celery worker logs, if you are using Celery
- Redis availability
- FFmpeg installation

## Roadmap-Friendly Areas

This repository is already structured to support continued work in:

- transcript quality improvements
- richer summary schemas
- stronger chat retrieval
- extension-specific flows
- operational tooling around benchmarking and backfills

## License

Add your preferred license information here if you want the repository to state it explicitly.
