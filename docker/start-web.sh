#!/bin/sh
set -e

cd /app/backend

python manage.py migrate --noinput

exec gunicorn videoiq.wsgi:application \
  --bind 0.0.0.0:${PORT:-8000} \
  --workers ${WEB_CONCURRENCY:-1} \
  --threads ${GUNICORN_THREADS:-1} \
  --timeout ${GUNICORN_TIMEOUT:-420}
