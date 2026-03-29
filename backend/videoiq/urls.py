"""URL configuration for VideoIQ AI Video Intelligence System project."""

from django.http import HttpResponseRedirect, JsonResponse
from django.contrib import admin
from django.urls import path, re_path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import TemplateView
from drf_yasg.views import get_schema_view
from drf_yasg import openapi


def _frontend_available() -> bool:
    return (settings.FRONTEND_DIST_DIR / "index.html").exists()


def spa_or_frontend_redirect(request):
    """
    Serve built SPA from Django when available.
    In DEBUG without build artifacts, fallback to Vite dev server.
    """
    if _frontend_available():
        return TemplateView.as_view(template_name="index.html")(request)
    if settings.DEBUG:
        return HttpResponseRedirect("http://localhost:5173")
    return HttpResponseRedirect("/swagger/")


def healthz(_request):
    return JsonResponse({"status": "ok"})

# API Documentation
schema_view = get_schema_view(
    openapi.Info(
        title="VideoIQ AI Video Intelligence System API",
        default_version='v1',
        description="API for AI-powered video summarization, chatbot, and short video generation",
        terms_of_service="https://www.google.com/policies/terms/",
        contact=openapi.Contact(email="contact@example.com"),
        license=openapi.License(name="MIT License"),
    ),
    public=True,
)

urlpatterns = [
    # Root: serve SPA build if available, else Vite dev redirect in DEBUG.
    path('', spa_or_frontend_redirect, name='root'),
    path('healthz/', healthz, name='healthz'),
    
    # Admin
    path('admin/', admin.site.urls),
    
    # API Documentation
    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
    
    # App URLs with API v1 prefix
    path('api/v1/videos/', include('videos.urls')),
    path('api/v1/chatbot/', include('chatbot.urls')),
    path('api/v1/summarizer/', include('summarizer.urls')),
    path('api/extension/', include('videos.extension_urls')),
]

# SPA fallback route (must be after API routes).
urlpatterns += [
    re_path(
        r'^(?!api/|admin/|swagger/|redoc/|media/|static/).*$',
        spa_or_frontend_redirect,
        name='spa-fallback'
    ),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
