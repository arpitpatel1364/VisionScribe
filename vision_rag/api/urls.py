from django.contrib import admin
from django.urls import path, include
from rest_framework.authtoken.views import obtain_auth_token
from . import views

urlpatterns = [
    path("admin/", admin.site.urls),

    # Auth
    path("api/auth/token/", obtain_auth_token, name="api_token_auth"),

    # Documents
    path("api/ingest/", views.IngestView.as_view(), name="ingest"),
    path("api/documents/", views.DocumentListView.as_view(), name="document_list"),
    path("api/documents/<uuid:id>/", views.DocumentDetailView.as_view(), name="document_detail"),

    # Query
    path("api/query/", views.QueryView.as_view(), name="query"),

    # Analytics
    path("api/stats/", views.stats_view, name="stats"),
    path("api/logs/", views.QueryLogListView.as_view(), name="query_logs"),
]
