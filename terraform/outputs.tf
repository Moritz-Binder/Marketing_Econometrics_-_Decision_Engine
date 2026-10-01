output "backend_url" {
  value       = google_cloud_run_v2_service.backend.uri
  description = "The URL of the FastAPI backend service"
}

output "frontend_url" {
  value       = google_cloud_run_v2_service.frontend.uri
  description = "The URL of the Streamlit frontend service"
}

output "data_bucket_name" {
  value       = google_storage_bucket.data_bucket.name
  description = "The GCS bucket for data storage"
}
