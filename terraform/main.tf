terraform {
  required_version = ">= 1.3.0"
  backend "gcs" {
    bucket = "mede-platform-prod-tfstate"
    prefix = "terraform/state"
  }
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# --- Service Accounts ---
resource "google_service_account" "backend_sa" {
  account_id   = "mede-backend-sa"
  display_name = "MEDE Backend Service Account"
}

resource "google_service_account" "frontend_sa" {
  account_id   = "mede-frontend-sa"
  display_name = "MEDE Frontend Service Account"
}

# --- Secret Manager ---
data "google_secret_manager_secret" "gemini_api_key" {
  secret_id = "mede-gemini-api-key"
}

data "google_secret_manager_secret" "langfuse_secret_key" {
  secret_id = "mede-langfuse-secret-key"
}

resource "google_secret_manager_secret_iam_member" "backend_gemini_access" {
  secret_id = data.google_secret_manager_secret.gemini_api_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.backend_sa.email}"
}

resource "google_secret_manager_secret_iam_member" "backend_langfuse_access" {
  secret_id = data.google_secret_manager_secret.langfuse_secret_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.backend_sa.email}"
}

# --- GCS Bucket for Parquet Data ---
resource "google_storage_bucket" "data_bucket" {
  name                        = "${var.project_id}-data"
  location                    = var.region
  force_destroy               = true
  uniform_bucket_level_access = true
}

resource "google_storage_bucket_iam_member" "backend_bucket_access" {
  bucket = google_storage_bucket.data_bucket.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.backend_sa.email}"
}

# --- Cloud Run Services ---

# Note: Before applying this, ensure the images exist in Artifact Registry,
# or uncomment a dummy image approach. We assume images are pushed via CI/CD.
resource "google_cloud_run_v2_service" "backend" {
  name     = "mede-backend"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL" # Accessible via IAM, not public

  template {
    service_account = google_service_account.backend_sa.email
    
    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.artifact_repo}/mede-backend:latest"
      
      env {
        name  = "PROJECT_NAME"
        value = "Marketing Econometrics & Decision Engine (MEDE)"
      }
      env {
        name  = "GOLD_DATA_PATH"
        value = "gs://${google_storage_bucket.data_bucket.name}/marketing_mix_gold.parquet"
      }
      
      env {
        name = "GEMINI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = data.google_secret_manager_secret.gemini_api_key.secret_id
            version = "latest"
          }
        }
      }
      env {
        name = "LANGFUSE_SECRET_KEY"
        value_source {
          secret_key_ref {
            secret  = data.google_secret_manager_secret.langfuse_secret_key.secret_id
            version = "latest"
          }
        }
      }
    }
  }
}

resource "google_cloud_run_v2_service" "frontend" {
  name     = "mede-frontend"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.frontend_sa.email
    
    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.artifact_repo}/mede-frontend:latest"
      
      env {
        name  = "API_BASE_URL"
        value = "${google_cloud_run_v2_service.backend.uri}/api/v1"
      }
    }
  }
}

# --- IAM Constraints for Cloud Run ---
# Restrict backend access ONLY to the frontend service account.
resource "google_cloud_run_service_iam_member" "frontend_invokes_backend" {
  location = google_cloud_run_v2_service.backend.location
  project  = google_cloud_run_v2_service.backend.project
  service  = google_cloud_run_v2_service.backend.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.frontend_sa.email}"
}

# Allow public access to the frontend application
resource "google_cloud_run_service_iam_member" "public_frontend" {
  location = google_cloud_run_v2_service.frontend.location
  project  = google_cloud_run_v2_service.frontend.project
  service  = google_cloud_run_v2_service.frontend.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
