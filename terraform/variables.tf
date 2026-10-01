variable "project_id" {
  type        = string
  description = "The GCP project ID"
  default     = "mede-platform-prod"
}

variable "region" {
  type        = string
  description = "The default GCP region"
  default     = "europe-west3"
}

variable "artifact_repo" {
  type        = string
  description = "Artifact Registry repository name"
  default     = "mede-repo"
}
