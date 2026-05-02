#!/usr/bin/env bash
# Apply the Stage 2b S3 lifecycle policy to s3://cosmo-gas-vision-storage/
# per [D-14]: objects under mlflow-artifacts/ transition to Standard-IA after
# 30 days and to Glacier Deep Archive after 180 days.
#
# Idempotent: the JSON below is the full desired state. Re-running this script
# replaces whatever lifecycle config is currently on the bucket.
#
# Usage:
#   set -a; source D:/Data/sujin/CosmoGasVision/.env; set +a
#   bash D:/Data/sujin/CosmoGasVision/scripts/configure_s3_lifecycle.sh

set -euo pipefail

BUCKET="cosmo-gas-vision-storage"
# Use an in-tree temp file so the path is the same in bash and in the Windows-native
# AWS CLI (mktemp's /tmp/... is unresolvable by aws.exe outside MSYS).
TMP_JSON="${TMPDIR:-${SCRIPT_DIR:-$(dirname "$0")}}/lifecycle.$$.json"
trap 'rm -f "${TMP_JSON}"' EXIT

cat > "${TMP_JSON}" <<'JSON'
{
  "Rules": [
    {
      "ID": "mlflow-artifacts-tiering",
      "Status": "Enabled",
      "Filter": { "Prefix": "mlflow-artifacts/" },
      "Transitions": [
        { "Days": 30,  "StorageClass": "STANDARD_IA" },
        { "Days": 180, "StorageClass": "DEEP_ARCHIVE" }
      ],
      "NoncurrentVersionTransitions": [
        { "NoncurrentDays": 30,  "StorageClass": "STANDARD_IA" },
        { "NoncurrentDays": 180, "StorageClass": "DEEP_ARCHIVE" }
      ],
      "AbortIncompleteMultipartUpload": { "DaysAfterInitiation": 7 }
    }
  ]
}
JSON

echo "Applying lifecycle to s3://${BUCKET}/ ..."
aws s3api put-bucket-lifecycle-configuration \
    --bucket "${BUCKET}" \
    --lifecycle-configuration "file://${TMP_JSON}"

echo
echo "Verifying ..."
aws s3api get-bucket-lifecycle-configuration --bucket "${BUCKET}"
