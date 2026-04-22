#!/usr/bin/env bash
# scripts/deploy_azure.sh
# Deploy PharmIQ API to Azure Container Apps
# Usage: bash scripts/deploy_azure.sh
# Prerequisites: az login, Docker Desktop running

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
RESOURCE_GROUP="rg-ml-pharmiq"
LOCATION="eastus"
ACR_NAME="mlpharmiqacr"
APP_NAME="pharmiq-api"
ENVIRONMENT_NAME="pharmiq-env"
IMAGE_TAG="latest"
IMAGE_NAME="${ACR_NAME}.azurecr.io/pharmiq-api:${IMAGE_TAG}"

echo "=== PharmIQ Azure Container Apps Deploy ==="
echo "Resource Group : $RESOURCE_GROUP"
echo "ACR            : $ACR_NAME"
echo "App            : $APP_NAME"
echo ""

# ── Step 1: Resource group ────────────────────────────────────────────────────
echo "[1/6] Creating resource group..."
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output none

# ── Step 2: ACR ───────────────────────────────────────────────────────────────
echo "[2/6] Creating Azure Container Registry..."
az acr create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$ACR_NAME" \
  --sku Basic \
  --admin-enabled true \
  --output none

# ── Step 3: Docker build + push ───────────────────────────────────────────────
echo "[3/6] Building and pushing Docker image..."
az acr build \
  --registry "$ACR_NAME" \
  --image "pharmiq-api:${IMAGE_TAG}" \
  --file docker/Dockerfile \
  .

# ── Step 4: Container Apps environment ───────────────────────────────────────
echo "[4/6] Creating Container Apps environment..."
az containerapp env create \
  --name "$ENVIRONMENT_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --output none

# ── Step 5: Get ACR credentials ───────────────────────────────────────────────
ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" --query "passwords[0].value" -o tsv)

# ── Step 6: Deploy Container App ──────────────────────────────────────────────
echo "[5/6] Deploying Container App..."
az containerapp create \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --environment "$ENVIRONMENT_NAME" \
  --image "$IMAGE_NAME" \
  --registry-server "${ACR_NAME}.azurecr.io" \
  --registry-username "$ACR_NAME" \
  --registry-password "$ACR_PASSWORD" \
  --target-port 8000 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 3 \
  --cpu 1.0 \
  --memory 2.0Gi \
  --output none

# ── Step 7: Get URL ───────────────────────────────────────────────────────────
FQDN=$(az containerapp show \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query "properties.configuration.ingress.fqdn" \
  -o tsv)

echo ""
echo "=== Deploy Complete ==="
echo "API URL  : https://${FQDN}"
echo "Docs URL : https://${FQDN}/docs"
echo "Health   : https://${FQDN}/health"
