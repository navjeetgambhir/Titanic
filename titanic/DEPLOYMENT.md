# Deploying Titanic App to Azure

This guide walks through deploying the app using the GitHub Actions pipeline defined in `.github/workflows/deploy.yml`.

---

## Prerequisites

- Azure CLI installed and logged in (`az login`)
- Docker installed locally
- GitHub repository with Actions enabled

---

## Step 1 — Create Azure Resources

### 1a. Resource Group
```bash
az group create --name titanic-rg --location eastus
```

### 1b. Azure Container Registry (ACR)
```bash
az acr create \
  --resource-group titanic-rg \
  --name <your-acr-name> \
  --sku Basic \
  --admin-enabled true
```

Get the login server, username, and password:
```bash
az acr show --name <your-acr-name> --query loginServer --output tsv
az acr credential show --name <your-acr-name>
```

### 1c. Azure Container Apps Environment
```bash
az containerapp env create \
  --name titanic-env \
  --resource-group titanic-rg \
  --location eastus
```

### 1d. Create the Container App (first deploy only)
```bash
az containerapp create \
  --name titanic-app \
  --resource-group titanic-rg \
  --environment titanic-env \
  --image <acr-login-server>/titanic:latest \
  --registry-server <acr-login-server> \
  --registry-username <acr-username> \
  --registry-password <acr-password> \
  --target-port 8000 \
  --ingress external \
  --min-replicas 1
```

---

## Step 2 — Add Secrets to the Container App

The app requires the following secrets set on the Container App:

```bash
az containerapp secret set \
  --name titanic-app \
  --resource-group titanic-rg \
  --secrets \
    openai-api-key=<your-openai-api-key> \
    jwt-secret=<your-jwt-secret> \
    fernet-key=<your-fernet-key> \
    hmac-secret=<your-hmac-secret>
```

---

## Step 3 — Configure GitHub Secrets

In your GitHub repository go to **Settings → Secrets and variables → Actions** and add:

| Secret Name         | Value                                                  |
|---------------------|--------------------------------------------------------|
| `ACR_LOGIN_SERVER`  | e.g. `myregistry.azurecr.io`                           |
| `ACR_USERNAME`      | ACR admin username                                     |
| `ACR_PASSWORD`      | ACR admin password                                     |
| `AZURE_CREDENTIALS` | JSON output from the service principal command below   |

### Create a service principal for GitHub Actions
```bash
az ad sp create-for-rbac \
  --name titanic-github-actions \
  --role contributor \
  --scopes /subscriptions/<subscription-id>/resourceGroups/titanic-rg \
  --sdk-auth
```

Copy the full JSON output into the `AZURE_CREDENTIALS` GitHub secret.

---

## Step 4 — Trigger the Deployment

The pipeline runs automatically on every push to `main`:

```bash
git push origin main
```

Or trigger it manually from **GitHub → Actions → Build and Deploy to Azure Container Apps → Run workflow**.

---

## Pipeline Stages

```
push to main
     │
     ▼
 ┌─────────┐
 │  test   │  runs pytest via uv
 └────┬────┘
      │ pass
      ▼
 ┌─────────┐
 │  build  │  builds Docker image, pushes to ACR
 └────┬────┘   tags: <sha> and :latest
      │
      ▼
 ┌──────────┐
 │  deploy  │  updates Azure Container App with new image
 └──────────┘
```

---

## Environment Variables in the Running App

These are injected at deploy time via secret references:

| Variable        | Source                      |
|-----------------|-----------------------------|
| `OPENAI_API_KEY`| secret `openai-api-key`     |
| `JWT_SECRET`    | secret `jwt-secret`         |
| `FERNET_KEY`    | secret `fernet-key`         |
| `HMAC_SECRET`   | secret `hmac-secret`        |
| `DB_PATH`       | `/app/db/titanic.db`        |

---

## Verify the Deployment

```bash
# Check the app status
az containerapp show \
  --name titanic-app \
  --resource-group titanic-rg \
  --query "properties.latestRevisionFqdn" \
  --output tsv

# Stream live logs
az containerapp logs show \
  --name titanic-app \
  --resource-group titanic-rg \
  --follow
```
