# OpenHands Agent Canvas — Kubernetes Deployment Guide

> This is a DIY deployment of the OSS (MIT-licensed) Agent Canvas on Kubernetes. No Enterprise license required.

---

## 1. Architecture Overview

Agent Canvas is **one container** containing **three internal services**:

| Service | Internal Port | Role |
|---|---|---|
| Ingress Proxy / Frontend | `8000` (unified entry) | Static React SPA + reverse proxy to backends |
| Agent Server | `18000` (internal) | REST API + WebSocket for agent conversations |
| Automation Server | `18001` (internal) | Cron schedules, webhooks, event-driven workflows |

All three are exposed through a single port (`8000`) with path-based routing:
- `/*` → Static frontend (SPA)
- `/api/*`, `/sockets` → Agent Server
- `/api/automation/*` → Automation Backend

**Resources:** 2 vCPU / 4 GB RAM minimum (single user). No external databases required.

---

## 2. Docker Image

```
ghcr.io/openhands/agent-canvas:latest
```

Pre-built images are published to GitHub Container Registry. Versioned tags also exist (e.g., `:1.0.0-rc.11`).

---

## 3. Environment Variables

| Variable | Purpose | Required | Default |
|---|---|---|---|
| `PORT` | Unified ingress port | No | `8000` |
| `AGENT_SERVER_PORT` | Internal agent server port | No | `18000` |
| `AUTOMATION_PORT` | Internal automation port | No | `18001` |
| `LOCAL_BACKEND_API_KEY` | API key for authentication | Yes for public mode | Auto-generated on first start |
| `OH_SECRET_KEY` | Secret for settings/secrets encryption | No | Auto-generated on first start |
| `OH_PERSISTENCE_DIR` | Parent directory for state | No | `/home/openhands/.openhands` |
| `OH_CONVERSATIONS_PATH` | Conversation history storage | No | `~/.openhands/agent-canvas/conversations` |
| `OH_BASH_EVENTS_DIR` | Bash event log storage | No | `~/.openhands/agent-canvas/bash_events` |
| `AUTOMATION_DB_URL` | Database connection string | No | `sqlite+aiosqlite:///~/.openhands/automation/automations.db` |
| `AUTOMATION_BASE_URL` | Public callback URL for webhooks | Yes for webhooks | `http://127.0.0.1:$PORT` |
| `AUTOMATION_WORKSPACE_BASE` | Automation run workspace dir | No | `~/.openhands/workspaces` |
| `AUTOMATION_AGENT_SERVER_URL` | URL to agent server internally | No | `http://127.0.0.1:$AGENT_SERVER_PORT` |
| `FILE_STORE` | Storage backend for automation tarballs | No | `local` |
| `LOCAL_STORAGE_PATH` | Local file storage directory | No | `~/.openhands/storage` |
| `OH_AGENT_SERVER_VERSION` | Pin agent server version | No | Latest |

### LLM Agent Credentials (pass as env vars to agents, not to Agent Canvas)

| Agent | Env Var | Purpose |
|---|---|---|
| Claude Code | `ANTHROPIC_API_KEY` | Claude API access |
| Codex | `OPENAI_API_KEY` | OpenAI API access |
| Gemini CLI | `GEMINI_API_KEY` | Google API access |

These are set inside the container or on the host where agents run. Agent Canvas itself does not require them — LLM configuration is done through the Settings UI.

---

## 4. Volume Mounts

Two volume mounts are required for full functionality:

| Container Path | Purpose | Must Be Persistent |
|---|---|---|
| `/home/openhands/.openhands` | Settings, secrets, conversation history, automation DB, API keys | **Yes** |
| `/projects` | User project code — the workspace the agent reads/edits | **Yes** |

---

## 5. Health Checks

All exposed through port `8000`:

| Endpoint | Purpose | Kubernetes Use |
|---|---|---|
| `GET /alive` | Liveness — returns 200 if process is running | `livenessProbe` |
| `GET /health` | Health — returns 200 if healthy | Optional |
| `GET /ready` | Readiness — returns 200 only after init completes (503 during startup) | `readinessProbe` |

---

## 6. Complete Kubernetes Manifests

### Minimal Deployment + Service

```yaml
# k8s/agent-canvas.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: agent-canvas
---
apiVersion: v1
kind: Secret
metadata:
  name: agent-canvas-secrets
  namespace: agent-canvas
stringData:
  # Generate with: openssl rand -base64 32
  LOCAL_BACKEND_API_KEY: "<replace-with-random-string>"
  OH_SECRET_KEY: "<replace-with-random-string>"
  # Optional: pre-configure LLM keys so agents have them inside the container
  ANTHROPIC_API_KEY: "<your-anthropic-api-key>"
  OPENAI_API_KEY: "<your-openai-api-key>"
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: agent-canvas-state
  namespace: agent-canvas
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: agent-canvas-projects
  namespace: agent-canvas
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-canvas
  namespace: agent-canvas
  labels:
    app: agent-canvas
spec:
  replicas: 1
  strategy:
    type: Recreate  # Important: single replica for stateful app
  selector:
    matchLabels:
      app: agent-canvas
  template:
    metadata:
      labels:
        app: agent-canvas
    spec:
      containers:
        - name: agent-canvas
          image: ghcr.io/openhands/agent-canvas:latest
          imagePullPolicy: Always
          ports:
            - containerPort: 8000
              name: http
              protocol: TCP
          env:
            - name: PORT
              value: "8000"
            - name: LOCAL_BACKEND_API_KEY
              valueFrom:
                secretKeyRef:
                  name: agent-canvas-secrets
                  key: LOCAL_BACKEND_API_KEY
            - name: OH_SECRET_KEY
              valueFrom:
                secretKeyRef:
                  name: agent-canvas-secrets
                  key: OH_SECRET_KEY
            - name: AUTOMATION_BASE_URL
              # Replace with your actual public URL (for webhook callbacks)
              value: "http://canvas.example.com"
            - name: ANTHROPIC_API_KEY
              valueFrom:
                secretKeyRef:
                  name: agent-canvas-secrets
                  key: ANTHROPIC_API_KEY
            # Optional: pin agent server version for reproducible deployments
            # - name: OH_AGENT_SERVER_VERSION
            #   value: "1.28.1"
          volumeMounts:
            - name: state
              mountPath: /home/openhands/.openhands
            - name: projects
              mountPath: /projects
          livenessProbe:
            httpGet:
              path: /alive
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 30
            timeoutSeconds: 5
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10
            timeoutSeconds: 5
            # Allow some startup time (agent server takes a moment to initialize)
            failureThreshold: 10
          resources:
            requests:
              cpu: "1"
              memory: "2Gi"
            limits:
              cpu: "2"
              memory: "4Gi"
      volumes:
        - name: state
          persistentVolumeClaim:
            claimName: agent-canvas-state
        - name: projects
          persistentVolumeClaim:
            claimName: agent-canvas-projects
---
apiVersion: v1
kind: Service
metadata:
  name: agent-canvas
  namespace: agent-canvas
spec:
  selector:
    app: agent-canvas
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
      name: http
  type: ClusterIP
```

### Ingress (with TLS)

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: agent-canvas
  namespace: agent-canvas
  annotations:
    # Use your ingress controller's annotations (nginx, traefik, cert-manager, etc.)
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"
    # WebSocket support is required for live agent events (SSE + WS)
    nginx.ingress.kubernetes.io/proxy-set-headers: "Connection upgrade"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600s"
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - canvas.example.com
      secretName: agent-canvas-tls
  rules:
    - host: canvas.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: agent-canvas
                port:
                  number: 80
```

### If Using MetalLB or LoadBalancer

```yaml
apiVersion: v1
kind: Service
metadata:
  name: agent-canvas-lb
  namespace: agent-canvas
spec:
  selector:
    app: agent-canvas
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
  # MetalLB annotation if using MetalLB:
  # metadata.annotations:
  #   load-balancer.org/ip: "10.0.0.50"
```

---

## 7. PostgreSQL Option (Optional — For Production)

The default SQLite works fine for single-user. For multi-user or higher load, use PostgreSQL:

```yaml
# k8s/postgres-secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: postgres-creds
  namespace: agent-canvas
stringData:
  POSTGRES_USER: agentcanvas
  POSTGRES_PASSWORD: "<strong-password>"
  POSTGRES_DB: agentcanvas
---
# Add this to the Agent Canvas deployment env:
# - name: AUTOMATION_DB_URL
#   value: "postgresql+asyncpg://agentcanvas:<password>@postgres:5432/agentcanvas"
```

The Docker image already includes `libpq-dev` and `asyncpg` — no extra dependencies needed.

---

## 8. Horizontal Pod Autoscaler (Optional)

**Not recommended** — Agent Canvas is stateful (conversations, SQLite DB). Running multiple replicas requires an external database and shared file storage.

If you want multiple agent backends, the correct pattern is **one Agent Canvas + multiple backend agents** (on separate machines/pods), not multiple Agent Canvas replicas.

---

## 9. Accessing the UI

After applying the manifests:

```bash
# Port-forward for local access:
kubectl port-forward -n agent-canvas svc/agent-canvas 8000:80

# Visit:
# http://localhost:8000

# Or with ingress/domain:
# https://canvas.example.com
```

### First-Time Setup

1. Open the UI in your browser
2. Go to **Settings → LLM**
3. Choose your provider (Anthropic, OpenAI, Google, etc.)
4. Select a model and enter your API key
5. Save — agents are now configured
6. Start a conversation and give it a task

### Configuring Agents (Claude Code, Codex, Gemini)

LLM configuration is done **through the UI** (Settings → LLM). Alternatively, set the API key environment variables in the Deployment (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`).

For ACP-compatible agents (Claude Code, Codex, Gemini CLI), authentication uses either:
- Subscription login (auto-detected from host credentials)
- API key (stored in the container via env vars or global secrets)

---

## 10. Remote Access from Phone

### Tailscale / Cloudflare Tunnel

If you're using Tailscale, expose the service:

```yaml
# Using Cloudflare Tunnel (as an alternative to Tailscale):
# Create a Cloudflare Tunnel pointing to:
# http://agent-canvas.agent-canvas.svc.cluster.local:80
```

### Tailscale Funnel (free, public access)

```bash
# On a node with Tailscale installed:
tailscale funnel 8000 $(kubectl port-forward -n agent-canvas svc/agent-canvas --address 0.0.0.0 8000:80 &)
```

Or use `--public` mode in Agent Canvas so anyone can connect with the API key.

---

## 11. Upgrading

```bash
# Update the image tag and apply:
kubectl set image deployment/agent-canvas \
  -n agent-canvas \
  agent-canvas=ghcr.io/openhands/agent-canvas:latest

# Check rollout:
kubectl rollout status deployment/agent-canvas -n agent-canvas
```

Since state is in PersistentVolumes, upgrades are safe — conversation history and settings persist across image updates.

---

## 12. Backup

Two volumes need backup:

| Volume | Contains | Backup Strategy |
|---|---|---|
| `agent-canvas-state` | Settings, secrets, conversations, automation DB, API keys | `kubectl exec` to tar, or Velero for PVC backup |
| `agent-canvas-projects` | Your actual project code | Git push from inside the container, or Velero |

```bash
# Quick backup of state volume:
kubectl exec -n agent-canvas -it $(kubectl get pod -n agent-canvas -l app=agent-canvas -o jsonpath='{.items[0].metadata.name}') \
  -- tar czf /tmp/state-backup.tar.gz /home/openhands/.openhands

# Download it:
kubectl cp agent-canvas/<pod-name>:/tmp/state-backup.tar.gz ./state-backup.tar.gz -n agent-canvas
```

---

## 13. Troubleshooting

### Pod Stuck on Readiness

The `/ready` endpoint returns 503 during initialization. The readiness probe has `failureThreshold: 10` to allow enough startup time. If it fails:

```bash
# Check logs:
kubectl logs -n agent-canvas -l app=agent-canvas --tail=100

# Check if the health endpoint responds:
kubectl exec -n agent-canvas <pod-name> -- curl -s http://localhost:8000/alive
```

### WebSocket/SSE Issues

Agent conversations use WebSocket and SSE for live event streaming. Ensure:
- Ingress controller supports WebSocket upgrades (`Connection: upgrade` header)
- Timeout is long enough (agent conversations can run for hours)
- Reverse proxy `proxy_read_timeout` and `proxy_send_timeout` are set to `3600s`

### Volume Permission Errors

```bash
# Fix permissions on state volume:
kubectl exec -n agent-canvas <pod-name> -- chown -R 1000:1000 /home/openhands/.openhands
```

### Agent Fails to Start

Check the container logs for agent-specific errors. Common issues:
- Missing LLM API key
- Agent CLI not installed (some agents need to be installed inside the container)
- Agent requires subscription login (Claude Code Max, etc.)

---

## 14. Known Limitations for K8s Deployment

| Limitation | Impact |
|---|---|
| **No official Helm chart for OSS** | You manage the manifests yourself |
| **SQLite default** — not ideal for stateful K8s workloads | Switch to PostgreSQL for production |
| **Single replica** — stateful app, not designed for multiple pods | One pod per deployment; add more backends via "Manage Backends" |
| **No auto-scaling** for agent workloads | Scale by adding more agent backends, not more Agent Canvas pods |
| **Agents run inside the container** — they have filesystem access to `/projects` | Treat the pod as trusted infrastructure |

---

## 15. Quick Install Commands

```bash
# 1. Apply manifests:
kubectl apply -f k8s/agent-canvas.yaml
kubectl apply -f k8s/ingress.yaml

# 2. Wait for readiness:
kubectl wait --for=condition=Available deployment/agent-canvas -n agent-canvas --timeout=120s

# 3. Port-forward for local testing:
kubectl port-forward -n agent-canvas svc/agent-canvas 8000:80

# 4. Open in browser:
# http://localhost:8000

# 5. Configure LLM and start coding
```

---

## References

- Agent Canvas repo: https://github.com/OpenHands/agent-canvas
- OpenHands main repo: https://github.com/All-Hands-AI/OpenHands
- Documentation: https://docs.openhands.dev
- Self-hosting guide: https://github.com/OpenHands/agent-canvas/blob/main/docs/SELF_HOSTING.md
- Docker backend setup: https://docs.openhands.dev/openhands/usage/agent-canvas/backend-setup/docker
- LLM settings: https://docs.openhands.dev/openhands/usage/agent-canvas/llm-profiles
- ACP agents: https://docs.openhands.dev/openhands/usage/agent-canvas/acp-agents
- Architecture: https://github.com/OpenHands/agent-canvas/blob/main/docs/architecture.md
