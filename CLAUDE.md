# Home K8s Cluster

Single-node Talos Linux v1.13.6 cluster with an NVIDIA GPU.

## Repo structure

- `flux-system/` - Flux GitOps controllers and sync config
- `prereqs/` - Cluster prerequisites (tailscale-operator, etc.)
- `infra/` - Infrastructure (ingress-nginx, metallb, cert-manager, external-dns, local-path-provisioner, etc.)
- `apps/` - Application workloads (vllm, pihole, jellyfin, etc.)
- `talos/` - Talos machine config patches (not tracked by Flux, contains secrets)
- `_newconfig/` - Generated Talos configs (not tracked by Flux, contains secrets)

Flux reconciles in order: `prereqs` -> `infra` -> `apps`

## Secrets (SOPS + age)

**This repo is PUBLIC on GitHub. No unencrypted secrets ever go in git, period.** Plaintext machine configs that contain cluster CAs / tokens / encryption keys (`controlplane.yaml`, `worker.yaml`, `talosconfig`, etc.) are gitignored. Anything sensitive that does need to live in git must be SOPS-encrypted first.

Secrets are encrypted in-repo with SOPS + age. Flux decrypts automatically.

- Config: `.sops.yaml` in repo root
- Encrypted files use `.enc.yaml` suffix
- Flux decryption configured in `flux-system/apps.yaml` (decryption block referencing `sops-age` secret)
- Age private key: `~/.config/sops/age/keys.txt` (not in repo, must be backed up)

To create an encrypted secret:
```bash
kubectl create secret generic NAME --namespace=default \
  --from-literal=KEY=value \
  --dry-run=client -o yaml \
  | sops encrypt --input-type yaml --output-type yaml /dev/stdin \
  > apps/NAME.enc.yaml
```

After a cluster wipe, recreate the age key:
```bash
kubectl create secret generic sops-age --namespace=flux-system \
  --from-file=age.agekey=$HOME/.config/sops/age/keys.txt
```

## Key conventions

- **All changes must go through GitOps** — edit files in the repo, commit, and let Flux reconcile. Do not patch deployments directly with kubectl.
- Suspended apps are renamed to `.yaml.hold` so Flux ignores them
- Scaled-down deployments use `replicas: 0` in their yaml (e.g. `apps/vllm-tts.yaml`)
- Node IP is DHCP-assigned (currently **10.0.0.136**, verified via `kubectl get nodes`). If it changes, update the kubeconfig cluster server and the talosctl endpoints. NOTE: `talos/talconfig.yaml` and the talosctl examples below still reference the older `10.0.0.177`; reconcile those to the live IP when convenient.
- `enableServiceLinks: false` is required on vLLM pods (K8s service named "vllm" conflicts with vLLM's VLLM_PORT env var)
- GPU workloads need `runtimeClassName: nvidia`
- Node needs label `feature.node.kubernetes.io/pci-10de.present=true` for nvidia-device-plugin DaemonSet
- GPU sidecar containers use `NVIDIA_VISIBLE_DEVICES=all` env var to share the GPU (only the main container holds the `nvidia.com/gpu` resource limit)

## vLLM

- Main deployment: `apps/vllm.yaml` — Qwen3.6-27B (coding) + Granite Speech 4.1 2B (speech-to-text) as sidecar
- TTS deployment: `apps/vllm-tts.yaml` — Qwen3-TTS, scaled down
- Endpoints: `vllm.hoam.lan` (LLM), `speech.hoam.lan` (speech-to-text)
- Qwen3.6-27B is a hybrid model (DeltaNet+Attention) — TurboQuant KV cache is NOT compatible, use fp8_e4m3
- NVFP4 quantization leverages Blackwell FP4 tensor cores for native 4-bit compute

## Serving stack (KServe / llm.birks.dev)

The public coworker endpoint `llm.birks.dev` is served by a KServe LLM stack, NOT by the `apps/vllm.yaml` deployment above. Full diagram-ready reference: GitHub issue #95.

- **Model catalog**: six models, each a KServe `LLMInferenceService` (LLMISVC, KServe v0.20.0): qwen3.8-27b, muse-glimmer-30b, qwen3.6-27b, qwen3.6-35b-a3b, gemma-4-31b, laguna-s-2.1. One LLMISVC expands into a vLLM engine Deployment, a workload Service (`<name>-kserve-workload-svc:8000`, the raw OpenAI server), an `InferencePool`, and an EPP router-scheduler.
- **Two gateways, layered (we need both, they are not alternatives)**:
  - **Envoy Gateway v1.8.1** is the only data plane. Single `GatewayClass` `envoy`; it runs every proxy pod (in `envoy-gateway-system`) and provides TLS, API-key auth (`SecurityPolicy`), and timeouts (`BackendTrafficPolicy`).
  - **Envoy AI Gateway v1.0.0** is a control-plane extension only (`ai-gateway-controller`, no proxy of its own). Its `AIGatewayRoute` reads the request-body `model`, stamps `x-ai-eg-model`, and routes to the right `InferencePool` via EPP. It emits the generated `llm-model-router` HTTPRoute that Envoy Gateway then serves. Base Envoy Gateway cannot do body-based model routing or InferencePool targeting, which is why the extension exists.
  - Mnemonic: AI Gateway decides which model, Envoy Gateway carries the bytes.
- **Scale-to-zero**: KEDA v2.20.2, one ScaledObject per model, triggered on the Envoy ext_proc request-rate metric scraped by the `envoy-gw` Prometheus job (15s). Idle models scale their engine to 0 and cold-start on the first request (warmup ~2min).
- **Co-residence limit**: the binding constraint is **46GB host RAM (no swap), NOT the 96GB VRAM**. About two models fit resident at once; a rollout that surges to a third overloads the node. Scale a model to 0 before changing it.
- **Auth / access**: external coworkers reach it as `llm.birks.dev` via Cloudflare Tunnel to the `llm-public-gw` Gateway, API-key authenticated (keys issued as k8s Secrets by the Entra-OIDC Go portal). Internal high-volume batch clients (for example the regen job) should hit the vLLM `*-workload-svc:8000` DIRECTLY, bypassing the gateway; the AI Gateway ext-proc buffers and translates every body and is fragile on large batch responses.

## Talos

- Schematic uses `nvidia-open-gpu-kernel-modules` (required for Blackwell GPUs)
- Schematic ID: `036d341b186bfa76a1c0a545125bbd667908a09a50dfe5e7ab32cc93901b84a2`
- Talos config: `talosctl --talosconfig _newconfig/talosconfig -e 10.0.0.177 -n 10.0.0.177`
- Kubeconfig context: `admin@home`

## GPU crash recovery (FULLCHIP_RESET wedge) — the one command to remember

The RTX PRO 6000 Blackwell GSP firmware periodically crashes (Xid 79/154, NVIDIA bug 6426268, issue #46). The bad outcome is a **FULLCHIP_RESET wedge**: the GPU falls off the bus, the node keeps its K8s API up but advertises `nvidia.com/gpu: 0`, `hami-device-plugin` goes RunContainerError, and every GPU pod (ninfer, qwen38, etc.) lands in `ContainerStatusUnknown`. dmesg is a wall of `NV_ERR_GPU_IN_FULLCHIP_RESET` assertions.

**A warm `talosctl reboot` does NOT clear this** (the card won't re-enumerate: `load nvidia failed: no such device`). The ONLY fix is a BMC/IPMI power-cycle:

```bash
# node IP is DHCP (no reservation yet) — discover it, then power-cycle:
NODE_IP=$(kubectl get node talos-210-73x -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}')
talosctl --talosconfig _newconfig/talosconfig -e "$NODE_IP" -n "$NODE_IP" reboot --mode powercycle
```

The `--mode powercycle` is the key part: it escalates to the BMC to actually cut and restore power. The node comes back in ~2-3 min, the GPU re-enumerates (`nvidia.com/gpu: 10`), and ninfer/qwen38 pods recover on their own. Clean up any leftover dead pods with `kubectl delete pods --field-selector=status.phase=Failed -A`.

**Auto-recovery is installed** as a user systemd timer on David's workstation (NOT in-cluster): `~/.local/share/home-k8s-auto/gpu-wedge-watchdog.sh`, fired every 2 min by `gpu-wedge-watchdog.timer`. It power-cycles ONLY on a confirmed, persistent wedge (`gpu cap 0` AND dmesg `GPU_IN_FULLCHIP_RESET`, held ≥120s), with a 30-min cooldown to prevent reboot loops. Logs: `~/.local/share/home-k8s-auto/gpu-wedge-watchdog.log`. Check it with `systemctl --user list-timers gpu-wedge-watchdog.timer`.

**First, though, CHECK THE POWER CAP.** A recurring ~fixed-interval crash storm (every ~18 min) was caused by the `gpu-power-limit` DaemonSet re-asserting 600W and overriding the 400W `nvidia-power-cap`. Keep it at **400W** (`apps/gpu-power-limit.yaml` `TARGET_WATTS: "400"`); if crashes persist at 400W, drop to 350W → 300W. Do NOT raise it.

## Networking

- DNS: Pi-hole at 10.0.0.202 (MetalLB LoadBalancer)
- external-dns watches Ingresses and creates DNS records in Pi-hole (v6 API)
- Ingress classes: `private` (internal), `public` (external)
- Tailscale subnet router advertises 10.0.0.0/24 for remote access
- Domain: `*.hoam.lan`

## Pitfalls learned the hard way

- NVFP4 on SM120 (Blackwell): dense models always work. MoE was broken but is now LARGELY FIXED (mid-2026) — **W4A4** NVFP4 MoE serves natively via FlashInfer b12x/CUTLASS on SM120 (vLLM PR #40082 merged 2026-05, flashinfer ≥0.6.13), needs a recent vLLM (~v0.24+) and may need `VLLM_USE_FLASHINFER_MOE_FP4` until auto-select (vLLM PR #47577) merges. Caveat: weight-only **W4A16**-NVFP4 MoE exports still fall back to Marlin (#47749) — export W4A4 for MoE. AutoRound can produce MoE-NVFP4 today.
- TurboQuant KV cache does NOT work with hybrid attention+Mamba/DeltaNet models (like Qwen3.6)
- FP8 e4m3 KV cache works; e5m2 does NOT (incompatible with compressed-tensors)
- `/var/mnt` is read-only on Talos — use `/var/lib/local-path-provisioner` for local storage
- HelmRepository and OCIRepository must use API v1 (not v1beta2) with Flux v2.8+
- Qwen3.6-27B (77.2% SWE-bench Verified) outperforms much larger models including Qwen3-Coder-Next 80B on coding benchmarks
