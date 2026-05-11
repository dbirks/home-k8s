# Home K8s Cluster

Single-node Talos Linux v1.12.6 cluster with an NVIDIA GPU.

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
- Node IP is DHCP-assigned (currently 10.0.0.177) — update talosconfig endpoints and kubeconfig cluster server if it changes
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

## Talos

- Schematic uses `nvidia-open-gpu-kernel-modules` (required for Blackwell GPUs)
- Schematic ID: `036d341b186bfa76a1c0a545125bbd667908a09a50dfe5e7ab32cc93901b84a2`
- Talos config: `talosctl --talosconfig _newconfig/talosconfig -e 10.0.0.177 -n 10.0.0.177`
- Kubeconfig context: `admin@home`

## Networking

- DNS: Pi-hole at 10.0.0.202 (MetalLB LoadBalancer)
- external-dns watches Ingresses and creates DNS records in Pi-hole (v6 API)
- Ingress classes: `private` (internal), `public` (external)
- Tailscale subnet router advertises 10.0.0.0/24 for remote access
- Domain: `*.hoam.lan`

## Pitfalls learned the hard way

- NVFP4 on SM120 (Blackwell) has known vLLM kernel bugs with MoE models — dense models work fine
- TurboQuant KV cache does NOT work with hybrid attention+Mamba/DeltaNet models (like Qwen3.6)
- FP8 e4m3 KV cache works; e5m2 does NOT (incompatible with compressed-tensors)
- `/var/mnt` is read-only on Talos — use `/var/lib/local-path-provisioner` for local storage
- HelmRepository and OCIRepository must use API v1 (not v1beta2) with Flux v2.8+
- Qwen3.6-27B (77.2% SWE-bench Verified) outperforms much larger models including Qwen3-Coder-Next 80B on coding benchmarks
