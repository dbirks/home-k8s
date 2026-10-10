# Home K8s Cluster

Single-node Talos Linux v1.13.6 cluster with an NVIDIA GPU.

> `AGENTS.md` and `CLAUDE.md` are the SAME file — `CLAUDE.md` is a symlink to `AGENTS.md`. Edit `AGENTS.md`; both toolchains read it.

## Skills: capture what you learn (a living how-to ledger)

**This repo keeps a growing, shared how-to ledger as Agent Skills in `.agents/skills/`.** We do a lot of oddball, hard-won things here. Whenever you (any agent) work out how to do something non-obvious — a multi-step process, a troubleshooting path, a fix you had to fight through, a handy script — **write it down as a skill so the next agent doesn't have to rediscover it.** Treat it as a living notebook: correct and improve existing skills in place as understanding grows, don't just pile on new ones.

Format (the open Agent Skills spec, https://agentskills.io/specification): one folder per skill at `.agents/skills/<name>/SKILL.md`, where `SKILL.md` is YAML frontmatter + a markdown body:

```
---
name: <lowercase-kebab; MUST match the folder name exactly; <= 64 chars>
description: <what it does AND when to use it; one or two sentences; <= 1024 chars>
---
<the how-to: steps, exact commands, gotchas, where things live, how to verify it worked>
```

Optional bundles: helper scripts under `scripts/`, longer docs under `references/`, templates under `assets/`. Keep the body skimmable and action-oriented. Avoid `<`/`>` in the frontmatter (they can be misread as instructions).

Both toolchains discover these: other agents read `.agents/skills/` directly; Claude Code finds them because `.claude/skills` is a symlink to `.agents/skills` (they also become `/`-invocable). That symlink and the `CLAUDE.md -> AGENTS.md` one are the whole reason to keep the canonical copies under `.agents/` and `AGENTS.md`.

**NEVER put anything sensitive in a skill.** No passwords, API keys, tokens, private keys, or secret values of any kind — this repo is PUBLIC on GitHub (see Secrets below), and skills are committed. Instead describe *how* to obtain or rotate a credential and *where* it lives ("read it from the SOPS secret", "mint a scoped API key in the dashboard"), never the value. The bar is the same as the rest of the repo: process, pointers, and troubleshooting — not secrets.

Rule of thumb for when to write one: if you'd have to re-derive it next time, it's a skill. If it's already captured by the code, git history, or a plain fact, it isn't.

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
- Node IP is DHCP-assigned (currently **10.0.0.194**, verified via `kubectl get nodes -o wide` 2026-10-08 — it moves often, always re-check before using it). If it changes, update the kubeconfig cluster server and the talosctl endpoints. NOTE: `talos/talconfig.yaml` and the talosctl examples below still reference the older `10.0.0.177`; reconcile those to the live IP when convenient.
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

## Serving stack (retired 2026-10-10)

The old KServe / Envoy / KEDA stack that served the public coworker endpoint `llm.birks.dev`
(six-model catalog, scale-to-zero, Cloudflare Tunnel, Entra-OIDC API-key portal) has been
**fully torn down** — the endpoint, portal, keys, gateway control planes, KEDA, and the Gateway
API CRDs are all gone. History: issues #95, #105, #135, #144 and git history.

What serves models today:

- **Pennyroyal** (`apps/pennyroyal-flashnext.yaml`, SGLang fork, Qwen3.8-Flash-Next NVFP4) owns
  the whole card and is the primary engine: model name `pennyroyal`, context 524,288, no
  scale-to-zero (sleep-on-idle keeps it hot). Operate it via the pennyroyal-flashnext skill.
- **LLM Collective contributor** (`apps/llm-collective-contributor*.yaml`) is the consumer and it
  talks to engines DIRECTLY over the cluster Service (e.g.
  `http://pennyroyal-flashnext.default.svc.cluster.local:8001`). This direct pattern is the only
  wiring; there is no in-cluster gateway.
- **Parked engines** (ninfer, flashnext rollback) keep Deployment+Service manifests with revival
  notes in their headers (direct Service + Collective backend, or a tailscale Ingress). KEDA is
  gone, so a revived engine is always-on at `replicas: 1`.
- **Co-residence limit still applies**: the binding constraint is **46GB host RAM (no swap), NOT
  the 96GB VRAM**; scale one engine to 0 before rolling out another (see the ninfer-serving-memory
  and pennyroyal-flashnext skills).
- Do NOT reintroduce KEDA / Envoy Gateway / KServe LLMISVC assumptions into manifests or
  monitoring: those CRDs are removed and the KServe-engine/EPP/`envoy-gw` scrape jobs are deleted.


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
