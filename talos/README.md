# Talos config (talhelper-managed)

Machine configs are rendered declaratively with [talhelper](https://github.com/budimanjojo/talhelper) from `talconfig.yaml` + a SOPS-encrypted `talsecret.sops.yaml`. This repo is **public** — only encrypted secrets ever land in git.

## Layout

```
talos/
├── talconfig.yaml          # cluster topology, node spec, schematic, patch refs ─┐
├── talsecret.sops.yaml     # cluster CAs/tokens (SOPS+age encrypted)              │ COMMITTED
├── patches/                # per-node patches                                     │
│   ├── local-storage.yaml                                                         │
│   └── nvidia-module-options.yaml                                                 │
├── README.md               # this file                                            ─┘
└── clusterconfig/          # talhelper genconfig output                          ─ GITIGNORED
```

## Tooling (versions verified May 2026)

| Tool | Version | Source |
|---|---|---|
| Talos / talosctl | v1.13.0 (release line) | https://github.com/siderolabs/talos/releases |
| Kubernetes | v1.36.0 (max for Talos 1.13) | https://kubernetes.io/releases/ |
| talhelper | v3.1.9 | https://github.com/budimanjojo/talhelper/releases |
| sops | v3.13.0 | https://github.com/getsops/sops/releases |
| age | v1.3.1 | https://github.com/FiloSottile/age/releases |

Match `talosctl` to your cluster's Talos version. Check the [Talos support matrix](https://www.talos.dev/v1.13/introduction/support-matrix/) before bumping Kubernetes.

## First-time setup on a new machine

Install the tools:

```bash
# talosctl — match your cluster's Talos version, install latest is fine
curl -sL https://github.com/siderolabs/talos/releases/latest/download/talosctl-linux-amd64 \
  -o ~/.local/bin/talosctl && chmod +x ~/.local/bin/talosctl

# talhelper
TH=$(curl -s https://api.github.com/repos/budimanjojo/talhelper/releases/latest | grep tag_name | cut -d'"' -f4)
curl -sL "https://github.com/budimanjojo/talhelper/releases/download/${TH}/talhelper_linux_amd64.tar.gz" \
  | tar -xz -C ~/.local/bin talhelper

# sops
SO=$(curl -s https://api.github.com/repos/getsops/sops/releases/latest | grep tag_name | cut -d'"' -f4)
curl -sL "https://github.com/getsops/sops/releases/download/${SO}/sops-${SO}.linux.amd64" \
  -o ~/.local/bin/sops && chmod +x ~/.local/bin/sops

# age (only needed to bootstrap a NEW cluster; existing clusters reuse the key)
AG=$(curl -s https://api.github.com/repos/FiloSottile/age/releases/latest | grep tag_name | cut -d'"' -f4)
curl -sL "https://github.com/FiloSottile/age/releases/download/${AG}/age-${AG}-linux-amd64.tar.gz" \
  | tar -xz -C ~/.local/bin --strip-components=1 age/age age/age-keygen
```

Restore the age private key:

```bash
mkdir -p ~/.config/sops/age
# Paste/copy your existing key file here as keys.txt
chmod 600 ~/.config/sops/age/keys.txt
export SOPS_AGE_KEY_FILE=~/.config/sops/age/keys.txt    # add to ~/.bashrc / ~/.zshrc
```

The public key recipient is in `/.sops.yaml`. If you don't have the private key, you can't decrypt anything — back it up offline.

## SOPS secret workflow

talhelper **auto-decrypts** SOPS-encrypted files when `SOPS_AGE_KEY_FILE` is set. No separate decrypt step. The convention is `talsecret.sops.yaml` (and optionally `talenv.sops.yaml`); talhelper finds them by filename.

### Bootstrap talsecret.sops.yaml from an existing running cluster

You have a populated `controlplane.yaml` locally (gitignored). Extract the secrets from it and encrypt:

```bash
export SOPS_AGE_KEY_FILE=~/.config/sops/age/keys.txt
talosctl gen secrets --from-controlplane-config controlplane.yaml -o talos/talsecret.sops.yaml
sops --encrypt --in-place talos/talsecret.sops.yaml
# Verify it's encrypted (top of file should show sops metadata, not raw cluster:/secrets:)
head -3 talos/talsecret.sops.yaml
git add talos/talsecret.sops.yaml
git commit -m "chore(talos): add encrypted talsecret"
```

### Bootstrap talsecret.sops.yaml for a fresh new cluster

```bash
# Generate a new age key pair if you don't have one
age-keygen -o ~/.config/sops/age/keys.txt
chmod 600 ~/.config/sops/age/keys.txt
# Copy the public key (printed on stderr by age-keygen) into .sops.yaml

export SOPS_AGE_KEY_FILE=~/.config/sops/age/keys.txt
talhelper gensecret > talos/talsecret.sops.yaml
sops --encrypt --in-place talos/talsecret.sops.yaml
```

### Never re-generate talsecret on a running cluster

Once a cluster is bootstrapped with a given `talsecret`, the cluster's PKI is tied to it. Regenerating `talsecret` and applying breaks the cluster — you'd have to reset and rebuild. Edit `talconfig.yaml` and patches; leave `talsecret.sops.yaml` alone.

## Day-2 workflow

Edit `talconfig.yaml` or a file under `patches/`, then:

```bash
export SOPS_AGE_KEY_FILE=~/.config/sops/age/keys.txt

talhelper genconfig                                          # renders clusterconfig/
talhelper gencommand apply --extra-flags "--dry-run" | bash  # preview
talhelper gencommand apply | bash                            # apply
```

The rendered `clusterconfig/` directory is gitignored — never commit it.

## Upgrade workflows

Always upgrade **Talos OS first**, then Kubernetes. Both are minor-version-at-a-time (no skipping). The Talos support matrix lists which K8s versions each Talos minor supports.

### Talos OS upgrade (minor or patch)

1. Bump `talosVersion:` in `talconfig.yaml` (e.g. `v1.12.6` → `v1.13.0`).
2. The **schematic ID stays the same** — it's a content hash of your extension list, not Talos-version-bound. Only regenerate the schematic if you're adding/removing extensions.
3. Render and apply:
   ```bash
   talhelper genconfig
   talhelper gencommand upgrade --extra-flags "--preserve" | bash
   ```
   `--preserve` keeps the EPHEMERAL partition (etcd state) intact — required on control-plane nodes.
4. Wait for the node to reboot into the new Talos version (~3-5 minutes).
5. Verify: `talosctl version --nodes 10.0.0.177`.

### Kubernetes upgrade (minor)

Requires Talos version that supports the target K8s. Check support matrix.

1. Bump `kubernetesVersion:` in `talconfig.yaml` (e.g. `v1.35.0` → `v1.36.0`).
2. Render and apply:
   ```bash
   talhelper genconfig
   talhelper gencommand upgrade-k8s | bash
   ```
   This runs `talosctl upgrade-k8s --to <version>` which orchestrates control plane + kubelet rollout.
3. Verify: `kubectl version` shows the new server version.

### Schematic change (add/remove a system extension)

1. Edit the `schematic:` block in `talconfig.yaml`.
2. talhelper recomputes the new ID against `factory.talos.dev` automatically:
   ```bash
   talhelper genurl installer    # see the new installer URL with the new schematic ID
   talhelper genconfig
   talhelper gencommand upgrade | bash
   ```
3. This is an OS upgrade-style operation — it reboots the node.

## CI/CD (without keys on disk)

`sops` accepts the age private key via env var `SOPS_AGE_KEY` (literal key contents, not a path). In GitHub Actions:

```yaml
- name: Render Talos config
  env:
    SOPS_AGE_KEY: ${{ secrets.SOPS_AGE_KEY }}
  run: talhelper genconfig
```

This lets a CI job validate that `talconfig.yaml` + patches still render cleanly without persisting the key to the runner filesystem.

## Gotchas

- **`SOPS_AGE_KEY_FILE` not exported** — `talhelper genconfig` fails to decrypt referenced files, may produce subtly wrong output. Add the export to your shell rc.
- **Hand-editing `clusterconfig/`** — never; always regenerate. It's gitignored for a reason.
- **`talsecret.sops.yaml` regeneration on a live cluster** — destroys the cluster's PKI. Treat the file as write-once-per-cluster.
- **Schematic ID drift** — the ID is a content hash. Same ID works across all Talos versions. You only get a new ID when the schematic YAML changes.
- **K8s ahead of Talos** — Talos 1.12 maxes at K8s 1.35; you cannot bump K8s to 1.36 until Talos is on 1.13.
- **Skipping minors** — don't go 1.12 → 1.14 in one step; go through 1.13 first.
- **`talosctl patch mc` drift** — direct patches modify the live config without going through git. Use only for ad-hoc debugging; permanent changes go through `talconfig.yaml`/`patches/` + `talhelper gencommand apply`.
- **Node IP changes** — DHCP-assigned. Update `talconfig.yaml`'s `nodes[].ipAddress` and re-render if it moves.
