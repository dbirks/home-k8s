# Talos config (talhelper-managed)

Machine configs are rendered declaratively with [talhelper](https://github.com/budimanjojo/talhelper) from `talconfig.yaml` + the SOPS-encrypted secrets bundle.

## Layout

```
talos/
├── talconfig.yaml          # cluster topology, node spec, schematic, patches  ─┐
├── talsecret.sops.yaml     # cluster CAs/tokens (SOPS-encrypted)               │ COMMITTED
├── patches/                # per-node config patches                            │
│   ├── local-storage.yaml                                                      │
│   └── nvidia-module-options.yaml                                              ─┘
└── clusterconfig/          # talhelper genconfig output                        ─ GITIGNORED
```

## First-time setup on a new machine

You need:
- `talosctl` (`https://github.com/siderolabs/talos/releases`)
- `talhelper` (`https://github.com/budimanjojo/talhelper/releases`)
- `sops` (`https://github.com/getsops/sops/releases`)
- The age private key restored to `~/.config/sops/age/keys.txt`

## Bootstrapping the talsecret.sops.yaml file

The repo ships without `talsecret.sops.yaml` initially. To populate it from an existing live cluster (e.g. after a fresh clone, where you have access to the current `controlplane.yaml`):

```bash
talosctl gen secrets --from-controlplane-config /path/to/controlplane.yaml -o talos/talsecret.yaml
sops --encrypt --in-place talos/talsecret.yaml
mv talos/talsecret.yaml talos/talsecret.sops.yaml
```

For a brand-new cluster:

```bash
talhelper gensecret > talos/talsecret.yaml
sops --encrypt --in-place talos/talsecret.yaml
mv talos/talsecret.yaml talos/talsecret.sops.yaml
```

The `.sops.yaml` creation rule at the repo root encrypts the whole file when the path matches `talos/talsecret.sops.yaml`.

## Day-2 workflow

Edit `talconfig.yaml` or a file under `patches/`, then:

```bash
talhelper genconfig                                         # renders clusterconfig/
talhelper gencommand apply --extra-flags --dry-run | bash   # preview diff
talhelper gencommand apply | bash                           # apply to live node
```

For OS upgrades (bumping `talosVersion` in `talconfig.yaml`):

```bash
talhelper gencommand upgrade | bash
```

To regenerate the talosconfig (kubeconfig-equivalent for talosctl):

```bash
talhelper gencommand kubeconfig | bash    # writes kubeconfig
# talosconfig is regenerated into clusterconfig/ alongside the machine config
```

## Notes

- **Never** commit plaintext `controlplane.yaml`, `worker.yaml`, `talosconfig`, or `talsecret.yaml` — `.gitignore` enforces this but always double-check.
- Prefer this workflow over `talosctl patch mc` for permanent changes — direct patches drift from git.
- Node IP is DHCP-assigned. Currently `10.0.0.177`; update `talconfig.yaml` if it changes.
