---
name: flux-image-automation
description: Make Flux auto-roll a Deployment to the newest image, including private GHCR packages and CI that only publishes a moving tag like main. Use when adding an ImageRepository/ImagePolicy/ImageUpdateAutomation, when an automation shows "key ... marked as read only", or when no fluxcdbot commits ever land.
---

# Flux image automation on home-k8s

Controllers (`image-reflector-controller`, `image-automation-controller`) are already
installed. Working example: `apps/communication/communication-image-automation.yaml`.

## Gotcha 1: the flux-system deploy key is READ-ONLY

`flux-system` GitRepository uses the `flux-system-main-flux-system` deploy key, which is
read-only. Any ImageUpdateAutomation whose `sourceRef` is `flux-system` fails with
`failed to push to remote: ... The key you are authenticating with has been marked as read only`.
That is why `shepherd` (apps/shepherd-image-automation.yaml) has never pushed. Check
for it with `git log --author=fluxcd` (empty means nothing ever landed).

Fix (already in place): point the automation at the **`flux-image-automation`**
GitRepository (flux-system ns). It uses the `flux-image-automation` **read-write**
deploy key (`gh repo deploy-key list -R dbirks/home-k8s`), whose private half is in
`apps/communication/flux-image-automation-git.enc.yaml` (keys `identity`,
`identity.pub`, `known_hosts`). Reuse it for new automations:

```yaml
  sourceRef: { kind: GitRepository, name: flux-image-automation }
```

To rotate it: `ssh-keygen -t ed25519`, `gh repo deploy-key add KEY.pub --allow-write`,
re-create the secret with `kubectl create secret generic ... --dry-run=client -o yaml | sops encrypt ...`,
then delete the old key in GitHub.

**Scope `update.path` tightly** (e.g. `./apps/communication`). Several stale
`$imagepolicy` markers in `./apps` (shepherd, llm-collective) have never been applied.
An automation over all of `./apps` with a working key would roll ALL of them at once.

## Gotcha 2: CI only publishes a moving tag (`:main`, `:sha-abc1234`)

Short-sha tags aren't orderable. Elect the single moving tag and reflect its digest.
The Setters marker is then rewritten to `repo:main@sha256:...` on every push (a real
commit, so a real rollout):

```yaml
  filterTags: { pattern: '^main$' }
  policy: { alphabetical: { order: asc } }
  digestReflectionPolicy: Always
  interval: 5m0s
```

If you control the CI, `<sha>-<epoch>` tags with a `numerical` policy (shepherd's
pattern) also work.

## Gotcha 3: private GHCR packages

- An ImageRepository needs `secretRef` to a dockerconfigjson **in flux-system**. Pods
  need one in their own namespace.
- `ghcr-pull` (default ns) is an older token and 403s on newer dbirks packages.
  `market-agent-ghcr-pull` (classic PAT, read:packages) works. Copies:
  `communication-ghcr-pull` (flux-system). Rotate them together.
- Test a credential without printing it: dump the secret's `.dockerconfigjson` to a 0600
  file, then run `skopeo inspect --authfile FILE docker://ghcr.io/OWNER/IMG:TAG`.
- `gh` CLI tokens here lack `read:packages`, so `gh api .../packages` 403s. Use skopeo.

## Marker

Put `# {"$imagepolicy": "flux-system:NAME"}` on the line above every `image:` that
should move, init containers included.

## Verify

```bash
kubectl --context admin@home get imagerepository,imagepolicy,imageupdateautomation -n flux-system NAME
# policy: "resolved to main with digest sha256:..."; automation: "repository up-to-date" or a pushed commit
```
