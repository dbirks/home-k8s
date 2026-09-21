---
name: dvd-rip-to-jellyfin
description: Rip a physical DVD (home videos or personal discs) on David's workstation and load it into a Jellyfin library. Use when digitizing DVDs into Jellyfin, creating/scanning a Jellyfin library, or identifying an unknown DVD title.
---

# Rip a DVD into Jellyfin

The optical burner is on **David's workstation** (`/dev/sr0`, symlink `/dev/cdrom`) — the Talos cluster node has NO optical drive. Reading the drive needs elevated access (the user is not in the `optical` group), so run rip/eject under `sudo` (passwordless on the workstation).

## One-time setup

Tools (Arch, `extra` repo): `sudo pacman -S --needed handbrake-cli lsdvd dvdbackup libdvdcss`. `libdvdcss` decrypts commercial (CSS) discs; home-video discs are unencrypted so it is usually moot. `ffmpeg` is already present.

## Per-disc procedure

1. **Eject / load:** `sudo eject /dev/sr0`, insert the disc.
2. **Identify:** wait for spin-up, then
   ```bash
   sudo blkid /dev/sr0          # UDF volume LABEL (e.g. YesVideo transfers show LABEL="Yesvideo")
   sudo lsdvd /dev/sr0          # enumerate titles + lengths
   ```
   The main feature is the longest title. Titles under ~1 min are logos/menu loops — skip them. Commercial discs list the movie plus several extras.
3. **Rip a title to MKV** (HandBrake reads the disc directly — no VIDEO_TS copy needed). Write to `/tmp` (tmpfs, ~6-7 GB free); the workstation root `/` runs ~96% full, so do NOT write big files there.
   ```bash
   sudo HandBrakeCLI -i /dev/sr0 -t <TITLE#> -o /tmp/out.mkv \
     -f av_mkv -e x264 -q 20 --encoder-preset medium \
     --comb-detect --decomb \
     --all-audio -E av_aac -B 192 --audio-fallback av_aac \
     --all-subtitles --markers
   ```
   - `--comb-detect --decomb`: camcorder/anime DVDs are interlaced; this deinterlaces only combed frames.
   - `--all-audio --all-subtitles`: keep every audio track + subtitle (important for anime: Japanese audio + English subs both survive).
   - Speed on the i7-7700: ~130-150 fps for SD (720x480), so a ~50-min title takes ~10 min and lands ~800 MB. Run long rips with a background job and poll the HandBrake log (`tr '\r' '\n'` to read the progress line).
4. **Push into Jellyfin** (media lives on the `jellyfin-media-hdd` PVC). `kubectl cp`/tar chokes on spaces, so stage under a no-space name and rename in-pod:
   ```bash
   POD=$(kubectl get pods -n default -o name | grep -i jellyfin | head -1 | cut -d/ -f2)
   kubectl cp /tmp/out.mkv "default/$POD:/media/<dest>/_staging.mkv"
   kubectl exec -n default "$POD" -- sh -c 'mv -f "/media/<dest>/_staging.mkv" "/media/<dest>/Real Name.mkv"'
   rm -f /tmp/out.mkv    # /tmp is RAM-backed; free it
   ```

## Jellyfin library layout & API

Library definitions live in the config PVC under `/config/root/default/<Library>/`; media under `/media/...`. Two libraries exist:
- **Home Videos** — `collectionType=homevideos`, path `/media/home-videos`. Name files descriptively (the view lists by filename), e.g. `Birks Family - Newborn David.mkv`.
- **Movies** — `collectionType=movies`, path `/media/movies`. One folder per film named `Movie Name (YEAR)/Movie Name (YEAR).mkv` so Jellyfin matches TMDb metadata/artwork (leave internet providers ON). Put bonus features in an `extras/` subfolder inside the movie folder — Jellyfin shows them as an "Extras" row.

Create + scan a library via the REST API (create needs an **admin API key** — mint one in Jellyfin *Dashboard -> Advanced -> API Keys*; **do not store the key here or anywhere in git**). Every call needs header `Authorization: MediaBrowser Token=<KEY>`. Easiest to curl from inside the pod against `http://localhost:8096`:
- Create: `POST /Library/VirtualFolders?name=Movies&collectionType=movies&paths=%2Fmedia%2Fmovies&refreshLibrary=false` (expect HTTP 204). No Jellyfin restart needed.
- Scan all: `POST /Library/Refresh` (HTTP 204).
- Verify: `GET /Items?userId=<adminId>&recursive=true&includeItemTypes=Movie&searchTerm=<term>` -> check `TotalRecordCount`. Get `<adminId>` from `GET /Users`.

## Identify an unknown title without watching it

Extract a mid-clip frame with the pod's bundled ffmpeg and view it:
```bash
kubectl exec -n default "$POD" -- sh -c \
  '/usr/lib/jellyfin-ffmpeg/ffmpeg -y -ss 140 -i "/media/.../file.mkv" -frames:v 1 /tmp/frame.png'
kubectl cp "default/$POD:/tmp/frame.png" /tmp/frame.png   # then open/Read it
```

## Gotchas

- Do NOT name a bash variable `UID` — it is readonly and the assignment silently fails, breaking any query that uses it. Use `AUID` or similar.
- Filenames with spaces: always stage + `mv` in-pod (kubectl cp uses tar).
- `/tmp` on the workstation is tmpfs (RAM) — fine for one rip, but delete each MKV after pushing.
- Client apps (Moonfin, etc.) cache the list of views; after adding a library the user must pull-to-refresh or reopen the app to see it.
- A commercial disc that will not read may be CSS-encrypted — `libdvdcss` must be installed (HandBrake/dvdbackup use it automatically).
