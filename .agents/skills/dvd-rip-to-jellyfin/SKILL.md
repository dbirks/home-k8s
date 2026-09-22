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

For a NAME on a title card (episodes of old shows show their title after the intro), sample many frames into ONE montage image and Read it once — far cheaper than many single frames:
```bash
# 8 frames across 8s..64s, tiled 4x2 (widen window / raise density if the card is missed)
kubectl exec -n default "$POD" -- sh -c \
  '/usr/lib/jellyfin-ffmpeg/ffmpeg -y -ss 8 -t 64 -i "/media/.../file.mkv" \
   -vf "fps=1/8,scale=360:-1,tile=4x2" -frames:v 1 /tmp/m.png'
```
If the card is not in the window, resample wider/denser (e.g. `-ss 30 -t 85 -vf "fps=1/5,...,tile=4x4"`).

### Identify EVERY title straight off the disc BEFORE ripping (fastest triage)

Workstation `ffmpeg` reads a DVD title directly via the `dvdvideo` demuxer — no rip needed. Use this to figure out what's on an unknown multi-title disc (which title is which interview/episode) in ~1 min each, so you only rip what you actually want. Read from the START of each title (the `dvdvideo` demuxer does not seek well with `-ss` before `-i`; instead grab a window from 0 and let the montage span it):
```bash
# tile the first ~110s of title N at 1 frame / 4s -> catches intro logo + the lower-third name plate
sudo timeout 260 ffmpeg -y -f dvdvideo -title <N> -i /dev/sr0 -t 110 \
  -vf "fps=1/4,scale=380:-1,tile=5x5" -frames:v 1 -update 1 /tmp/tN.png
```
`-update 1` silences the "image sequence pattern" warning for a single output image. Documentary interview discs (e.g. the *Puritan: All of Life to the Glory of God* collector's-edition bonus discs) stamp a **lower-third name plate** ~20-60s in — the montage almost always catches it, and burned-in question captions ("Dr. MacArthur, who is your favorite Puritan?") confirm the subject. Read the on-screen name/credential; WebSearch only if a name is ambiguous. Then rip the wanted titles as whole titles (single chapter each) with the standard `-E copy` command.

## Documentary / box-set organization in a Jellyfin *Shows* library

A multi-part documentary (main film + companion teaching series + bonus interviews) maps cleanly onto one `Shows` series with numbered season folders. Jellyfin ONLY treats folders literally named `Season N` (or `Season NN`) as seasons — folders like `Part One` / `Part Three` are NOT parsed as seasons and show up as loose junk. Example end state that works:
```
Puritan/
  Season 0 - The Documentary/         <- the main feature film
  Season 1 - Pastors & Influential Figures/
  Season 2 - Puritan Teaching/
  Season 3 - Puritan Legacy/
  Season 4 - Extended Interviews/      <- collector's-edition bonus discs, one file per interviewee
```
Bonus-disc interviews go in their own `Season N - Extended Interviews`, one `Interviewee Name.mkv` per title (no forced SxxEyy needed — a set of named files sorts fine, and these won't match an online episode list anyway).

### De-duplicate a messy library SAFELY (verify byte-for-byte before deleting)

Imported libraries sometimes carry the same content twice under parallel naming schemes (a `Part …` tree AND a `Season …` tree; a film at both top level and in `Season 0/`). Before deleting a suspected duplicate, PROVE it is identical — compare name+size listings, never delete on a hunch. The pod shell is `dash` (no `<()` process substitution), so write listings to temp files and `diff`:
```bash
kubectl exec -n default "$POD" -- sh -c '
listing(){ ( cd "$1" && for f in *; do [ -f "$f" ] && printf "%s\t%s\n" "$(stat -c%s -- "$f")" "$f"; done | sort ); }
listing "/media/shows/Puritan/Part One - ..." > /tmp/a.txt
listing "/media/shows/Puritan/Season 1 - ..." > /tmp/b.txt
diff -q /tmp/a.txt /tmp/b.txt && echo IDENTICAL || diff /tmp/a.txt /tmp/b.txt'
```
Only once every pair reports IDENTICAL, `rm -rf` the redundant tree and `rm -f` the stray top-level dup, `mv` `Season 0` -> `Season 0 - The Documentary`, then `POST /Library/Refresh`. (Real case: this took the Puritan folder from 31G to 16G.) Deleting library files is destructive — get the user's OK on the target layout first.

## TV series across many discs (episode discs + bonus discs)

Real example: Davey and Goliath box sets. Series lives at `/media/shows/Davey and Goliath/` in the existing `Shows` (tvshows) library.

- **Pipeline shape:** keep the single optical drive ripping continuously (rip each title as its own background job, chain the next on completion) while curation (title-card read + naming) runs in parallel on already-staged files. Stage rips to a NON-library incoming dir like `/media/_incoming/<show>/tNN.mkv` so half-finished files never appear mid-scan; move into `Season NN/` only once named. Notify the user per finished rip.
- **Naming:** `Davey and Goliath S01E02 - Stranded on an Island.mkv` under `Season 01/`. Jellyfin parses SxxEyy for structure; the human title is for you/the user.
- **Real seasons vs disc order:** discs rarely equal broadcast seasons and often jump around. Look up true season/episode from TheTVDB (`https://thetvdb.com/series/<slug>/allseasons/official`) — Jellyfin's default TV metadata source, so matching its numbering makes artwork/synopsis line up. Read the title card for the NAME, then map name -> real SxxEyy. Do collision-safe renames when re-filing (move an episode out of a slot before moving another into it).
- **Specials / bonus features:** real specials (e.g. "Halloween Who-Dun-It") go in `Season 00/` as `S00Exx` using TheTVDB's special number. Making-of docs / read-alongs that are not TVDB episodes still go in `Season 00` with descriptive names (numbers approximate). Menu-only "bonus features" (trivia games, clickable episode guides, photo galleries) are not video titles and cannot be ripped — skip them.
- **Sanity-check the disc vs its sleeve:** `lsdvd` shows the real titles. If the case lists six episodes but `lsdvd` shows two, the disc is a subset / mislabeled / damaged — tell the user, do not fabricate the missing ones. Episodes run ~14-15 min; a lone ~29-30 min title is usually a half-hour special (or two episodes joined — check `lsdvd -c -t N` for chapter splits).

### Episodes packed into one long title (common on anime DVDs)

Some discs expose no per-episode titles — just one ~90-min title (often duplicated as titles 02 AND 03 — use either) holding several episodes back-to-back. Split by CHAPTER RANGE instead of post-processing:
1. `sudo lsdvd -c -t <title>` and add up chapter lengths. Episode boundaries fall where the running sum hits ~episode length (e.g. NieA_7: ~23:27 each → chapters 1-5, 6-10, 11-15, 16-21; each group = OP + 2 parts + ED + eyecatch).
2. Rip each group with HandBrake `-c C1-C2`:
   ```bash
   sudo HandBrakeCLI -i /dev/sr0 -t <title> -c 1-5 -o /tmp/e01.mkv \
     -f av_mkv -e x264 -q 20 --encoder-preset medium --comb-detect --decomb \
     --all-audio -E copy --audio-fallback av_aac --all-subtitles --markers
   ```
   Episodes are sequential on the disc, so number them in chapter-group order (Disc 1 = E01.. ; confirm the disc's volume/number with the user).

### No title cards? Identify by PLOT (budget / public-domain prints)

Cheap "Volume" DVD prints (e.g. an Andy Griffith Show budget set) often strip the on-screen episode title entirely — the show goes straight from cast credits into the story. Don't guess from memory.

**BEST method — read the CLOSING CAST CREDITS (definitive + fast).** Guest stars are unique per episode. Sample the last ~40s into a montage and Read it: `-ss <len-35> -t 40 -vf "fps=1/3,scale=360:-1,tile=4x4"` (episodes here run ~25:20, so `-ss 1490 -t 45`). If credits are low-contrast over the closing scene, grab a single full-res frame right on the "Cast"/"Guest Star" card (`-ss <t> -frames:v 1`, no scaling). Then WebSearch the guest-cast names ("Andy Griffith Show" + two or three guest names) — it lands the exact episode almost every time. An "Introducing ..." credit flags a character's DEBUT episode (e.g. "Introducing The Dillards / Maggie Peterson" ⇒ the Darlings' first episode "The Darlings Are Coming", not a later one). Map the confirmed title to its real SxxEyy via the episode guide.

**Fallback — identify by PLOT** (if credits are unreadable): sample mid-episode stills `-ss 120 -t 660 -vf "fps=1/60,scale=340:-1,tile=4x3"`, pull distinctive beats, WebSearch against the guide/IMDb/a fan wiki, and sample a second window at the climax to break ties. Plot-guessing is error-prone (it nearly mislabeled a Barney-matchmaking episode as a choir episode — the cast card corrected it), so prefer the credits and confirm on multiple details. Never file an episode you can't corroborate — hold it in an `_unsorted/` folder and ask the user.

Note: there is no image-reverse-search tool here; reading on-screen text (credits) and text-searching it is the reliable substitute.

### Cover art missing for a whole library? Internet providers are probably OFF

Libraries created via `POST /Library/VirtualFolders` default to `EnableInternetProviders: false` with empty `TypeOptions`, so Jellyfin NEVER fetches posters/metadata — every item stays art-less. Fix (non-destructive, API only):
1. `GET /Library/VirtualFolders` and check `LibraryOptions.EnableInternetProviders` on the movie/TV libraries.
2. Enable providers + configure fetchers via `POST /Library/VirtualFolders/LibraryOptions` (set `EnableInternetProviders: true` and populate `TypeOptions` with MetadataFetchers/ImageFetchers). **Only TheMovieDb + OMDb are installed on this instance — the TheTVDB plugin is NOT** — so use TheMovieDb for both movies AND TV (it serves TV fine).
3. `POST /Items/{id}/Refresh?metadataRefreshMode=FullRefresh&imageRefreshMode=FullRefresh&replaceAllMetadata=false&replaceAllImages=false` per series/movie (or a library `POST /Library/Refresh`), then verify each item now returns a `PrimaryImageTag`.
Caveat: episode NUMBERING is then matched by TMDb; if you named files with TheTVDB numbers they usually agree, but can differ for some shows — a per-episode Identify fixes any stragglers. When creating a NEW movie/TV library, set `EnableInternetProviders: true` up front to avoid this.

### Preserve ALL audio + subtitles (anime, or any multi-track disc)

Use `--all-audio -E copy --audio-fallback av_aac --all-subtitles`. `-E copy` PASSES THROUGH the original audio codecs LOSSLESSLY (keeps the Japanese AC3 alongside the English dub, and keeps 5.1 on movies instead of downmixing) — strongly preferred over re-encoding to AAC when the disc has a track worth keeping. DVD subs are image-based (`dvd_subtitle`/VOBSUB) and pass through into MKV with their language tags. Verify afterwards:
```bash
/usr/lib/jellyfin-ffmpeg/ffprobe -v error \
  -show_entries stream=index,codec_type,codec_name,channels:stream_tags=language,title \
  -of default=noprint_wrappers=1 file.mkv
```
Expect e.g. audio ac3/jpn + audio ac3/eng + subtitle dvd_subtitle/eng.

## Gotchas

- Do NOT name a bash variable `UID` — it is readonly and the assignment silently fails, breaking any query that uses it. Use `AUID` or similar.
- Filenames with spaces: always stage + `mv` in-pod (kubectl cp uses tar).
- `/tmp` on the workstation is tmpfs (RAM) — fine for one rip, but delete each MKV after pushing.
- Client apps (Moonfin, etc.) cache the list of views; after adding a library the user must pull-to-refresh or reopen the app to see it.
- A commercial disc that will not read may be CSS-encrypted — `libdvdcss` must be installed (HandBrake/dvdbackup use it automatically).
