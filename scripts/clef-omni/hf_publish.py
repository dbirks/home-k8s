"""Publish (or refresh) a Clef-Omni quant on the Hugging Face Hub (issue #150).

    python hf_publish.py --repo dbirks/clef-omni-int4 --export /work/int4/export --card cards/clef-omni-int4.md \
        --private --tag v0.1                       # new repo: whole export, private, tagged
    python hf_publish.py --repo dbirks/clef-omni-nvfp4 --export /work/nvfp4-experts/export \
        --card cards/clef-omni-nvfp4.md --only README.md serve_clef.py SHA256SUMS   # refresh docs only

The card becomes README.md and serve_clef.py ships next to the weights. SHA256SUMS is rewritten so it
stays true. The token must belong to the repo's namespace. New repos are created private unless --public.
"""
import argparse
import hashlib
import shutil
from pathlib import Path

from huggingface_hub import HfApi

HERE = Path(__file__).resolve().parent


def sha256(p):
    with open(p, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--export", required=True)
    ap.add_argument("--card", required=True)
    vis = ap.add_mutually_exclusive_group()
    vis.add_argument("--private", action="store_true", default=True)
    vis.add_argument("--public", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--tag-message", default=None)
    ap.add_argument("--only", nargs="*", default=None, help="upload just these files (docs refresh)")
    ap.add_argument("--set-public", action="store_true", help="flip an existing private repo to public")
    args = ap.parse_args()
    export = Path(args.export)
    card = Path(args.card) if Path(args.card).is_absolute() else HERE / args.card

    api = HfApi()
    who = api.whoami()["name"]
    assert args.repo.split("/")[0] == who, f"token belongs to {who}, not {args.repo.split('/')[0]}"

    shutil.copy2(card, export / "README.md")
    shutil.copy2(HERE / "serve_clef.py", export / "serve_clef.py")
    files = sorted(p for p in export.iterdir() if p.is_file() and not p.name.startswith(".") and p.name != "SHA256SUMS")
    (export / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in files))
    print(f"staged README.md, serve_clef.py, SHA256SUMS ({len(files)} files) in {export}", flush=True)

    exists = api.repo_exists(args.repo)
    if not exists:
        api.create_repo(args.repo, private=not args.public, exist_ok=True)
        print(f"created {args.repo} ({'public' if args.public else 'private'})", flush=True)
    if args.only:
        for name in args.only:
            api.upload_file(path_or_fileobj=str(export / name), path_in_repo=name, repo_id=args.repo,
                            commit_message=f"Update {name}")
    else:
        api.upload_large_folder(repo_id=args.repo, repo_type="model", folder_path=str(export),
                                ignore_patterns=[".cache/**"], num_workers=4)
    remote = set(api.list_repo_files(args.repo))
    wanted = set(args.only) if args.only else {p.name for p in export.iterdir() if p.is_file() and not p.name.startswith(".")}
    missing = sorted(wanted - remote)
    assert not missing, f"missing on the Hub: {missing}"
    if args.set_public:
        api.update_repo_settings(args.repo, private=False)
        print(f"{args.repo} is now public", flush=True)
    info = api.model_info(args.repo)
    if args.tag and args.tag not in {t.name for t in api.list_repo_refs(args.repo).tags}:
        note = args.tag_message
        if note is None:   # default: the version note from quants.json, so tags and the card's history agree
            import json
            for q in json.loads((HERE / "quants.json").read_text())["quants"]:
                if q["repo"] == args.repo:
                    note = next((v["note"] for v in q.get("versions", []) if v["tag"] == args.tag), None)
        api.create_tag(args.repo, tag=args.tag, revision=info.sha, tag_message=note or args.tag)
    print(f"PUBLISHED https://huggingface.co/{args.repo} @ {info.sha} private={info.private} "
          f"({len(remote)} files){' tag ' + args.tag if args.tag else ''}", flush=True)


if __name__ == "__main__":
    main()
