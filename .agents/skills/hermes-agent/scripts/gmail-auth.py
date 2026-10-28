#!/usr/bin/env python3
"""Authorize one Gmail account for the gmail-broker and write its SOPS-encrypted Secret into the repo.

Usage (on the workstation, from anywhere in the repo):
  python3 .agents/skills/hermes-agent/scripts/gmail-auth.py ALIAS [--client ~/Downloads/hermes-gmail-client.json]

Runs Google's installed-app OAuth flow (loopback redirect + PKCE) in your browser, asks for gmail.readonly +
gmail.compose, and writes apps/gmail-broker/gmail-broker-acct-ALIAS.enc.yaml (refresh_token + email). On the
first run it also writes apps/gmail-broker/gmail-broker-oauth-client.enc.yaml from the client JSON.
Nothing secret is printed or passed on a command line: values go through 0600 temp files straight into sops.
Needs: kubectl, sops (PATH or $SOPS), a browser. Then commit the new .enc.yaml and add the account to the
broker's projected volume (see apps/gmail-broker/gmail-broker.yaml).
"""

import argparse
import base64
import hashlib
import http.server
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.parse
import urllib.request
import webbrowser

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly", "https://www.googleapis.com/auth/gmail.compose"]


def repo_root():
    return subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()


def find_sops():
    for cand in (os.environ.get("SOPS"), shutil.which("sops")):
        if cand and os.access(cand, os.X_OK):
            return cand
    sys.exit("sops not found: put it on PATH or set SOPS=/path/to/sops")


def write_secret(sops, out_path, name, values):
    """kubectl create secret --from-file (0600 temp files) | sops encrypt > out_path."""
    with tempfile.TemporaryDirectory() as d:
        os.chmod(d, 0o700)
        args = ["kubectl", "create", "secret", "generic", name, "--namespace=default", "--dry-run=client", "-o", "yaml"]
        for k, v in values.items():
            p = os.path.join(d, k)
            with open(os.open(p, os.O_WRONLY | os.O_CREAT, 0o600), "w") as f:
                f.write(v)
            args.append(f"--from-file={k}={p}")
        manifest = subprocess.check_output(args)
    enc = subprocess.run([sops, "encrypt", "--filename-override", out_path, "--input-type", "yaml",
                          "--output-type", "yaml", "/dev/stdin"], input=manifest, capture_output=True, check=True)
    with open(out_path, "wb") as f:
        f.write(enc.stdout)
    text = enc.stdout.decode()
    for k, v in values.items():
        if v and v in text:
            os.remove(out_path)
            sys.exit(f"refusing: plaintext {k} found in {out_path}")
    print(f"wrote {os.path.relpath(out_path, repo_root())} ({len(values)} encrypted fields)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("alias", help="short account nickname, e.g. personal (lowercase letters, digits, dashes)")
    ap.add_argument("--client", default=os.path.expanduser("~/Downloads/hermes-gmail-client.json"))
    a = ap.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,30}", a.alias):
        sys.exit("alias must be lowercase letters/digits/dashes")
    with open(a.client) as f:
        cj = json.load(f)
    c = cj.get("installed") or sys.exit("client JSON is not a 'Desktop app' client (no 'installed' key)")
    sops, root = find_sops(), repo_root()
    outdir = os.path.join(root, "apps", "gmail-broker")

    client_path = os.path.join(outdir, "gmail-broker-oauth-client.enc.yaml")
    if not os.path.exists(client_path):
        write_secret(sops, client_path, "gmail-broker-oauth-client",
                     {"client_id": c["client_id"], "client_secret": c["client_secret"]})

    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    state = secrets.token_urlsafe(24)
    got = {}

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def do_GET(self):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            if q.get("state", [""])[0] == state and ("code" in q or "error" in q):
                got.update({k: v[0] for k, v in q.items()})
                msg = "Done. You can close this tab and go back to the terminal."
            else:
                msg = "Waiting for Google..."
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(msg.encode())

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    redirect = f"http://127.0.0.1:{srv.server_port}"
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode({
        "client_id": c["client_id"], "redirect_uri": redirect, "response_type": "code", "scope": " ".join(SCOPES),
        "access_type": "offline", "prompt": "consent", "state": state,
        "code_challenge": challenge, "code_challenge_method": "S256"})
    print(f"Opening Google sign-in for account '{a.alias}'. If no browser opens, visit:\n{url}\n")
    webbrowser.open(url)
    t = threading.Thread(target=lambda: [srv.handle_request() for _ in iter(lambda: "code" in got or "error" in got, True)])
    t.daemon = True
    t.start()
    t.join(timeout=600)
    srv.server_close()
    if "error" in got or "code" not in got:
        sys.exit(f"authorization failed: {got.get('error', 'timed out')}")

    data = urllib.parse.urlencode({"code": got["code"], "client_id": c["client_id"], "client_secret": c["client_secret"],
                                   "redirect_uri": redirect, "grant_type": "authorization_code",
                                   "code_verifier": verifier}).encode()
    with urllib.request.urlopen(urllib.request.Request("https://oauth2.googleapis.com/token", data=data), timeout=30) as r:
        tok = json.load(r)
    granted = set(tok.get("scope", "").split())
    missing = [s.rsplit("/", 1)[1] for s in SCOPES if s not in granted]
    if missing or not tok.get("refresh_token"):
        sys.exit(f"Google did not grant everything needed (missing: {missing or 'refresh_token'}); re-run and tick all boxes.")
    req = urllib.request.Request("https://gmail.googleapis.com/gmail/v1/users/me/profile",
                                 headers={"Authorization": "Bearer " + tok["access_token"]})
    with urllib.request.urlopen(req, timeout=30) as r:
        email = json.load(r)["emailAddress"]

    write_secret(sops, os.path.join(outdir, f"gmail-broker-acct-{a.alias}.enc.yaml"), f"gmail-broker-acct-{a.alias}",
                 {"refresh_token": tok["refresh_token"], "email": email})
    print(f"Account '{a.alias}' authorized. Tell Claude to commit it and add it to the broker.")


if __name__ == "__main__":
    main()
