"""Gmail read + draft-only broker for Hermes (issue #162, Phase 5).

A tiny MCP server (streamable HTTP, JSON responses, stdlib only) that holds the Google OAuth refresh tokens
so Hermes never does. It exposes ONLY: list accounts, search/get messages, get threads, list/get drafts,
create/update drafts. There is no send tool, no generic Gmail passthrough, and every outbound Gmail call
goes through _gmail(), which allowlists method + path and refuses anything containing "send" — so even a
bug in a tool handler cannot transmit mail. (The token's gmail.compose scope CAN send at Google; that
authority lives only inside this process.)

Secrets (mounted read-only):
  /secrets/client/client_id, client_secret      Google OAuth "Desktop app" client
  /secrets/accounts/<alias>.refresh_token, <alias>.email   one pair per authorized Gmail account
  BROKER_TOKEN env                              bearer token Hermes presents
Logs carry tool name, account alias and outcome only: never addresses, subjects or bodies.
"""

import base64
import hmac
import html
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from email.message import EmailMessage
from email.utils import getaddresses
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gmail-broker")

CLIENT_DIR = os.environ.get("CLIENT_DIR", "/secrets/client")
ACCOUNTS_DIR = os.environ.get("ACCOUNTS_DIR", "/secrets/accounts")
BROKER_TOKEN = os.environ["BROKER_TOKEN"].strip()
PORT = int(os.environ.get("PORT", "8080"))
API = "https://gmail.googleapis.com/gmail/v1/users/me/"
MAX_BODY_CHARS = 40000
SERVER_INFO = {"name": "gmail-broker", "version": "1.0.0"}


def _read(path):
    with open(path) as f:
        return f.read().strip()


def load_accounts():
    accounts = {}
    if os.path.isdir(ACCOUNTS_DIR):
        for name in sorted(os.listdir(ACCOUNTS_DIR)):
            if name.endswith(".refresh_token"):
                alias = name[: -len(".refresh_token")]
                email_path = os.path.join(ACCOUNTS_DIR, alias + ".email")
                accounts[alias] = {
                    "refresh_token": _read(os.path.join(ACCOUNTS_DIR, name)),
                    "email": _read(email_path) if os.path.exists(email_path) else "",
                }
    return accounts


ACCOUNTS = load_accounts()
_tokens = {}  # alias -> (access_token, expiry)
_token_lock = threading.Lock()


class ToolError(Exception):
    pass


def access_token(alias):
    with _token_lock:
        tok = _tokens.get(alias)
        if tok and tok[1] > time.time() + 60:
            return tok[0]
        data = urllib.parse.urlencode({
            "client_id": _read(os.path.join(CLIENT_DIR, "client_id")),
            "client_secret": _read(os.path.join(CLIENT_DIR, "client_secret")),
            "refresh_token": ACCOUNTS[alias]["refresh_token"],
            "grant_type": "refresh_token",
        }).encode()
        req = urllib.request.Request("https://oauth2.googleapis.com/token", data=data, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                body = json.load(r)
        except urllib.error.HTTPError as e:
            err = json.loads(e.read() or b"{}").get("error", str(e.code))
            log.error("token refresh failed account=%s error=%s", alias, err)
            raise ToolError(f"Google token refresh failed for '{alias}' ({err}); re-run the auth helper for it.")
        _tokens[alias] = (body["access_token"], time.time() + int(body.get("expires_in", 3600)))
        return body["access_token"]


# The whole outbound surface. Anything not matching is refused before a request is built.
_ALLOWED = [
    ("GET", re.compile(r"^profile$")),
    ("GET", re.compile(r"^messages(/[A-Za-z0-9_-]+)?$")),
    ("GET", re.compile(r"^threads/[A-Za-z0-9_-]+$")),
    ("GET", re.compile(r"^drafts(/[A-Za-z0-9_-]+)?$")),
    ("POST", re.compile(r"^drafts$")),
    ("PUT", re.compile(r"^drafts/[A-Za-z0-9_-]+$")),
]


def _gmail(alias, method, path, query=None, body=None):
    if "send" in path.lower() or not any(m == method and rx.match(path) for m, rx in _ALLOWED):
        log.error("BLOCKED outbound gmail call method=%s path=%s", method, path)
        raise ToolError("Blocked: that Gmail operation is not allowed by this broker.")
    url = API + path + ("?" + urllib.parse.urlencode(query, doseq=True) if query else "")
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": "Bearer " + access_token(alias), "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        detail = (json.loads(e.read() or b"{}").get("error") or {}).get("message", "")
        raise ToolError(f"Gmail API error {e.code}: {detail}")


def _account(args):
    alias = str(args.get("account", "")).strip()
    if alias not in ACCOUNTS:
        raise ToolError(f"Unknown account '{alias}'. Known accounts: {', '.join(ACCOUNTS) or '(none)'}.")
    return alias


def _headers(payload):
    return {h["name"].lower(): h["value"] for h in (payload or {}).get("headers", [])}


def _b64d(data):
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode("utf-8", "replace")


def _text_body(payload):
    plain, htmls = [], []

    def walk(p):
        mt = p.get("mimeType", "")
        data = (p.get("body") or {}).get("data")
        if data and mt == "text/plain":
            plain.append(_b64d(data))
        elif data and mt == "text/html":
            htmls.append(_b64d(data))
        for c in p.get("parts", []) or []:
            walk(c)

    walk(payload or {})
    if plain:
        text = "\n".join(plain)
    else:
        raw = "\n".join(htmls)
        raw = re.sub(r"(?is)<(script|style).*?</\1>", "", raw)
        raw = re.sub(r"(?i)<br\s*/?>|</p>|</div>", "\n", raw)
        text = html.unescape(re.sub(r"<[^>]+>", "", raw))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:MAX_BODY_CHARS] + ("\n[truncated]" if len(text) > MAX_BODY_CHARS else "")


def _attachments(payload):
    out = []

    def walk(p):
        if p.get("filename"):
            out.append({"filename": p["filename"], "mimeType": p.get("mimeType"),
                        "size": (p.get("body") or {}).get("size")})
        for c in p.get("parts", []) or []:
            walk(c)

    walk(payload or {})
    return out


def _summary(m):
    h = _headers(m.get("payload"))
    return {"id": m["id"], "threadId": m.get("threadId"), "date": h.get("date"), "from": h.get("from"),
            "to": h.get("to"), "subject": h.get("subject"), "snippet": html.unescape(m.get("snippet", "")),
            "labels": m.get("labelIds", [])}


def _full(m):
    h = _headers(m.get("payload"))
    return {**_summary(m), "cc": h.get("cc"), "message_id_header": h.get("message-id"),
            "body": _text_body(m.get("payload")), "attachments": _attachments(m.get("payload"))}


def t_list_accounts(args):
    return [{"account": a, "email": v["email"]} for a, v in ACCOUNTS.items()]


def t_search_messages(args):
    alias = _account(args)
    n = max(1, min(int(args.get("max_results", 10)), 50))
    res = _gmail(alias, "GET", "messages", {"q": args.get("query", ""), "maxResults": n})
    out = []
    for ref in res.get("messages", []):
        m = _gmail(alias, "GET", f"messages/{ref['id']}", {
            "format": "metadata", "metadataHeaders": ["From", "To", "Subject", "Date"]})
        out.append(_summary(m))
    return {"account": alias, "count": len(out), "messages": out}


def t_get_message(args):
    alias = _account(args)
    return _full(_gmail(alias, "GET", f"messages/{_id(args, 'message_id')}", {"format": "full"}))


def t_get_thread(args):
    alias = _account(args)
    t = _gmail(alias, "GET", f"threads/{_id(args, 'thread_id')}", {"format": "full"})
    return {"threadId": t["id"], "messages": [_full(m) for m in t.get("messages", [])]}


def t_list_drafts(args):
    alias = _account(args)
    n = max(1, min(int(args.get("max_results", 10)), 50))
    res = _gmail(alias, "GET", "drafts", {"maxResults": n, **({"q": args["query"]} if args.get("query") else {})})
    out = []
    for d in res.get("drafts", []):
        full = _gmail(alias, "GET", f"drafts/{d['id']}", {"format": "metadata"})
        out.append({"draft_id": d["id"], **_summary(full["message"])})
    return {"account": alias, "drafts": out}


def t_get_draft(args):
    alias = _account(args)
    d = _gmail(alias, "GET", f"drafts/{_id(args, 'draft_id')}", {"format": "full"})
    return {"draft_id": d["id"], **_full(d["message"])}


def _id(args, key):
    v = str(args.get(key, "")).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", v):
        raise ToolError(f"'{key}' is required and must be a Gmail id.")
    return v


def _addr_list(v):
    if not v:
        return ""
    if isinstance(v, list):
        v = ", ".join(v)
    pairs = getaddresses([str(v)])
    if not all("@" in a for _, a in pairs):
        raise ToolError(f"Invalid address list: {v!r}")
    return ", ".join(str(v).split("\n"))


def _build_message(alias, args):
    """Build the MIME message for a draft. A reply keeps Gmail threading (threadId, In-Reply-To, References)."""
    msg = EmailMessage()
    thread_id = None
    subject = str(args.get("subject", "") or "")
    reply_to = args.get("reply_to_message_id")
    if reply_to:
        orig = _gmail(alias, "GET", f"messages/{_id({'m': reply_to}, 'm')}", {
            "format": "metadata", "metadataHeaders": ["Subject", "From", "Reply-To", "Message-ID", "References"]})
        h = _headers(orig.get("payload"))
        thread_id = orig.get("threadId")
        if h.get("message-id"):
            msg["In-Reply-To"] = h["message-id"]
            msg["References"] = (h.get("references", "") + " " + h["message-id"]).strip()
        if not subject:
            s = h.get("subject", "")
            subject = s if s.lower().startswith("re:") else f"Re: {s}"
        if not args.get("to"):
            args = {**args, "to": h.get("reply-to") or h.get("from", "")}
    to = _addr_list(args.get("to"))
    if not to:
        raise ToolError("'to' is required (or give reply_to_message_id).")
    msg["To"] = to
    if args.get("cc"):
        msg["Cc"] = _addr_list(args["cc"])
    if args.get("bcc"):
        msg["Bcc"] = _addr_list(args["bcc"])
    if ACCOUNTS[alias]["email"]:
        msg["From"] = ACCOUNTS[alias]["email"]
    msg["Subject"] = subject
    msg.set_content(str(args.get("body", "")))
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    return {"raw": raw, **({"threadId": thread_id} if thread_id else {})}


def t_create_draft(args):
    alias = _account(args)
    d = _gmail(alias, "POST", "drafts", body={"message": _build_message(alias, args)})
    return {"draft_id": d["id"], "message_id": d["message"]["id"], "threadId": d["message"].get("threadId"),
            "status": "saved as an UNSENT draft; the user reviews and sends it in Gmail"}


def t_update_draft(args):
    alias = _account(args)
    draft_id = _id(args, "draft_id")
    d = _gmail(alias, "PUT", f"drafts/{draft_id}", body={"id": draft_id, "message": _build_message(alias, args)})
    return {"draft_id": d["id"], "message_id": d["message"]["id"], "threadId": d["message"].get("threadId"),
            "status": "draft replaced (still UNSENT)"}


_ACCT = {"type": "string", "description": "Account alias from list_accounts (required)."}
_DRAFT_FIELDS = {
    "account": _ACCT,
    "to": {"type": "string", "description": "Comma-separated recipients. Optional for replies (defaults to the sender)."},
    "cc": {"type": "string"}, "bcc": {"type": "string"},
    "subject": {"type": "string", "description": "Optional for replies (defaults to 'Re: <original>')."},
    "body": {"type": "string", "description": "Plain-text body."},
    "reply_to_message_id": {"type": "string", "description": "Gmail message id to reply to; keeps the thread."},
}
TOOLS = {
    "list_accounts": (t_list_accounts, "List the Gmail accounts this broker can read and draft in.",
                      {"type": "object", "properties": {}}),
    "search_messages": (t_search_messages, "Search one account with Gmail query syntax (e.g. 'is:unread newer_than:2d').",
                        {"type": "object", "required": ["account"], "properties": {
                            "account": _ACCT, "query": {"type": "string"},
                            "max_results": {"type": "integer", "minimum": 1, "maximum": 50}}}),
    "get_message": (t_get_message, "Get one message with its plain-text body.",
                    {"type": "object", "required": ["account", "message_id"],
                     "properties": {"account": _ACCT, "message_id": {"type": "string"}}}),
    "get_thread": (t_get_thread, "Get every message in a thread.",
                   {"type": "object", "required": ["account", "thread_id"],
                    "properties": {"account": _ACCT, "thread_id": {"type": "string"}}}),
    "list_drafts": (t_list_drafts, "List saved drafts in one account.",
                    {"type": "object", "required": ["account"], "properties": {
                        "account": _ACCT, "query": {"type": "string"},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": 50}}}),
    "get_draft": (t_get_draft, "Get one draft.",
                  {"type": "object", "required": ["account", "draft_id"],
                   "properties": {"account": _ACCT, "draft_id": {"type": "string"}}}),
    "create_draft": (t_create_draft, "Save a new UNSENT draft (or a threaded draft reply). This broker cannot send mail; "
                     "the user sends drafts themselves in Gmail.",
                     {"type": "object", "required": ["account", "body"], "properties": _DRAFT_FIELDS}),
    "update_draft": (t_update_draft, "Replace the contents of an existing draft (still unsent).",
                     {"type": "object", "required": ["account", "draft_id", "body"],
                      "properties": {**_DRAFT_FIELDS, "draft_id": {"type": "string"}}}),
}


def handle_rpc(msg):
    method, mid = msg.get("method"), msg.get("id")
    if mid is None:  # notification
        return None
    if method == "initialize":
        pv = (msg.get("params") or {}).get("protocolVersion", "2025-06-18")
        result = {"protocolVersion": pv, "capabilities": {"tools": {"listChanged": False}}, "serverInfo": SERVER_INFO,
                  "instructions": "Gmail read + draft-only. Sending is impossible through this server: create drafts "
                                  "and tell the user to review and send them in Gmail."}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": [{"name": n, "description": d, "inputSchema": s} for n, (_, d, s) in TOOLS.items()]}
    elif method == "tools/call":
        p = msg.get("params") or {}
        name, args = p.get("name"), p.get("arguments") or {}
        if name not in TOOLS:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": f"Unknown tool: {name}"}}
        try:
            out = TOOLS[name][0](args)
            log.info("tool=%s account=%s ok", name, args.get("account", "-"))
            result = {"content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False)}], "isError": False}
        except ToolError as e:
            log.info("tool=%s account=%s error", name, args.get("account", "-"))
            result = {"content": [{"type": "text", "text": str(e)}], "isError": True}
        except Exception:
            log.exception("tool=%s crashed", name)
            result = {"content": [{"type": "text", "text": "Internal broker error."}], "isError": True}
    else:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # no request-line logging (query strings never carry data, but keep it quiet)
        pass

    def _send(self, code, obj=None, extra=None):
        body = b"" if obj is None else json.dumps(obj).encode()
        self.send_response(code)
        if obj is not None:
            self.send_header("Content-Type", "application/json")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/healthz":
            return self._send(200, {"ok": True, "accounts": len(ACCOUNTS)})
        self._send(405, extra={"Allow": "POST"})  # no server-initiated SSE stream

    def do_DELETE(self):
        self._send(405, extra={"Allow": "POST"})

    def do_POST(self):
        if self.path.split("?")[0] != "/mcp":
            return self._send(404, {"error": "not found"})
        auth = self.headers.get("Authorization", "")
        if not (auth.startswith("Bearer ") and hmac.compare_digest(auth[7:].strip().encode(), BROKER_TOKEN.encode())):
            log.warning("unauthorized request")
            return self._send(401, {"error": "unauthorized"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n > 1_000_000:
                return self._send(413, {"error": "too large"})
            payload = json.loads(self.rfile.read(n) or b"null")
        except (ValueError, json.JSONDecodeError):
            return self._send(400, {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})
        batch = payload if isinstance(payload, list) else [payload]
        replies = [r for r in (handle_rpc(m) for m in batch if isinstance(m, dict)) if r is not None]
        if not replies:
            return self._send(202)
        self._send(200, replies if isinstance(payload, list) else replies[0])


if __name__ == "__main__":
    log.info("gmail-broker starting: %d account(s): %s", len(ACCOUNTS), ", ".join(ACCOUNTS) or "(none)")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
