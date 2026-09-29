# Cockpit operator access

## Opening the cockpit on the Spark (plain steps)

The cockpit now asks for a password-like access token before it shows anything or lets anyone press a button. Only the Spark itself can reach it.

1. Open a terminal on the Spark and type:

   ```
   atlas-vault get AIEN_COCKPIT_TOKEN
   ```

   It prints one long line of letters and numbers. Select it and copy it. That line is the cockpit key; do not paste it into chats, issues or screenshots.
2. Open a web browser on the Spark and go to `http://127.0.0.1:18095` (or `http://localhost:18095`).
3. The browser shows a page titled "Sign in to AIEN". Paste the key into the box and press "Sign in".
4. The normal cockpit appears. The sign-in lasts one hour, and restarting the cockpit service also signs you out. After that, the browser returns to the sign-in page; repeat steps 1 to 3.

From the MacBook, open a tunnel first (Terminal on the Mac: `ssh -L 18095:127.0.0.1:18095 <spark>`, leave it open), then follow steps 2 to 4 in a Mac browser at `http://127.0.0.1:18095`. The LAN address and the Tailscale address no longer work on purpose.

## Token provisioning

Both keys live in atlas-vault, the machine's TPM-sealed key store (encrypted by the Spark's security chip, kept under `~/.config/atlas/vault` with owner-only permissions). They never sit in a repository or a plain text file, and the services never print them to logs.

| Key | Who reads it |
| --- | --- |
| `AIEN_COCKPIT_TOKEN` | spark-cockpit-rs at start; Drake pastes it at `/login` |
| `AIEN_MAIL_API_TOKEN` | spark-mail (checks it) and the cockpit (sends it when forwarding mail requests) |

Create or rotate a key, then restart the service that reads it:

```
head -c 48 /dev/urandom | base64 | tr -d '/+=\n' | atlas-vault add AIEN_COCKPIT_TOKEN
```

Store `AIEN_COCKPIT_TOKEN` and `AIEN_MAIL_API_TOKEN` with `atlas-vault add <NAME>`, which reads the value from standard input. Use a random token of at least 32 ASCII characters. The cockpit and mail service resolve these through process memory or `atlas-vault get <NAME>`. Plaintext token files are not supported. Missing credentials fail closed. Restart the cockpit after rotating its token; existing sessions are cleared.

## What is protected

- Listener: `127.0.0.1:18095` by default. `AIEN_COCKPIT_ADDR` changes it explicitly.
- Allowed browser origins: `http://127.0.0.1:18095` and `http://localhost:18095`. `AIEN_COCKPIT_ORIGINS` (comma separated, exact origins) replaces the list. Use HTTPS for any remote browser origin.
- Every route except `/login` and `/auth/session` needs either `Authorization: Bearer <token>` or a signed-in browser session. Browser sessions are random, HttpOnly, SameSite=Strict cookies with a server-enforced one-hour lifetime; HTTPS origins also get Secure cookies.
- Browser mutations (anything other than GET/HEAD) and terminal upgrades additionally need an allowed Origin. A request carrying a foreign Origin gets 403.
- The terminal (`/ws/terminal`) keeps its own check on top: exact loopback Origin and a loopback peer address. Remote origins cannot open a shell.
- Health reads stay open for local monitors: `GET /api/pulse` and `GET /api/status` answer without a token, but only when the request's Host is `127.0.0.1`, `localhost` or `[::1]` (blocks DNS-rebinding pages) and it carries no foreign Origin. aien-cli, spark-debugger, spark-supervisor and bench_inference_stack only use these reads.

The token authorizes the operator's administrative surface, not an untrusted agent principal.
