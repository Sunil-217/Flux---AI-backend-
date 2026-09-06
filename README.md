# Flux AI — Backend

FastAPI service behind Close AI. Deployed on **Render Free**.

Configuration lives in `.env` (see `.env.example`). Tests: `pytest`.

---

## Render Free Keep-Alive

Render's free tier spins a service down after a period with no traffic. The next
request then pays a **cold start** while the process boots — the user sees a long
first response, not an error.

This is mitigated (not solved) by having an **external** scheduler request the
health endpoint on a timer:

```
GET https://flux-ai-backend-k3h0.onrender.com/health
```

```json
{ "status": "ok" }
```

**Recommended interval: every 10–14 minutes.**

Comfortably under Render's idle window, and far from often enough to look like
abuse. Do not go lower — a tighter loop buys nothing and is just traffic.

### What this is, and what it is not

This is **cold-start mitigation on a best-effort basis. It does not guarantee
24/7 uptime**, and nothing configured here can:

- Render may restart, redeploy, recycle or relocate the service at any time,
  entirely independently of incoming traffic.
- Free-tier instances are subject to Render's own limits and maintenance.
- A scheduler can miss runs; free schedulers make no delivery guarantee.
- A ping keeps the process warm. It does not make the platform's free tier
  behave like a paid one.

Describe the result as *"keep-alive configured / cold-start mitigation"* — not
as uptime.

### Why the ping is external

The backend deliberately does **not** ping itself. A process cannot keep itself
awake by talking to itself: once Render has spun the instance down, there is no
process left to run the timer. An internal loop would burn a worker thread
during normal operation and do nothing at the only moment it was wanted.

For the same reason there is no Render Cron Job here — that is a separate
Render service, and the point is to add no Render resources.

### Setting up a scheduler

Any free HTTP scheduler works; the endpoint needs no credentials, so there is
**nothing to configure on the backend and no key to store anywhere**. Common
free options include cron-job.org, UptimeRobot and GitHub Actions' `schedule`
trigger. Point whichever you choose at the URL above on a 10–14 minute interval,
with `GET`, no headers, and no body.

Never put a scheduler's API key or webhook secret into this repository or into
the backend's environment. The keep-alive is a one-way, unauthenticated `GET`;
it needs no secret in either direction.

### About the endpoint

`GET /health` is the cheapest route in the app: no authentication, no database
query, no LLM call, no embedding call, no outbound request, no computation. It
answers one question — *is this process up and serving?*

That narrow scope is deliberate. A health check that also verified Groq, Jina or
Postgres would report the service as down whenever a provider was rate-limited,
which is exactly when you least want a monitor crying wolf. Provider health is
reported separately and behind authentication at `GET /agent/providers`.

`GET /` also returns 200 and would work as a ping target, but `/health` is the
stable contract — the root is free to change.
