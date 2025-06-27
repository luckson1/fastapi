# Screenshot Capture Endpoint

## 1. Introduction / Overview

This feature introduces a new HTTP endpoint that generates full-page screenshots of public landing pages. It is intended for internal services that need reliable visual captures (e.g., thumbnail generation, content audits). The system must automatically dismiss or bypass common cookie consent banners and other obstructive pop-ups, ensuring clean screenshots.

## 2. Goals

1. Provide a deterministic, high-quality screenshot of any public URL's first page.
2. Achieve >95 % success rate on a representative sample of marketing sites, with visible cookie banners suppressed.
3. Median end-to-end latency < 5 s; p95 < 10 s.
4. Store screenshots in Amazon S3 with predictable, queryable keys for downstream processing.

## 3. User Stories

- **US-01** – _As an internal microservice_, I want to request a screenshot of a given URL so I can display a thumbnail preview.
- **US-02** – _As a monitoring job_, I want screenshots uploaded to S3 so I can run visual regression tests over time.
- **US-03** – _As a developer_, I want the API to hide cookie banners and lazy-loaded elements so the screenshot reflects the final rendered page.

## 4. Functional Requirements

1. **Endpoint** – The API **MUST** expose `POST /v1/screenshot` accepting a JSON body `{ "url": string }`.
2. **Validation** – The system **MUST** validate the URL (http/https, max length 2 048 chars).
3. **Screenshot Capture**
   1. The system **MUST** render the page in a headless browser at 1920×1080 viewport, emulating desktop.
   2. It **MUST** scroll to the bottom and back to top (or use Playwright's `fullPage: true`) to trigger lazy-loaded images.
   3. It **MUST** wait until network is idle (≥ 500 ms with ≤ 2 in-flight requests) **OR** until a 15 s hard timeout.
4. **Cookie / Pop-up Handling**
   1. The system **SHOULD** automatically click common "Accept" buttons or inject CSS to hide overlays (`[role="dialog" i]`, `.cookie`, `.consent`, etc.).
   2. It **MUST NOT** expose consent state to the target site beyond clicking/accepting.
5. **Storage** – The resulting PNG **MUST** be uploaded to S3 at `s3://{BUCKET}/screenshots/{YYYY}/{MM}/{DD}/{UUID}.png`.
6. **Response** – The API **MUST** return `201 Created` with body `{ "s3_url": string, "width": int, "height": int, "captured_at": ISO-8601 }`.
7. **Error Handling** – For unreachable or blocked pages, the API **MUST** return `422 Unprocessable Entity` with an error message.
8. **Security** – The endpoint **MUST** be protected by internal authentication (e.g., Bearer token) and rate-limited (default 30 req/min per client).
9. **Metrics & Logging** – The system **MUST** log each request's URL, latency, and outcome; and expose Prometheus metrics (`screenshot_success_total`, `screenshot_latency_ms`).

## 5. Non-Goals (Out of Scope)

- Capturing authenticated or pay-walled pages.
- Rendering dynamic user interactions beyond initial load (e.g., multi-step forms).
- Capturing mobile or device-specific breakpoints (future work).

## 6. Design Considerations

- UX is internal only; no public UI required.
- S3 bucket lifecycle policy of 90 days should be applied to manage storage costs.

## 7. Technical Considerations

- **Runtime**: Service runs on Railway as a container. Use **Playwright** with headless Chromium for consistency and robustness.
- **Concurrency**: Single stateless FastAPI app delegates heavy browser work to an async task queue (e.g., **Redis Queue** or Railway workers) to avoid blocking HTTP threads.
- **Packaging**: Use the official Playwright image (`mcr.microsoft.com/playwright/python:v1.44.0-jammy`) to simplify fonts/sandboxing.
- **Network**: Outbound traffic must respect Railway egress policies and DNS resolution.
- **Timeouts**: Overall capture hard timeout 30 s; HTTP layer timeout 35 s.

## 8. Success Metrics

- ≥ 95 % of requests succeed without manual intervention.
- Median latency < 5 s, p95 < 10 s.
- < 1 % of images contain visible cookie banners on curated QA set of 100 sites.

## 9. Open Questions

1. Which AWS account & IAM role should own the screenshot bucket?
2. Should we support JPEG output as an optional format via query param in v1?
3. Do we need regional screenshot workers (latency vs. egress cost)?
4. What retention policy should be applied beyond the proposed 90 days?
