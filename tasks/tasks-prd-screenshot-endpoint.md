## Relevant Files

- `requirements.txt` – Declare Python dependencies including Playwright and Prometheus.
- `main.py` – FastAPI application root; will host the new `/v1/screenshot` route.
- `app/screenshot/capture.py` – Core module that drives Playwright to capture full-page screenshots.
- `app/storage/s3.py` – Helper for uploading screenshots to S3 and forming presigned URLs.
- `app/metrics/prom.py` – Prometheus metrics setup (counters, histograms).
- `tests/test_capture.py` – Unit tests for screenshot capture logic (mock browser).
- `tests/test_api.py` – Integration tests for the HTTP endpoint using TestClient.

### Notes

- Playwright requires installing browser binaries via `playwright install chromium`. This will be scripted in Dockerfile or Railway build.
- Unit tests should mock out Playwright and S3 interactions to avoid network calls.

## Tasks

- [x] 1.0 Environment Setup and Dependencies

  - [x] 1.1 Add `playwright`, `playwright-asyncapi`, and `prometheus-client` to `requirements.txt`.
  - [x] 1.2 Create a Railway build step (or Dockerfile) that runs `playwright install chromium`.
  - [x] 1.3 Define environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `SCREENSHOT_BUCKET`.
  - [x] 1.4 Verify FastAPI and Hypercorn versions are compatible with Playwright's async loop.

- [x] 2.0 Screenshot Capture Module

  - [x] 2.1 Scaffold `app/screenshot/` package and add `__init__.py`.
  - [x] 2.2 Implement `capture.py` with async `capture_full_page(url: str) -> bytes`.
  - [x] 2.3 Scroll page to trigger lazy-loaded images and wait for network idle.
  - [x] 2.4 Inject JS/CSS to dismiss common cookie banners and overlays.
  - [x] 2.5 Implement 15-second soft timeout and 30-second hard timeout.
  - [x] 2.6 Unit-test capture logic with Playwright in headed mode disabled.

- [x] 3.0 API Endpoint for Screenshot Requests

  - [x] 3.1 Add `POST /v1/screenshot` route in `main.py` accepting `{ "url": str }`.
  - [x] 3.2 Validate URL schema and length; return `422` on failure.
  - [x] 3.3 Call capture module and store result in S3 (await S3 helper).
  - [x] 3.4 Structure successful response `{ s3_key, width, height, captured_at }` with `201 Created`.
  - [x] 3.5 Integration tests using FastAPI TestClient.

- [x] 4.0 S3 Storage Integration

  - [x] 4.1 Create `app/storage/s3.py` helper with boto3 client wrapper.
  - [x] 4.2 Upload PNG bytes to path `screenshots/{YYYY}/{MM}/{DD}/{uuid}.png`.
  - [x] 4.3 Return S3 object key.
  - [x] 4.4 Add unit tests mocking boto3 (moto).

- [x] 5.0 Observability, Monitoring, and Testing

  - [x] 5.1 Add Prometheus counters and histograms for success and latency in `app/metrics/prom.py`.
  - [x] 5.2 Expose `/metrics` endpoint via FastAPI middleware.
  - [x] 5.3 Add structured logging for each request (URL, latency, outcome).
  - [x] 5.4 Configure Railway alerts on high error rate.
  - [x] 5.5 Ensure `pytest` suite passes in CI.
