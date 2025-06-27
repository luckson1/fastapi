from prometheus_client import Counter, Histogram

# A counter to track the number of screenshot requests.
SCREENSHOT_REQUESTS_TOTAL = Counter(
    "screenshot_requests_total",
    "Total number of screenshot requests received."
)

# A counter to track successful screenshots.
SCREENSHOT_SUCCESS_TOTAL = Counter(
    "screenshot_success_total",
    "Total number of successful screenshot captures."
)

# A counter to track failed screenshots.
SCREENSHOT_FAILURE_TOTAL = Counter(
    "screenshot_failure_total",
    "Total number of failed screenshot captures."
)

# A histogram to track the duration of screenshot captures.
SCREENSHOT_LATENCY_SECONDS = Histogram(
    "screenshot_latency_seconds",
    "Latency of screenshot captures in seconds."
) 