import os


DEFAULT_CORS_ALLOWED_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


def get_cors_allowed_origins() -> list[str]:
    """Return explicit browser origins allowed to call the API."""
    configured_origins = os.getenv("CORS_ALLOWED_ORIGINS")
    if configured_origins is None:
        return list(DEFAULT_CORS_ALLOWED_ORIGINS)

    origins = [
        origin.strip().rstrip("/")
        for origin in configured_origins.split(",")
        if origin.strip()
    ]
    if "*" in origins:
        raise ValueError(
            "CORS_ALLOWED_ORIGINS must contain explicit origins, not '*'."
        )

    return list(dict.fromkeys(origins))
