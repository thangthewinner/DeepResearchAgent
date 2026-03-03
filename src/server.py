"""
Simple FastAPI server for health checks.
Runs concurrently with the Telegram bot polling to provide a liveness endpoint
for deployment orchestrators (e.g., Docker Compose, Kubernetes, Load Balancers).
"""

from fastapi import FastAPI

from .logging_config import get_logger

logger = get_logger(__name__)

app = FastAPI(title="DeepResearchAgent Health Check")


@app.get("/health")
async def health_check():
    """Liveness probe endpoint."""
    return {"status": "ok"}


async def run_server():
    """Run the Uvicorn server programmatically."""
    import uvicorn

    config = uvicorn.Config(
        app, host="0.0.0.0", port=8080, log_level="warning", access_log=False
    )
    server = uvicorn.Server(config)
    
    logger.info("Starting FastAPI health check server on port 8080")
    await server.serve()
