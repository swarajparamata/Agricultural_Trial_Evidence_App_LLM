"""`python -m agritrial_llm` starts the service with the host and port from the settings."""

import uvicorn

from .config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run("agritrial_llm.main:app", host=settings.llm_service_host, port=settings.llm_service_port)
