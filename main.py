"""
Production Application Entrypoint for Railway / Cloud Deployment
================================================================
Reads PORT dynamically from the environment without shell expansion issues.
"""

import os
import uvicorn

if __name__ == "__main__":
    port_str = os.environ.get("PORT", "8000")
    try:
        port = int(port_str)
    except ValueError:
        port = 8000

    print(f"Starting Greenko Energy Operations RAG Assistant on port {port}...")
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

