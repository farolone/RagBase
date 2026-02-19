from fastapi import APIRouter

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


@router.get("/status")
def pipeline_status():
    """Get pipeline status from Prefect if available."""
    import httpx
    try:
        resp = httpx.get("http://localhost:4200/api/flow_runs", timeout=5.0, params={"sort": "EXPECTED_START_TIME_DESC", "limit": 10})
        if resp.status_code == 200:
            runs = resp.json()
            return {"status": "connected", "recent_runs": runs[:10]}
    except Exception:
        pass
    return {"status": "unavailable", "recent_runs": []}


@router.post("/trigger")
def trigger_pipeline():
    """Trigger a manual pipeline run."""
    import subprocess
    try:
        result = subprocess.Popen(
            ["/root/.local/bin/uv", "run", "python", "-m", "rag.pipeline.flows"],
            cwd="/root/rag",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return {"triggered": True, "pid": result.pid}
    except Exception as e:
        return {"triggered": False, "error": str(e)}
