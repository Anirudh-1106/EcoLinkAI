from fastapi import FastAPI

app = FastAPI(
    title="EcoLinkAI API",
    version="1.0.0",
    description="AI-Powered Circular Economy Waste Exchange Network"
)


@app.get("/")
def root():
    return {
        "status": "running",
        "project": "EcoLinkAI",
        "version": "1.0.0"
    }