from fastapi import FastAPI
from app.routes import router as api_router

# Initialize the FastAPI app
app = FastAPI(
    title="API for the DAS Homework",
    description="This api is is used to serve for the DAS Homework web application",
    version="1.0.0"
)

# Include the API router
app.include_router(api_router)

@app.get("/")
async def root():
    return {"message": "API for the DAS Homework"}
