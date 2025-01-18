from fastapi import FastAPI
from app.routes import router as api_router
from fastapi.middleware.cors import CORSMiddleware


# Initialize the FastAPI app
app = FastAPI(
    title="API for the DAS Homework",
    description="This api is is used to serve for the DAS Homework web application",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://das-prototype.web.app"],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (e.g., GET, POST)
    allow_headers=["*"],  # Allow all headers
)

# Include the API router
app.include_router(api_router)

@app.get("/")
async def root():
    return {"message": "API for the DAS Homework"}
