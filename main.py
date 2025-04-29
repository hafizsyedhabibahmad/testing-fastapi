import logging
from fastapi import FastAPI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
async def root():
    logger.info("Root endpoint called")
    return {"message": "Welcome to the FastAPI app on Railway!"}

@app.get("/hello/{name}")
async def say_hello(name: str):
    logger.info(f"Hello endpoint called with name: {name}")
    return {"message": f"Hello, {name}!"}
