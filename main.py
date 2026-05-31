from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db import Base, engine
from app import models  # noqa: F401  (register models on Base before create_all)

from app.api.routes import upload
from app.api.routes import chat
from app.api.routes import delete
from app.api.routes import auth

# Create database tables if they don't exist yet.
Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(upload.router)
app.include_router(chat.router)
app.include_router(delete.router)


@app.get("/")
def home():
    return {"message": "Close AI Backend Running"}
