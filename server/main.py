import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.routers import router
from dotenv import load_dotenv
from pathlib import Path

# 强制加载.env
load_dotenv(Path(__file__).parent.parent / ".env")
app = FastAPI()
app.include_router(router)

# CORS 设置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050)

