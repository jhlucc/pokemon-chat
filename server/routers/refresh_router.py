"""
Knowledge Refresh API Endpoints

Provides endpoints for triggering and monitoring knowledge base updates.
"""
from typing import List, Optional
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from src.knowledge.core.knowledge_refresh import get_knowledge_refresh_manager
from src.utils.logger import LogManager

logger = LogManager()

refresh_router = APIRouter(prefix="/refresh", tags=["Knowledge Refresh"])


class WebRefreshRequest(BaseModel):
    """Request to refresh from web sources."""
    urls: List[str]
    chunk_size: int = 500


class DirectoryRefreshRequest(BaseModel):
    """Request to refresh from local directory."""
    directory: str
    extensions: List[str] = [".txt", ".md", ".json"]


@refresh_router.get("/status")
async def get_refresh_status():
    """Get current knowledge refresh status and history."""
    manager = get_knowledge_refresh_manager()
    return manager.get_refresh_status()


@refresh_router.post("/web")
async def refresh_from_web(
    request: WebRefreshRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger knowledge refresh from web URLs.
    Runs in background.
    """
    manager = get_knowledge_refresh_manager()
    
    # Run in background
    background_tasks.add_task(
        manager.refresh_from_web,
        request.urls,
        request.chunk_size
    )
    
    return {
        "message": f"Refresh started for {len(request.urls)} URLs",
        "status": "running"
    }


@refresh_router.post("/directory")
async def refresh_from_directory(request: DirectoryRefreshRequest):
    """
    Trigger knowledge refresh from local directory.
    """
    directory = Path(request.directory)
    
    if not directory.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {request.directory}")
    
    manager = get_knowledge_refresh_manager()
    stats = manager.refresh_from_directory(directory, request.extensions)
    
    return {
        "message": "Refresh completed",
        "stats": stats
    }


# Bulbapedia refresh helper
BULBAPEDIA_POKEMON_URLS = [
    "https://bulbapedia.bulbagarden.net/wiki/Pikachu_(Pok%C3%A9mon)",
    "https://bulbapedia.bulbagarden.net/wiki/Charizard_(Pok%C3%A9mon)",
    "https://bulbapedia.bulbagarden.net/wiki/Mewtwo_(Pok%C3%A9mon)",
    # Add more as needed
]


@refresh_router.post("/bulbapedia")
async def refresh_from_bulbapedia(background_tasks: BackgroundTasks):
    """
    Refresh knowledge from Bulbapedia (predefined Pokemon pages).
    """
    manager = get_knowledge_refresh_manager()
    
    background_tasks.add_task(
        manager.refresh_from_web,
        BULBAPEDIA_POKEMON_URLS,
        500
    )
    
    return {
        "message": f"Started Bulbapedia refresh for {len(BULBAPEDIA_POKEMON_URLS)} pages",
        "status": "running"
    }
