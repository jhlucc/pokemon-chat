from typing import Any

from pydantic import BaseModel, Field


class Source(BaseModel):
    """Source information for a response"""

    title: str = Field(description="Title of the source")
    url: str | None = Field(default=None, description="URL of the source")
    content_snippet: str | None = Field(default=None, description="Relevant snippet from the source")
    score: float = Field(default=1.0, description="Relevance score")


class AgentResponse(BaseModel):
    """Standardized response from an agent"""

    content: str = Field(description="The main text response to the user")
    sources: list[Source] = Field(default=[], description="List of sources used to generate the response")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score of the response")
    tool_calls: list[dict[str, Any]] = Field(default=[], description="List of tool calls made during the process")
    metadata: dict[str, Any] = Field(default={}, description="Additional metadata")


class ErrorResponse(BaseModel):
    """Standardized error response"""

    error_code: str = Field(description="Error code")
    message: str = Field(description="Human readable error message")
    details: dict[str, Any] | None = Field(default=None, description="Error details")


class StructuredContent(BaseModel):
    """Structured content for multi-modal outputs"""

    type: str = Field(description="Type of content (text, image, table, etc.)")
    data: Any = Field(description="The actual content data")
    metadata: dict[str, Any] = Field(default={}, description="Metadata for the content")
