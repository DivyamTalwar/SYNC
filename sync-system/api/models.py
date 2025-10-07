from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime


class CollaborateRequest(BaseModel):

    query: str = Field(
        ...,
        description="The task or query for agents to collaborate on",
        min_length=10,
        max_length=5000,
    )

    context: Optional[str] = Field(
        None,
        description="Optional additional context for the query",
        max_length=2000,
    )

    num_agents: int = Field(
        default=3,
        description="Number of agents to use (2-6)",
        ge=2,
        le=6,
    )

    max_rounds: int = Field(
        default=5,
        description="Maximum collaboration rounds before forced termination",
        ge=1,
        le=10,
    )

    temperature: float = Field(
        default=0.7,
        description="LLM temperature for reasoning generation",
        ge=0.0,
        le=1.0,
    )

    @validator('query')
    def validate_query(cls, v):
        """Validate query content"""
        v = v.strip()
        if len(v) < 10:
            raise ValueError("Query must be at least 10 characters")
        if not v:
            raise ValueError("Query cannot be empty")
        return v

    @validator('context')
    def validate_context(cls, v):
        """Validate context if provided"""
        if v is not None:
            v = v.strip()
            if not v:
                return None
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the top 3 factors for building sustainable cities?",
                "context": "Focus on environmental, social, and economic sustainability.",
                "num_agents": 3,
                "max_rounds": 5,
                "temperature": 0.7,
            }
        }


class AgentContribution(BaseModel):
    """Model for individual agent contribution"""
    agent_id: int
    role: str
    contribution: str


class ConvergenceInfo(BaseModel):
    """Model for convergence information"""
    round_number: int
    convergence_score: float
    state_similarity: float
    gap_magnitude: float
    is_converged: bool
    reason: str


class CollaborateResponse(BaseModel):
    """Response model for multi-agent collaboration"""

    success: bool = Field(
        ...,
        description="Whether the collaboration completed successfully"
    )

    final_answer: str = Field(
        ...,
        description="The synthesized final answer from all agents"
    )

    confidence: float = Field(
        ...,
        description="Confidence score (0-1) in the final answer",
        ge=0.0,
        le=1.0,
    )

    consensus_level: float = Field(
        ...,
        description="Consensus level (0-1) among agents",
        ge=0.0,
        le=1.0,
    )

    reasoning_summary: str = Field(
        default="",
        description="Summary of key reasoning points"
    )

    conflicts: List[str] = Field(
        default_factory=list,
        description="List of conflicts or disagreements resolved"
    )

    agent_contributions: List[AgentContribution] = Field(
        default_factory=list,
        description="Individual agent contributions"
    )

    total_rounds: int = Field(
        ...,
        description="Number of collaboration rounds completed"
    )

    total_messages: int = Field(
        ...,
        description="Total number of messages exchanged"
    )

    computation_time: float = Field(
        ...,
        description="Total computation time in seconds"
    )

    convergence_info: List[ConvergenceInfo] = Field(
        default_factory=list,
        description="Convergence metrics for each round"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "final_answer": "The top 3 factors are: 1) Green infrastructure...",
                "confidence": 0.92,
                "consensus_level": 0.88,
                "reasoning_summary": "Agents agreed on environmental, social, economic factors",
                "conflicts": [],
                "agent_contributions": [
                    {
                        "agent_id": 0,
                        "role": "Analytical Reasoner",
                        "contribution": "Provided systematic analysis of sustainability metrics"
                    }
                ],
                "total_rounds": 3,
                "total_messages": 9,
                "computation_time": 32.5,
                "convergence_info": [],
                "metadata": {}
            }
        }


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint"""

    status: str = Field(
        ...,
        description="Overall service status (healthy/degraded/unhealthy)"
    )

    service: str = Field(
        default="SYNC",
        description="Service name"
    )

    version: str = Field(
        default="0.1.0",
        description="Service version"
    )

    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Health check timestamp"
    )

    components: Dict[str, str] = Field(
        default_factory=dict,
        description="Status of individual components"
    )

    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional health details"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "service": "SYNC",
                "version": "0.1.0",
                "timestamp": "2025-10-07T10:30:00",
                "components": {
                    "llm_client": "healthy",
                    "embedding_client": "healthy",
                    "neural_components": "healthy"
                },
                "details": {
                    "uptime_seconds": 3600,
                    "total_collaborations": 42
                }
            }
        }


class ErrorResponse(BaseModel):
    """Response model for errors"""

    success: bool = False

    error: str = Field(
        ...,
        description="Error type or code"
    )

    message: str = Field(
        ...,
        description="Human-readable error message"
    )

    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details"
    )

    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Error timestamp"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "ValidationError",
                "message": "Query must be at least 10 characters",
                "details": {"field": "query", "value_length": 5},
                "timestamp": "2025-10-07T10:30:00"
            }
        }
