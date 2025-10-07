import asyncio
from typing import Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.responses import JSONResponse

from api.models import (
    CollaborateRequest,
    CollaborateResponse,
    HealthCheckResponse,
    ErrorResponse,
    AgentContribution,
    ConvergenceInfo,
)
from api.middleware import setup_middleware
from api.auth import (
    get_current_api_key,
    get_optional_api_key,
    check_rate_limit,
    generate_api_key,
    create_jwt_token,
    APIKey,
)
from src.orchestrator.coordinator import MultiAgentCoordinator
from src.utils.logging import get_logger
from config.base import config

logger = get_logger("api.main")

# Global state
app_state = {
    "start_time": datetime.now(),
    "total_collaborations": 0,
    "total_requests": 0,
    "active_collaborations": 0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager

    Handles startup and shutdown events.
    """
    # Startup
    logger.info("=" * 80)
    logger.info("SYNC API - Starting up")
    logger.info("=" * 80)
    logger.info(f"Primary Model: {config.api.primary_model}")
    logger.info(f"Aggregator Model: {config.api.aggregator_model}")
    logger.info("Ready to receive requests")
    logger.info("=" * 80)

    yield

    # Shutdown
    logger.info("=" * 80)
    logger.info("SYNC API - Shutting down")
    logger.info(f"Total collaborations: {app_state['total_collaborations']}")
    logger.info(f"Total requests: {app_state['total_requests']}")
    logger.info("=" * 80)


# Create FastAPI app
app = FastAPI(
    title="SYNC API",
    description="Multi-Agent LLM Collaboration System with Strategic Communication",
    version="0.1.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
    lifespan=lifespan,
)

# Setup middleware
setup_middleware(app)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API information"""
    return {
        "service": "SYNC API",
        "version": "0.1.0",
        "description": "Multi-Agent LLM Collaboration System",
        "docs": "/api/v1/docs",
        "health": "/api/v1/health",
    }


@app.get(
    "/api/v1/health",
    response_model=HealthCheckResponse,
    tags=["Health"],
    summary="Health check endpoint",
)
async def health_check():
    """
    Health check endpoint

    Returns the current health status of the service and its components.
    """
    app_state["total_requests"] += 1

    try:
        # Check LLM client
        llm_status = "healthy"
        try:
            from src.llm.openrouter import OpenRouterClient
            client = OpenRouterClient()
            # Just verify client initializes correctly
            if not client.api_key or client.api_key == "":
                llm_status = "unhealthy - missing API key"
            await client.close()
        except Exception as e:
            llm_status = f"unhealthy - {str(e)[:50]}"
            logger.error(f"LLM client health check failed: {e}")

        # Check embedding client
        embedding_status = "healthy"
        try:
            from src.llm.embeddings import CohereEmbeddingsClient
            client = CohereEmbeddingsClient()
            if not client.api_key or client.api_key == "":
                embedding_status = "unhealthy - missing API key"
        except Exception as e:
            embedding_status = f"unhealthy - {str(e)[:50]}"
            logger.error(f"Embedding client health check failed: {e}")

        # Check neural components
        neural_status = "healthy"
        try:
            from src.core.state import StateEncoder
            encoder = StateEncoder()
            # Just verify it initializes
        except Exception as e:
            neural_status = f"unhealthy - {str(e)[:50]}"
            logger.error(f"Neural components health check failed: {e}")

        # Determine overall status
        component_statuses = [llm_status, embedding_status, neural_status]
        if all("healthy" == s for s in component_statuses):
            overall_status = "healthy"
        elif any("unhealthy" in s for s in component_statuses):
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"

        # Compute uptime
        uptime = (datetime.now() - app_state["start_time"]).total_seconds()

        return HealthCheckResponse(
            status=overall_status,
            service="SYNC",
            version="0.1.0",
            timestamp=datetime.now(),
            components={
                "llm_client": llm_status,
                "embedding_client": embedding_status,
                "neural_components": neural_status,
            },
            details={
                "uptime_seconds": round(uptime, 2),
                "total_collaborations": app_state["total_collaborations"],
                "total_requests": app_state["total_requests"],
                "active_collaborations": app_state["active_collaborations"],
            }
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return HealthCheckResponse(
            status="unhealthy",
            service="SYNC",
            version="0.1.0",
            timestamp=datetime.now(),
            components={"error": str(e)},
            details={}
        )


@app.post(
    "/api/v1/collaborate",
    response_model=CollaborateResponse,
    tags=["Collaboration"],
    summary="Start multi-agent collaboration",
    status_code=status.HTTP_200_OK,
)
async def collaborate(
    request: CollaborateRequest,
    api_key: APIKey = Depends(check_rate_limit)
):
    """
    Start multi-agent collaboration on a query

    This endpoint orchestrates multiple AI agents to collaboratively reason
    about a query, detect cognitive gaps, communicate strategically, and
    converge on a high-quality answer.

    **Process:**
    1. Create N agents with distinct roles
    2. Each agent generates initial reasoning
    3. Agents collaborate over multiple rounds:
       - Model each other's thinking (CKM)
       - Detect cognitive gaps
       - Select strategic communication actions
       - Exchange messages
       - Check for convergence
    4. Aggregate final response from all perspectives

    **Typical Duration:** 30-60 seconds depending on query complexity

    **Cost:** ~$0.05-$0.20 per collaboration
    """
    app_state["total_requests"] += 1
    app_state["active_collaborations"] += 1

    coordinator = None

    try:
        logger.info(f"Starting collaboration - Query: {request.query[:100]}...")

        # Create coordinator
        coordinator = MultiAgentCoordinator(
            num_agents=request.num_agents,
            max_rounds=request.max_rounds,
            device="cpu",
        )

        # Set timeout (5 minutes max)
        timeout_seconds = 300

        # Run collaboration with timeout
        result = await asyncio.wait_for(
            coordinator.collaborate(
                query=request.query,
                context=request.context,
            ),
            timeout=timeout_seconds
        )

        # Convert to API response
        agent_contributions = [
            AgentContribution(
                agent_id=agent_id,
                role=coordinator.agents[agent_id].role,
                contribution=contribution
            )
            for agent_id, contribution in result.final_response.agent_contributions.items()
        ]

        convergence_info = [
            ConvergenceInfo(
                round_number=m.round_number,
                convergence_score=m.convergence_score,
                state_similarity=m.state_similarity_avg,
                gap_magnitude=m.gap_magnitude_avg,
                is_converged=m.is_converged,
                reason=m.reason
            )
            for m in result.convergence_metrics
        ]

        response = CollaborateResponse(
            success=result.success,
            final_answer=result.final_response.final_answer,
            confidence=result.final_response.confidence,
            consensus_level=result.final_response.consensus_level,
            reasoning_summary=result.final_response.reasoning_summary,
            conflicts=result.final_response.conflicts,
            agent_contributions=agent_contributions,
            total_rounds=result.total_rounds,
            total_messages=result.total_messages,
            computation_time=result.computation_time,
            convergence_info=convergence_info,
            metadata={
                "model": config.api.primary_model,
                "aggregator_model": config.api.aggregator_model,
                "num_agents": request.num_agents,
                "max_rounds": request.max_rounds,
            }
        )

        app_state["total_collaborations"] += 1
        logger.info(
            f"Collaboration completed - "
            f"Rounds: {result.total_rounds}, "
            f"Messages: {result.total_messages}, "
            f"Time: {result.computation_time:.2f}s"
        )

        return response

    except asyncio.TimeoutError:
        logger.error(f"Collaboration timed out after {timeout_seconds}s")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Collaboration timed out after {timeout_seconds} seconds"
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    except Exception as e:
        logger.error(f"Collaboration failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Collaboration failed: {str(e)}"
        )

    finally:
        # Always cleanup
        if coordinator:
            try:
                await coordinator.close()
            except Exception as e:
                logger.error(f"Error closing coordinator: {e}")

        app_state["active_collaborations"] -= 1


@app.get(
    "/api/v1/stats",
    tags=["Stats"],
    summary="Get API statistics",
)
async def get_stats(api_key: APIKey = Depends(get_optional_api_key)):
    """
    Get API usage statistics

    Returns metrics about API usage and performance.
    """
    app_state["total_requests"] += 1

    uptime = (datetime.now() - app_state["start_time"]).total_seconds()

    stats = {
        "uptime_seconds": round(uptime, 2),
        "uptime_human": f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m",
        "total_requests": app_state["total_requests"],
        "total_collaborations": app_state["total_collaborations"],
        "active_collaborations": app_state["active_collaborations"],
        "avg_requests_per_minute": round(
            app_state["total_requests"] / (uptime / 60) if uptime > 0 else 0,
            2
        ),
        "start_time": app_state["start_time"].isoformat(),
    }

    # Add API key specific stats if authenticated
    if api_key:
        stats["api_key"] = {
            "name": api_key.name,
            "total_requests": api_key.total_requests,
            "rate_limit": api_key.rate_limit_per_minute,
            "last_used": api_key.last_used_at.isoformat() if api_key.last_used_at else None,
        }

    return stats


@app.post(
    "/api/v1/auth/generate-key",
    tags=["Authentication"],
    summary="Generate new API key (admin only)",
)
async def generate_key(
    name: str,
    rate_limit: int = 60,
    api_key: APIKey = Depends(get_current_api_key)
):
    """
    Generate new API key

    **Requires admin authentication**

    Args:
        name: Name/description for the key
        rate_limit: Requests per minute limit (default: 60)

    Returns:
        New API key (store this securely!)
    """
    # Check if admin
    if api_key.name != "Admin Key":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admin can generate API keys"
        )

    new_key = generate_api_key(name, rate_limit)

    logger.info(f"Generated new API key: {name}")

    return {
        "api_key": new_key,
        "name": name,
        "rate_limit_per_minute": rate_limit,
        "message": "Store this key securely! It won't be shown again.",
    }


@app.post(
    "/api/v1/auth/token",
    tags=["Authentication"],
    summary="Get JWT token from API key",
)
async def get_token(api_key: APIKey = Depends(get_current_api_key)):
    """
    Exchange API key for JWT token

    Provide your API key in Authorization header to get a JWT token.
    JWT tokens expire after 24 hours.

    Returns:
        JWT token and expiration time
    """
    token = create_jwt_token(api_key.key_hash)

    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in_hours": 24,
        "api_key_name": api_key.name,
    }


@app.get(
    "/api/v1/auth/verify",
    tags=["Authentication"],
    summary="Verify authentication",
)
async def verify_auth(api_key: APIKey = Depends(get_current_api_key)):
    """
    Verify that authentication is working

    Returns information about the authenticated API key.
    """
    return {
        "authenticated": True,
        "api_key_name": api_key.name,
        "rate_limit_per_minute": api_key.rate_limit_per_minute,
        "total_requests": api_key.total_requests,
        "last_used": api_key.last_used_at.isoformat() if api_key.last_used_at else None,
    }


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="ValidationError",
            message=str(exc),
            timestamp=datetime.now()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle all other exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            details={"type": type(exc).__name__},
            timestamp=datetime.now()
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting SYNC API server...")
    logger.info("Swagger docs will be available at: http://localhost:8000/api/v1/docs")

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes (dev only)
        log_level="info",
    )
