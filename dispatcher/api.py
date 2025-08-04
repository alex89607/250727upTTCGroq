"""
TTC Policy Dispatcher REST API
FastAPI microservice for policy selection
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import logging
import time
import uvicorn
from contextlib import asynccontextmanager

from .dispatcher import TTCDispatcher


# Pydantic models for API
class DispatchRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for policy selection")
    user_id: Optional[str] = Field(None, description="User identifier for personalization")
    session_id: Optional[str] = Field(None, description="Session identifier")


class DispatchResponse(BaseModel):
    policy_name: str = Field(..., description="Selected TTC policy")
    confidence: float = Field(..., description="Confidence score (0-1)")
    policy_probabilities: Dict[str, float] = Field(..., description="Probabilities for all policies")
    dispatch_time_ms: float = Field(..., description="Time taken for policy selection")
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: float = Field(..., description="Unix timestamp")


class FeedbackRequest(BaseModel):
    request_id: str = Field(..., description="Request ID from dispatch response")
    policy_name: str = Field(..., description="Policy that was used")
    actual_latency: float = Field(..., description="Actual latency in seconds")
    quality_score: float = Field(..., description="Quality score (0-1)")
    additional_metrics: Optional[Dict[str, float]] = Field(None, description="Additional performance metrics")


class StatsResponse(BaseModel):
    total_requests: int
    avg_dispatch_time_ms: float
    policy_usage_stats: Dict[str, int]
    feedback_count: int
    model_stats: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    is_ready: bool
    uptime_seconds: float
    version: str = "1.0.0"


# Global dispatcher instance
dispatcher = None
start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global dispatcher
    
    # Startup
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting TTC Policy Dispatcher API...")
    
    # Initialize dispatcher
    dispatcher = TTCDispatcher()
    
    # Train model if not ready
    if not dispatcher.is_ready():
        logger.info("Training policy selector model...")
        try:
            training_results = dispatcher.train_model()
            logger.info(f"Model training completed: {training_results}")
        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            # In production, you might want to fail startup here
    
    logger.info("TTC Policy Dispatcher API ready")
    
    yield
    
    # Shutdown
    logger.info("Shutting down TTC Policy Dispatcher API...")


# Create FastAPI app
app = FastAPI(
    title="TTC Policy Dispatcher API",
    description="Intelligent policy selection for Time-to-Content optimization",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if dispatcher and dispatcher.is_ready() else "not_ready",
        is_ready=dispatcher.is_ready() if dispatcher else False,
        uptime_seconds=time.time() - start_time
    )


@app.post("/dispatch", response_model=DispatchResponse)
async def dispatch_policy(request: DispatchRequest):
    """
    Select optimal TTC policy for a given prompt
    
    This is the main endpoint that replaces manual policy selection
    """
    if not dispatcher or not dispatcher.is_ready():
        raise HTTPException(
            status_code=503, 
            detail="Dispatcher not ready. Model may still be training."
        )
    
    try:
        result = dispatcher.dispatch(
            prompt=request.prompt,
            user_id=request.user_id,
            session_id=request.session_id
        )
        
        return DispatchResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dispatch failed: {str(e)}")


@app.post("/feedback")
async def add_feedback(request: FeedbackRequest, background_tasks: BackgroundTasks):
    """
    Add performance feedback for online learning
    
    Call this after running inference with the selected policy
    """
    if not dispatcher:
        raise HTTPException(status_code=503, detail="Dispatcher not available")
    
    try:
        # Prepare metrics
        metrics = {
            'latency': request.actual_latency,
            'quality_score': request.quality_score
        }
        
        if request.additional_metrics:
            metrics.update(request.additional_metrics)
        
        # Add feedback
        dispatcher.add_feedback(
            request_id=request.request_id,
            policy_name=request.policy_name,
            actual_metrics=metrics
        )
        
        # Schedule background retraining check
        background_tasks.add_task(check_retrain)
        
        return {"status": "feedback_added", "request_id": request.request_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback failed: {str(e)}")


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get dispatcher performance statistics"""
    if not dispatcher:
        raise HTTPException(status_code=503, detail="Dispatcher not available")
    
    try:
        stats = dispatcher.get_stats()
        return StatsResponse(**stats)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats failed: {str(e)}")


@app.post("/benchmark")
async def benchmark_performance(n_samples: int = 1000):
    """Benchmark dispatcher performance"""
    if not dispatcher or not dispatcher.is_ready():
        raise HTTPException(status_code=503, detail="Dispatcher not ready")
    
    if n_samples > 10000:
        raise HTTPException(status_code=400, detail="n_samples too large (max 10000)")
    
    try:
        results = dispatcher.benchmark_performance(n_samples)
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


@app.post("/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    """Manually trigger model retraining"""
    if not dispatcher:
        raise HTTPException(status_code=503, detail="Dispatcher not available")
    
    background_tasks.add_task(retrain_model)
    
    return {"status": "retraining_scheduled"}


async def check_retrain():
    """Background task to check if retraining is needed"""
    if dispatcher:
        retrained = dispatcher.retrain_if_needed()
        if retrained:
            logging.getLogger(__name__).info("Model retrained with new feedback")


async def retrain_model():
    """Background task to retrain the model"""
    if dispatcher:
        try:
            logging.getLogger(__name__).info("Starting manual model retraining...")
            training_results = dispatcher.train_model()
            logging.getLogger(__name__).info(f"Manual retraining completed: {training_results}")
        except Exception as e:
            logging.getLogger(__name__).error(f"Manual retraining failed: {e}")


# Additional utility endpoints
@app.get("/policies")
async def get_available_policies():
    """Get list of available TTC policies"""
    if not dispatcher or not dispatcher.is_ready():
        raise HTTPException(status_code=503, detail="Dispatcher not ready")
    
    try:
        policies = dispatcher.policy_selector.policies
        return {"policies": policies}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get policies: {str(e)}")


@app.get("/feature_importance")
async def get_feature_importance():
    """Get feature importance from the trained model"""
    if not dispatcher or not dispatcher.is_ready():
        raise HTTPException(status_code=503, detail="Dispatcher not ready")
    
    try:
        importance = dispatcher.policy_selector.get_feature_importance()
        return {"feature_importance": importance}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get feature importance: {str(e)}")


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "dispatcher.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
