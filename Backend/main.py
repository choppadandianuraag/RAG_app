# ============================================================================
# FASTAPI BACKEND FOR RAG CHATBOT
# ============================================================================
"""
Complete FastAPI backend that connects your React frontend to the RAG model.

API Endpoints:
- POST /chat - Send message and get RAG response
- POST /chat/stream - Streaming response
- GET /health - Health check
- GET /categories - Get available categories
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
import asyncio
import json
from contextlib import asynccontextmanager

# Import your RAG components (we'll set these up)
from rag_engine import RAGEngine

# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================

# Initialize RAG engine (loaded once at startup)
rag_engine: Optional[RAGEngine] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    global rag_engine
    print("🚀 Starting up FastAPI server...")
    print("📚 Loading RAG engine...")
    
    try:
        rag_engine = RAGEngine()
        await rag_engine.initialize()
        print("✅ RAG engine loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading RAG engine: {e}")
        raise
    
    yield
    
    # Shutdown
    print("👋 Shutting down FastAPI server...")

app = FastAPI(
    title="TechGear Customer Support API",
    description="RAG-powered customer support chatbot API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware - allows your React frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default port
        "http://localhost:3000",  # React default port
        "http://localhost:8080",
        "*"  # For development - restrict in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# ============================================================================

class ChatMessage(BaseModel):
    """Single chat message"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None

class ChatRequest(BaseModel):
    """Request for chat endpoint"""
    message: str
    history: Optional[List[ChatMessage]] = []
    category: Optional[str] = None

class ChatResponse(BaseModel):
    """Response from chat endpoint"""
    response: str
    category: Optional[str] = None
    retrieved_docs: Optional[List[Dict]] = None
    confidence: Optional[float] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    rag_engine: str
    version: str

class Category(BaseModel):
    """Category information"""
    id: str
    name: str
    description: str
    icon: Optional[str] = None

class IngestRequest(BaseModel):
    texts: List[str]

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "TechGear Customer Support API",
        "docs": "/docs",
    }
@app.post("/ingest")
async def ingest(req: IngestRequest):
    if not req.texts:
        return {"status": "no data"}

    # Add documents to Chroma
    rag_engine.vector_store.add_texts(req.texts)

    return {
        "status": "success",
        "ingested": len(req.texts)
    }


# @app.get("/categories", response_model=List[Category], tags=["Categories"])
# async def get_categories():
#     """
#     Get available support categories
#     """
#     categories = [
#         Category(
#             id="refund",
#             name="Refund Status",
#             description="Questions about refund processing and status",
#             icon="💰"
#         ),
#         Category(
#             id="returns",
#             name="Returns",
#             description="Return policy and procedures",
#             icon="↩️"
#         ),
#         Category(
#             id="payment",
#             name="Payment Issues",
#             description="Payment and billing related questions",
#             icon="💳"
#         ),
#         Category(
#             id="shipping",
#             name="Shipping",
#             description="Shipping times and tracking",
#             icon="📦"
#         ),
#         Category(
#             id="damaged",
#             name="Damaged Items",
#             description="Report damaged or defective products",
#             icon="⚠️"
#         ),
#         Category(
#             id="other",
#             name="Other",
#             description="General questions",
#             icon="❓"
#         )
#     ]
#     return categories

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Main chat endpoint - get response from RAG model
    
    Args:
        request: ChatRequest with message and optional history
        
    Returns:
        ChatResponse with answer and metadata
    """
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        # Get response from RAG engine
        result = await rag_engine.get_response(
            query=request.message,
            history=request.history,
            category=request.category
        )
        
        return ChatResponse(
            response=result["answer"],
            category=result.get("category"),
            retrieved_docs=result.get("docs"),
            confidence=result.get("confidence")
        )
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# @app.post("/chat/stream", tags=["Chat"])
# async def chat_stream(request: ChatRequest):
#     """
#     Streaming chat endpoint - returns response token by token
    
#     Args:
#         request: ChatRequest with message and optional history
        
#     Returns:
#         StreamingResponse with Server-Sent Events
#     """
#     if not rag_engine:
#         raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
#     if not request.message or not request.message.strip():
#         raise HTTPException(status_code=400, detail="Message cannot be empty")
    
#     async def generate_stream():
#         """Generate streaming response"""
#         try:
#             async for chunk in rag_engine.get_streaming_response(
#                 query=request.message,
#                 history=request.history,
#                 category=request.category
#             ):
#                 # Format as Server-Sent Event
#                 yield f"data: {json.dumps({'content': chunk})}\n\n"
#                 await asyncio.sleep(0.01)  # Small delay for smooth streaming
            
#             # Send end signal
#             yield f"data: {json.dumps({'done': True})}\n\n"
            
#         except Exception as e:
#             error_msg = f"Error: {str(e)}"
#             yield f"data: {json.dumps({'error': error_msg})}\n\n"
    
#     return StreamingResponse(
#         generate_stream(),
#         media_type="text/event-stream",
#         headers={
#             "Cache-Control": "no-cache",
#             "Connection": "keep-alive",
#         }
#     )

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return {
        "error": "Not Found",
        "message": "The requested endpoint does not exist",
        "path": str(request.url)
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors"""
    return {
        "error": "Internal Server Error",
        "message": "An unexpected error occurred"
    }
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested endpoint does not exist",
            "path": str(request.url)
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred"
        }
    )

# ============================================================================
# RUN SERVER
# ============================================================================

import os

PORT = int(os.getenv("PORT", 8000))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT
    )
