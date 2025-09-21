"""
Unified API server for AMPilot Manage Agent

This FastAPI server exposes a unified interface for the frontend and orchestrates
requests across the three specialized agents using LangGraph-based workflow.
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any
from configuration import RESEARCH_AGENT_URL, AMP_DESIGNER_AGENT_URL, DATA_ANALYSIS_AGENT_URL
import aiohttp


try:
    from multi_agent_workflow import MultiAgentWorkflow
    workflow_available = True
    print("✓ MultiAgentWorkflow imported successfully")
except ImportError as e:
    print(f"✗ MultiAgentWorkflow import failed: {e}")
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    try:
        from multi_agent_workflow import MultiAgentWorkflow
        workflow_available = True
        print("✓ MultiAgentWorkflow imported successfully (with path fix)")
    except ImportError as e2:
        print(f"✗ MultiAgentWorkflow import failed even with path fix: {e2}")
        workflow_available = False
        MultiAgentWorkflow = None

logger = logging.getLogger("manage_agent")
logging.basicConfig(level=logging.INFO)

# Paths
REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = REPO_ROOT / "frontend"
STATIC_DIR = FRONTEND_DIR

# Initialize app
app = FastAPI(title="AMPilot Manage Agent", version="1.0.0")

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory session store (simple)
SESSIONS: Dict[str, Dict[str, Any]] = {}

# Initialize workflow instance
if workflow_available:
    try:
        WORKFLOW = MultiAgentWorkflow()
        print("✓ MultiAgentWorkflow instance created successfully")
    except Exception as e:
        print(f"✗ Failed to create workflow instance: {e}")
        workflow_available = False
        WORKFLOW = None
else:
    WORKFLOW = None


class ChatMessage(BaseModel):
    message: str
    session_id: str
    context: Optional[Dict[str, Any]] = None


class DebugRouteRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None


@app.get("/")
async def root_index():
    """Serve the main frontend page or API info."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    else:
        # Return API information if frontend not available
        return {
            "message": "AMPilot Manage Agent API",
            "status": "running",
            "workflow_available": workflow_available,
            "timestamp": datetime.now().isoformat(),
            "endpoints": {
                "health": "GET /health - Health check",
                "chat": "POST /chat - Chat with AMPilot",
                "test_sequence": "POST /test-sequence - Test sequence query",
                "test_properties": "POST /test-properties - Test properties query",
                "test_greeting": "GET /test-greeting - Test greeting",
                "websocket": "WS /ws/{session_id} - WebSocket chat",
                "debug_route": "POST /debug-route - Inspect LLM routing decision",
                "e2e_smoke": "GET /e2e-smoke - Run 3-agent end-to-end smoke tests"
            }
        }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "workflow_available": workflow_available,
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/agents-health")
async def agents_health():
    async def ping(url: str):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{url}/health", timeout=3) as resp:
                    data = await resp.json()
                    return {"ok": resp.status == 200, "status": data}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    results = {
        "research": await ping(RESEARCH_AGENT_URL),
        "amp_designer": await ping(AMP_DESIGNER_AGENT_URL),
        "data_analysis": await ping(DATA_ANALYSIS_AGENT_URL),
    }
    return {"timestamp": datetime.now().isoformat(), "agents": results}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    SESSIONS.pop(session_id, None)
    return {"status": "deleted", "session_id": session_id}


@app.post("/test-sequence")
async def test_sequence_query():
    """Test the sequence-to-properties query that was failing"""
    if not workflow_available or WORKFLOW is None:
        return JSONResponse({"error": "Workflow not available"}, status_code=503)

    test_query = "What is the property of the AMP with LPLLAGLAANFLPKIFCKITRK sequence"
    try:
        result = await WORKFLOW.process_query(test_query)
        return {
            "test": "sequence-to-properties",
            "query": test_query,
            "success": result.get("success", False),
            "response": result.get("response", "No response"),
            "metadata": result.get("metadata", {}),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.exception("Sequence test failed")
        return JSONResponse({"error": str(e), "test": "sequence-to-properties"}, status_code=500)


@app.post("/test-properties")
async def test_properties_query():
    """Test the properties-to-sequence query"""
    if not workflow_available or WORKFLOW is None:
        return JSONResponse({"error": "Workflow not available"}, status_code=503)

    test_query = "Find me an antimicrobial peptide with a sequence length of 20-25 and an inhibitory effect on Staphylococcus aureus"
    try:
        result = await WORKFLOW.process_query(
            user_query=test_query,
            session_id="test-properties",
            user_context={}
        )
        return {
            "test": "properties-to-sequence",
            "query": test_query,
            "success": result.get("success", False),
            "response": result.get("response", "No response"),
            "metadata": result.get("metadata", {}),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.exception("Properties test failed")
        return JSONResponse({"error": str(e), "test": "properties-to-sequence"}, status_code=500)


@app.get("/test-greeting")
async def test_greeting():
    """Test basic greeting functionality"""
    if not workflow_available or WORKFLOW is None:
        return JSONResponse({"error": "Workflow not available"}, status_code=503)

    try:
        result = await WORKFLOW.process_query(
            user_query="hello",
            session_id="test-greeting",
            user_context={}
        )
        return {
            "test": "greeting",
            "query": "hello",
            "success": result.get("success", False),
            "response": result.get("response", "No response"),
            "metadata": result.get("metadata", {}),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.exception("Greeting test failed")
        return JSONResponse({"error": str(e), "test": "greeting"}, status_code=500)


@app.post("/chat")
async def chat_endpoint(chat_message: ChatMessage):
    """HTTP endpoint to process a message without WebSocket."""
    if not workflow_available or WORKFLOW is None:
        return JSONResponse({
            "error": "Workflow not available",
            "success": False,
            "timestamp": datetime.now().isoformat()
        }, status_code=503)

    try:
        result = await WORKFLOW.process_query(
            user_query=chat_message.message,
            session_id=chat_message.session_id,
            user_context=chat_message.context or {}
        )
        return {
            "response": result.get("response", ""),
            "metadata": result.get("metadata", {}),
            "processing_history": result.get("processing_history", []),
            "timestamp": datetime.now().isoformat(),
            "session_id": chat_message.session_id,
            "success": result.get("success", True)
        }
    except Exception as e:
        logger.exception("/chat failed")
        return JSONResponse({"error": str(e), "success": False}, status_code=500)


@app.post("/debug-route")
async def debug_route(req: DebugRouteRequest):
    """Return only the LLM routing decision without calling downstream agents."""
    if WORKFLOW is None:
        return JSONResponse({"error": "Workflow not initialized"}, status_code=503)
    try:
        router = WORKFLOW.router
        result = await router.route_query(req.message, req.context or {})
        return {
            "agent": result.agent_type.value,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "extracted_info": result.extracted_info,
            "fallback_agents": [a.value for a in result.fallback_agents]
        }
    except Exception as e:
        logger.exception("/debug-route failed")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/e2e-smoke")
async def e2e_smoke():
    """Run end-to-end smoke tests for all three agents via manage_agent routing.
    Returns selected agent, confidence, and success for each test case.
    """
    if not workflow_available or WORKFLOW is None:
        return JSONResponse({"error": "Workflow not available"}, status_code=503)

    tests = [
        {
            "name": "sequence_to_properties",
            "query": "What is the property of the AMP with LPLLAGLAANFLPKIFCKITRK sequence",
            "session_id": "smoke-seq"
        },
        {
            "name": "properties_to_sequence",
            "query": "Find peptides active against S. aureus with MIC < 10 µM",
            "session_id": "smoke-research"
        },
        {
            "name": "data_analysis",
            "query": "Analyze correlation between peptide charge and MIC values",
            "session_id": "smoke-analysis"
        }
    ]

    results = []
    for t in tests:
        try:
            res = await WORKFLOW.process_query(user_query=t["query"], session_id=t["session_id"], user_context={})
            metadata = res.get("metadata", {})
            history = res.get("processing_history", [])
            # Find routing step details if present
            routing = next((h for h in history if h.get("step") == "routing"), None)
            results.append({
                "name": t["name"],
                "query": t["query"],
                "success": res.get("success", False),
                "agent_used": metadata.get("agent_used"),
                "routing_confidence": metadata.get("routing_confidence"),
                "routing_reasoning": routing.get("reasoning") if routing else None
            })
        except Exception as e:
            logger.exception("Smoke test failed")
            results.append({
                "name": t["name"],
                "query": t["query"],
                "success": False,
                "error": str(e)
            })

    return {"results": results, "timestamp": datetime.now().isoformat()}



async def stream_content_intelligently(ws: WebSocket, content: str):
    """
    Stream content with intelligent chunking for better user experience.
    Breaks content at natural boundaries (sentences, paragraphs, etc.)
    """
    if not content:
        return

    # Split content into logical chunks
    chunks = []

    # First, split by double newlines (paragraphs)
    paragraphs = content.split('\n\n')

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # If paragraph is short enough, send as one chunk
        if len(paragraph) <= 200:
            chunks.append(paragraph + '\n\n')
        else:
            # Split long paragraphs by sentences
            sentences = []
            current_sentence = ""

            # Simple sentence splitting (improved)
            words = paragraph.split()
            for word in words:
                current_sentence += word + " "

                # Check for sentence endings
                if (word.endswith('.') or word.endswith('!') or word.endswith('?') or
                    word.endswith(':') or word.endswith(';')):
                    sentences.append(current_sentence.strip())
                    current_sentence = ""

            # Add remaining text as a sentence
            if current_sentence.strip():
                sentences.append(current_sentence.strip())

            # Group sentences into chunks of reasonable size
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk + sentence) <= 300:
                    current_chunk += sentence + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip() + '\n\n')
                    current_chunk = sentence + " "

            if current_chunk:
                chunks.append(current_chunk.strip() + '\n\n')

    # Stream chunks with appropriate delays
    for i, chunk in enumerate(chunks):
        await ws.send_text(json.dumps({
            "type": "stream_chunk",
            "content": chunk,
            "chunk_index": i,
            "total_chunks": len(chunks)
        }))

        # Variable delay based on chunk length and content type
        if '**' in chunk or '*' in chunk:  # Formatted content
            delay = 0.01
        elif len(chunk) > 100:  # Long chunks
            delay = 0.015
        else:  # Short chunks
            delay = 0.005

        await asyncio.sleep(delay)


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(ws: WebSocket, session_id: str):
    await ws.accept()
    SESSIONS.setdefault(session_id, {"created": datetime.now().isoformat()})
    try:
        while True:
            data = await ws.receive_text()
            try:
                payload = json.loads(data)
                message = payload.get("message", "").strip()
            except Exception:
                message = (data or "").strip()

            if not message:
                await ws.send_text(json.dumps({"type": "error", "content": "Empty message"}))
                continue

            # Immediately show typing indicator then start stream
            await ws.send_text(json.dumps({"type": "typing"}))
            await ws.send_text(json.dumps({"type": "stream_start"}))

            # Process query with streaming
            try:
                # Use the graph streaming method
                async for chunk in WORKFLOW.process_query_stream(user_query=message, session_id=session_id, user_context={}):
                    if chunk.get("type") == "stream_chunk":
                        await ws.send_text(json.dumps({
                            "type": "stream_chunk",
                            "content": chunk.get("content", "")
                        }))

            except Exception as e:
                logger.exception("WebSocket streaming error")
                await ws.send_text(json.dumps({
                    "type": "stream_chunk",
                    "content": f"❌ Sorry, an error occurred: {e}"
                }))

            await ws.send_text(json.dumps({"type": "stream_end", "timestamp": datetime.now().isoformat()}))
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception:
        logger.exception("Unexpected WebSocket error")
    finally:
        # Keep session state for history if needed; do not delete here
        pass

if __name__ == "__main__":
    import uvicorn

    # Write to log file for debugging
    with open("server_startup.log", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("AMPilot Manage Agent Server Starting...\n")
        f.write("=" * 60 + "\n")
        f.write(f"Workflow available: {workflow_available}\n")
        f.write(f"WORKFLOW instance: {WORKFLOW is not None}\n")
        f.flush()

    try:
        host = os.getenv("MANAGE_AGENT_HOST", "127.0.0.1")
        port = int(os.getenv("MANAGE_AGENT_PORT", "8002"))

        with open("server_startup.log", "a") as f:
            f.write(f"Starting server on http://{host}:{port}\n")
            f.write("Available endpoints:\n")
            f.write(f" - Main:           http://{host}:{port}/\n")
            f.write(f" - Health:         http://{host}:{port}/health\n")
            f.write(f" - Chat:           http://{host}:{port}/chat\n")
            f.write(f" - Test Sequence:  http://{host}:{port}/test-sequence\n")
            f.write(f" - Test Properties: http://{host}:{port}/test-properties\n")
            f.write(f" - Test Greeting:  http://{host}:{port}/test-greeting\n")
            f.write("=" * 60 + "\n")
            f.flush()

        logger.info(f"Starting AMPilot Manage Agent on http://{host}:{port}")
        uvicorn.run(app, host=host, port=port, reload=False, log_level="info")

    except Exception as e:
        with open("server_startup.log", "a") as f:
            f.write(f"ERROR: Failed to start server: {e}\n")
            f.flush()
        logger.error(f"Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        raise

