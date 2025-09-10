"""
FastAPI server for AMPilot - Antimicrobial Peptide Research Assistant
Provides REST API endpoints for the agentic RAG system.
"""

import uuid
import json
import asyncio
import re
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import os

try:
    from .retrieval_graph.graph import app as rag_app  # when imported as package
except Exception:
    from retrieval_graph.graph import app as rag_app     # when run as a script

def _is_router_output(content: str) -> bool:
    """Check if content is router output (contains intent/confidence JSON)."""
    if not content:
        return False

    # Check for router JSON patterns
    router_indicators = [
        '"intent":', '"confidence":', '"reasoning":', '"extracted_info":',
        'sequence_to_properties', 'properties_to_sequence', 'general_conversation'
    ]

    s = content.strip()
    content_lower = s.lower()

    # Pure JSON router object
    if s.startswith('{') and s.endswith('}') and any(indicator in content_lower for indicator in router_indicators):
        return True

    # JSON prefix followed by text (the case user saw)
    if s.startswith('{') and any(indicator in content_lower for indicator in router_indicators) and '}' in s:
        # treat as router output present (will be stripped by cleaner)
        return True

    return False

def _clean_assistant_content(content: str) -> str:
    """Remove any leading router JSON blob and return the human-friendly text."""
    if not content:
        return content
    s = content.strip()
    if s.startswith('{') and '}' in s and '"intent"' in s:
        # Remove up to first closing brace
        idx = s.find('}')
        rest = s[idx+1:].lstrip()
        return rest
    return s

# FastAPI app initialization
app = FastAPI(
    title="AMPilot API",
    description="Antimicrobial Peptide Research Assistant API",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend (repo root /frontend)
from pathlib import Path
_repo_root = Path(__file__).resolve().parents[2]
frontend_path = str((_repo_root / "frontend").resolve())
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# Pydantic models for request/response
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str
    status: str

class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    message_count: int

# In-memory session storage (in production, use Redis or database)
sessions: Dict[str, Dict[str, Any]] = {}

def build_history_messages(session_id: str, new_user_message: Optional[str] = None) -> List[tuple]:
    """Build message history for LangGraph from stored session messages.
    Converts stored roles to tuples ("human", text) and ("ai", text)."""
    history: List[tuple] = []
    try:
        session = sessions.get(session_id, {})
        for m in session.get("messages", []):
            role = m.get("role")
            content = m.get("content", "")
            if not content:
                continue
            if role == "user":
                history.append(("human", content))
            elif role == "assistant":
                history.append(("ai", content))
    except Exception:
        pass
    if new_user_message:
        history.append(("human", new_user_message))
    return history

# Streaming helper function
async def stream_html_content(content: str, websocket: WebSocket, manager, session_id: str):
    """Stream content as multiple small chunks while keeping a single message bubble on the client."""
    text = content if isinstance(content, str) else str(content)

    # Simple tokenizer: split by paragraphs first, then by ~120-char pieces
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    pieces: List[str] = []
    for p in paragraphs if paragraphs else [text]:
        s = p.strip()
        while len(s) > 0:
            pieces.append(s[:160])
            s = s[160:]

    total = max(1, len(pieces))
    for i, piece in enumerate(pieces or [text]):
        await manager.send_personal_message(
            json.dumps({
                "type": "stream_chunk",
                "content": piece,
                "session_id": session_id,
                "chunk_index": i,
                "total_chunks": total
            }),
            websocket
        )
        # tiny pacing for UX (kept minimal)
        await asyncio.sleep(0.03)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

@app.get("/")
async def root():
    """Serve the main frontend page."""
    frontend_file = os.path.join(frontend_path, "index.html")
    if os.path.exists(frontend_file):
        return FileResponse(frontend_file)
    else:
        return {
            "message": "Welcome to AMPilot API",
            "description": "Antimicrobial Peptide Research Assistant",
            "version": "1.0.0",
            "endpoints": {
                "chat": "/chat",
                "websocket": "/ws/{session_id}",
                "sessions": "/sessions",
                "health": "/health"
            }
        }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_message: ChatMessage):
    """
    Main chat endpoint for interacting with the AMPilot agent.
    """
    try:
        # Generate session ID if not provided
        session_id = chat_message.session_id or str(uuid.uuid4())

        # Initialize session if new
        if session_id not in sessions:
            sessions[session_id] = {
                "created_at": datetime.now().isoformat(),
                "message_count": 0,
                "messages": []
            }

        # Update session
        sessions[session_id]["message_count"] += 1
        sessions[session_id]["messages"].append({
            "role": "user",
            "content": chat_message.message,
            "timestamp": datetime.now().isoformat()
        })

        # Configure for LangGraph
        config = {"configurable": {"thread_id": session_id}}
        # Build history-aware messages (context memory)
        inputs = {"messages": build_history_messages(session_id, chat_message.message)}

        # Get response from RAG agent
        response_content = ""
        try:
            # Stream the response from the agent
            for chunk in rag_app.stream(inputs, config=config, stream_mode="values"):
                # Only accept the finalizer step to avoid duplicate intermediate outputs
                step = chunk.get("current_step")
                if step != "finalize":
                    continue

                final_message = chunk["messages"][-1]

                # Handle different message types
                if isinstance(final_message, tuple):
                    # Skip tuple messages (these are input messages)
                    continue
                elif getattr(final_message, 'type', None) == 'ai' and hasattr(final_message, 'content') and final_message.content:
                    content = _clean_assistant_content(final_message.content.strip())

                    # Filter out router output (JSON-like content with intent/confidence)
                    if content and not _is_router_output(content):
                        response_content = content
                elif isinstance(final_message, str):
                    # If it's a string, also check if it's router output
                    content = _clean_assistant_content(final_message)
                    if content and not _is_router_output(content):
                        response_content = content

        except Exception as e:
            import traceback
            print(f"Error from RAG agent: {e}")
            print("Full traceback:")
            traceback.print_exc()
            response_content = "I apologize, but I encountered an error while processing your request. Please try again or rephrase your question."


        # Store assistant response
        sessions[session_id]["messages"].append({
            "role": "assistant",
            "content": response_content,
            "timestamp": datetime.now().isoformat()
        })

        return ChatResponse(
            response=response_content,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            status="success"
        )

    except Exception as e:
        print(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/sessions")
async def get_sessions():
    """Get all active sessions."""
    session_list = []
    for session_id, session_data in sessions.items():
        session_list.append(SessionInfo(
            session_id=session_id,
            created_at=session_data["created_at"],
            message_count=session_data["message_count"]
        ))
    return {"sessions": session_list}

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get specific session details."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session_id,
        "session_data": sessions[session_id]
    }

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    del sessions[session_id]
    return {"message": f"Session {session_id} deleted successfully"}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time chat.
    """
    await manager.connect(websocket)

    # Initialize session if new
    if session_id not in sessions:
        sessions[session_id] = {
            "created_at": datetime.now().isoformat(),
            "message_count": 0,
            "messages": []
        }

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")

            print(f"🔍 Received WebSocket message: {user_message[:100]}...")
            print(f"📊 Session ID: {session_id}")

            if not user_message:
                continue

            # Update session
            sessions[session_id]["message_count"] += 1
            sessions[session_id]["messages"].append({
                "role": "user",
                "content": user_message,
                "timestamp": datetime.now().isoformat()
            })

            # Configure for LangGraph
            config = {"configurable": {"thread_id": session_id}}
            # Build history-aware messages (context memory)
            inputs = {"messages": build_history_messages(session_id, user_message)}

            # Send typing indicator
            await manager.send_personal_message(
                json.dumps({"type": "typing", "status": "thinking"}),
                websocket
            )

            # Get response from RAG agent with streaming
            response_content = ""
            streaming_started = False

            try:
                for chunk in rag_app.stream(inputs, config=config, stream_mode="values"):
                    # Only accept the finalizer step to avoid duplicate intermediate outputs
                    step = chunk.get("current_step")
                    if step != "finalize":
                        continue

                    final_message = chunk["messages"][-1]

                    # Handle different message types
                    if isinstance(final_message, tuple):
                        # Skip tuple messages (these are input messages)
                        continue
                    elif getattr(final_message, 'type', None) == 'ai' and hasattr(final_message, 'content') and final_message.content:
                        # Only stream assistant (ai) messages; skip tool/system/router outputs
                        content = final_message.content.strip()
                        if content:
                            # Clean and filter router JSON
                            clean_content = _clean_assistant_content(content)
                            if clean_content and not _is_router_output(clean_content):
                                # Start streaming if not already started
                                if not streaming_started:
                                    await manager.send_personal_message(
                                        json.dumps({"type": "stream_start"}),
                                        websocket
                                    )
                                    streaming_started = True

                                # Stream final content in chunks (finalize step only)
                                await stream_html_content(clean_content, websocket, manager, session_id)
                                response_content = clean_content

                    elif isinstance(final_message, str):
                        # If it's a string
                        response_content = final_message

            except Exception as e:
                import traceback
                print(f"❌ WebSocket RAG error: {e}")
                print("📍 Full traceback:")
                traceback.print_exc()
                print(f"📝 Input was: {inputs}")
                print(f"🔧 Config was: {config}")
                response_content = f"I apologize, but I encountered an error while processing your request: {str(e)[:100]}. Please try again."

            # End streaming
            if streaming_started:
                await manager.send_personal_message(
                    json.dumps({
                        "type": "stream_end",
                        "session_id": session_id,
                        "timestamp": datetime.now().isoformat()
                    }),
                    websocket
                )

            # Store assistant response
            sessions[session_id]["messages"].append({
                "role": "assistant",
                "content": response_content,
                "timestamp": datetime.now().isoformat()
            })

            # Only send final response if no streaming occurred
            if not streaming_started:
                await manager.send_personal_message(
                    json.dumps({
                        "type": "final_response",
                        "content": response_content,
                        "session_id": session_id,
                        "timestamp": datetime.now().isoformat()
                    }),
                    websocket
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print(f"WebSocket disconnected for session: {session_id}")

if __name__ == "__main__":
    import uvicorn
    print(" Starting AMPilot API Server...")
    print(" Dashboard: http://localhost:8001")
    print(" WebSocket: ws://localhost:8001/ws")
    print("📖 API Docs: http://localhost:8001/docs")
    print("ℹ️  Note: Using port 8001")

    uvicorn.run(app, host="127.0.0.1", port=8001, reload=False)
