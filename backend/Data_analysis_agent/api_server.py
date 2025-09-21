"""
Data Analysis Agent API Server

FastAPI-based data analysis agent API server
"""

import os
import logging
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import tempfile
from pathlib import Path
from datetime import datetime
# Ensure local package imports work when run via different entry points
import sys
from pathlib import Path as _Path
_pkg_dir = _Path(__file__).resolve().parent
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))


try:
    from .configuration import API_HOST, API_PORT, DATA_DIR, SUPPORTED_FORMATS, MAX_FILE_SIZE_MB, RESULTS_DIR
    from .graph.analysis_graph import DataAnalysisGraph
except Exception:
    from configuration import API_HOST, API_PORT, DATA_DIR, SUPPORTED_FORMATS, MAX_FILE_SIZE_MB, RESULTS_DIR
    from graph.analysis_graph import DataAnalysisGraph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Data Analysis Agent API",
    description="LangGraph-based wet lab data analysis agent API server",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
analysis_graph = None

# Request models
class AnalysisRequest(BaseModel):
    user_request: str
    data_description: Optional[str] = None  # Natural language data description
    data_file_path: Optional[str] = None

class AnalysisResponse(BaseModel):
    success: bool
    error: Optional[str] = None
    task_understanding: str = ""
    data_understanding: str = ""
    analysis_plan: str = ""
    generated_code: str = ""
    execution_result: Dict[str, Any] = {}
    final_analysis: str = ""
    recommendations: List[str] = []
    plots: List[Dict[str, Any]] = []
    current_step: str = ""
    pdf_report_path: Optional[str] = None
    pdf_generated: bool = False
    pdf_error: Optional[str] = None

class ChatMessage(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    response: str
    metadata: Optional[Dict[str, Any]] = None

@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    global analysis_graph

    logger.info("Starting Data Analysis Agent API server...")

    try:
        # Initialize analysis graph
        analysis_graph = DataAnalysisGraph()
        logger.info("Data analysis graph initialized successfully")

        # Ensure data directory exists
        Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

        logger.info("Data Analysis Agent API server startup completed")

    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        raise e

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Data Analysis Agent API Server",
        "version": "1.0.0",
        "status": "running",
        "features": [
            "Wet lab data analysis",
            "LangGraph-based intelligent workflow",
            "GPT-4o-mini powered analysis",
            "Multiple statistical analysis methods",
            "Automatic code generation and execution",
            "Professional result interpretation"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "graph_initialized": analysis_graph is not None
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_message: ChatMessage):
    """
    Main chat endpoint for Data Analysis Agent.
    Routes natural language queries to appropriate analysis functions.
    """
    try:
        message = chat_message.message.lower()

        # For now, use natural language analysis since we don't have uploaded data
        request = AnalysisRequest(
            user_request=chat_message.message,
            data_description="Antimicrobial peptide research data for analysis"
        )

        result = await analyze_natural_language(request)

        # Format response for chat interface
        response_text = ""
        if result.task_understanding:
            response_text += f"**Task Understanding:** {result.task_understanding}\n\n"
        if result.data_understanding:
            response_text += f"**Data Understanding:** {result.data_understanding}\n\n"
        if result.analysis_plan:
            response_text += f"**Analysis Plan:** {result.analysis_plan}\n\n"
        if result.final_analysis:
            response_text += f"**Analysis Results:** {result.final_analysis}\n\n"
        if result.recommendations:
            response_text += f"**Recommendations:**\n"
            for i, rec in enumerate(result.recommendations, 1):
                response_text += f"{i}. {rec}\n"

        if not response_text:
            response_text = """I'm the Data Analysis Agent. I can help you analyze antimicrobial peptide data.

**My capabilities:**
- **Statistical analysis** - Descriptive statistics, correlations, distributions
- **Comparative analysis** - Compare different peptide groups or conditions
- **Predictive modeling** - Build models to predict activity or properties
- **Data visualization** - Generate plots and charts

**Example queries:**
- "Analyze the correlation between charge and MIC values"
- "Compare activity data between different bacterial targets"
- "Generate statistical summary of peptide properties"

What data analysis would you like me to perform?"""

        return ChatResponse(
            response=response_text,
            metadata={
                "agent": "data_analysis",
                "analysis_success": result.success,
                "session_id": chat_message.session_id,
                "plots_generated": len(result.plots) if result.plots else 0
            }
        )

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return ChatResponse(
            response=f"Sorry, I encountered an error processing your request: {str(e)}",
            metadata={
                "agent": "data_analysis",
                "error": str(e),
                "session_id": chat_message.session_id
            }
        )

@app.post("/upload", response_model=Dict[str, Any])
async def upload_file(file: UploadFile = File(...)):
    """Upload data file"""
    try:
        # Check file format
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file_ext}. Supported formats: {SUPPORTED_FORMATS}"
            )

        # Check file size
        file_size_mb = len(await file.read()) / (1024 * 1024)
        await file.seek(0)  # Reset file pointer

        if file_size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"File too large: {file_size_mb:.2f}MB > {MAX_FILE_SIZE_MB}MB"
            )

        # Save file
        file_path = Path(DATA_DIR) / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        logger.info(f"File uploaded successfully: {file.filename}")

        return {
            "success": True,
            "filename": file.filename,
            "file_path": str(file_path),
            "size_mb": file_size_mb,
            "format": file_ext
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_data(request: AnalysisRequest):
    """Execute data analysis with natural language input"""
    try:
        if not analysis_graph:
            raise HTTPException(status_code=500, detail="Analysis graph not initialized")

        # Support both file-based and natural language data input
        if not request.data_file_path and not request.data_description:
            raise HTTPException(status_code=400, detail="Either data_file_path or data_description must be provided")

        logger.info(f"Starting analysis: {request.user_request}")

        # Execute analysis
        if request.data_file_path:
            # File-based analysis
            file_path = Path(request.data_file_path)
            if not file_path.exists():
                raise HTTPException(status_code=404, detail=f"File not found: {request.data_file_path}")

            result = analysis_graph.run_analysis(
                user_request=request.user_request,
                data_file=str(file_path)
            )
        else:
            # Natural language data analysis
            result = analysis_graph.run_analysis_with_description(
                user_request=request.user_request,
                data_description=request.data_description
            )

        logger.info(f"Analysis completed: {result['current_step']}")

        return AnalysisResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Data analysis failed: {str(e)}")

@app.post("/analyze_with_upload", response_model=AnalysisResponse)
async def analyze_with_upload(
    user_request: str = Form(...),
    file: UploadFile = File(...)
):
    """Upload file and execute analysis"""
    try:
        # Upload file first
        upload_result = await upload_file(file)

        if not upload_result["success"]:
            raise HTTPException(status_code=400, detail="File upload failed")

        # Execute analysis
        analysis_request = AnalysisRequest(
            user_request=user_request,
            data_file_path=upload_result["file_path"]
        )

        return await analyze_data(analysis_request)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload and analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload and analysis failed: {str(e)}")

@app.post("/analyze_natural", response_model=AnalysisResponse)
async def analyze_natural_language(request: AnalysisRequest):
    """Analyze data using natural language description only"""
    try:
        if not analysis_graph:
            raise HTTPException(status_code=500, detail="Analysis graph not initialized")

        if not request.data_description:
            raise HTTPException(status_code=400, detail="data_description is required for natural language analysis")

        logger.info(f"Starting natural language analysis: {request.user_request}")

        # Execute analysis with natural language description
        result = analysis_graph.run_analysis_with_description(
            user_request=request.user_request,
            data_description=request.data_description
        )

        logger.info(f"Natural language analysis completed: {result['current_step']}")

        return AnalysisResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Natural language analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Natural language analysis failed: {str(e)}")

@app.get("/files")
async def list_files():
    """List uploaded files"""
    try:
        data_dir = Path(DATA_DIR)
        files = []

        for file_path in data_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FORMATS:
                stat = file_path.stat()
                files.append({
                    "filename": file_path.name,
                    "path": str(file_path),
                    "size_mb": stat.st_size / (1024 * 1024),
                    "modified": stat.st_mtime,
                    "format": file_path.suffix.lower()
                })

        return {
            "files": files,
            "count": len(files)
        }

    except Exception as e:
        logger.error(f"Failed to get file list: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get file list: {str(e)}")

@app.get("/supported_formats")
async def get_supported_formats():
    """Get supported file formats"""
    return {
        "formats": SUPPORTED_FORMATS,
        "max_size_mb": MAX_FILE_SIZE_MB,
        "description": {
            ".csv": "Comma-separated values file",
            ".xlsx": "Excel workbook",
            ".xls": "Excel workbook (legacy)",
            ".json": "JSON data file",
            ".txt": "Text data file"
        }
    }

@app.get("/download_report/{filename}")
async def download_report(filename: str):
    """Download generated PDF report"""
    try:
        # Validate filename to prevent directory traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        # Check if file exists in results directory
        file_path = Path(RESULTS_DIR) / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Report not found")

        # Return file
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type="application/pdf"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download report: {str(e)}")

@app.get("/list_reports")
async def list_reports():
    """List all available PDF reports"""
    try:
        results_dir = Path(RESULTS_DIR)
        if not results_dir.exists():
            return {"reports": []}

        # Find all PDF files
        pdf_files = []
        for pdf_file in results_dir.glob("*.pdf"):
            stat = pdf_file.stat()
            pdf_files.append({
                "filename": pdf_file.name,
                "size_bytes": stat.st_size,
                "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })

        # Sort by creation time (newest first)
        pdf_files.sort(key=lambda x: x["created_time"], reverse=True)

        return {"reports": pdf_files}

    except Exception as e:
        logger.error(f"Failed to list reports: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list reports: {str(e)}")

if __name__ == "__main__":
    logger.info(f"Starting Data Analysis Agent API server at {API_HOST}:{API_PORT}")
    uvicorn.run(
        "api_server:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info"
    )
