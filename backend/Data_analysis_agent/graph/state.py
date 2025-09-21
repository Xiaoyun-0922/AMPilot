"""
State definitions for Data Analysis Agent LangGraph workflow.
"""

from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import dataclass

class AnalysisState(TypedDict):
    """State for the data analysis workflow"""
    # Input
    user_request: str
    data_file_path: Optional[str]
    data_description: Optional[str]  # Natural language data description
    
    # Processing steps
    current_step: str
    task_understanding: Optional[Dict[str, Any]]
    data_understanding: Optional[Dict[str, Any]]
    analysis_plan: Optional[Dict[str, Any]]
    generated_code: Optional[str]
    execution_result: Optional[Dict[str, Any]]
    
    # Output
    final_analysis: Optional[str]
    recommendations: Optional[List[str]]
    plots: Optional[List[Dict[str, Any]]]
    
    # Error handling
    error_message: Optional[str]
    success: bool

    # PDF report generation
    pdf_report_path: Optional[str]
    pdf_generated: bool
    pdf_error: Optional[str]

    # Session information
    session_id: Optional[str]

@dataclass
class DataContext:
    """Context information about the dataset"""
    file_path: str
    file_size: int
    columns: List[str]
    shape: tuple
    dtypes: Dict[str, str]
    sample_data: Dict[str, Any]
    missing_values: Dict[str, int]
    summary_stats: Optional[Dict[str, Any]] = None

@dataclass
class AnalysisResult:
    """Result of analysis execution"""
    success: bool
    stdout: str
    stderr: str
    plots: List[Dict[str, Any]]
    variables: Dict[str, Any]
    error: Optional[str] = None
