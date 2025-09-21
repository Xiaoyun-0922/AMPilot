"""
Data Analysis Agent LangGraph Implementation

This module implements the data analysis workflow using LangGraph,
providing a structured approach to understanding tasks, analyzing data,
generating code, and interpreting results.
"""

import logging
import json
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

import sys
sys.path.append(str(Path(__file__).parent.parent))

from configuration import OPENAI_API_KEY, CHAT_MODEL
from graph.state import AnalysisState, DataContext, AnalysisResult
from tools.data_loader import DataLoader
from tools.code_executor import CodeExecutor
from tools.pdf_report_generator import PDFReportGenerator
from configuration import RESULTS_DIR
from prompts import (
    TASK_UNDERSTANDING_SYSTEM_PROMPT,
    DATA_UNDERSTANDING_SYSTEM_PROMPT,
    ANALYSIS_PLANNING_SYSTEM_PROMPT,
    CODE_GENERATION_SYSTEM_PROMPT,
    RESULT_INTERPRETATION_SYSTEM_PROMPT,
    TASK_UNDERSTANDING_PROMPT_TEMPLATE,
    DATA_UNDERSTANDING_PROMPT_TEMPLATE,
    ANALYSIS_PLANNING_PROMPT_TEMPLATE,
    CODE_GENERATION_PROMPT_TEMPLATE,
    RESULT_INTERPRETATION_PROMPT_TEMPLATE,
    DEFAULT_TASK_UNDERSTANDING,
    DEFAULT_DATA_UNDERSTANDING,
    DEFAULT_ANALYSIS_PLAN
)

logger = logging.getLogger(__name__)

class DataAnalysisGraph:
    """LangGraph-based workflow for data analysis"""
    
    def __init__(self):
        """Initialize the analysis graph with required components"""
        self.llm = ChatOpenAI(
            model=CHAT_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0.1
        )
        self.data_loader = DataLoader()
        self.code_executor = CodeExecutor()
        self.pdf_generator = PDFReportGenerator(RESULTS_DIR)
        self.graph = self._build_graph()
        
        logger.info("DataAnalysisGraph initialized successfully")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AnalysisState)
        
        # Add nodes
        workflow.add_node("understand_task", self._understand_task)
        workflow.add_node("understand_data", self._understand_data)
        workflow.add_node("plan_analysis", self._plan_analysis)
        workflow.add_node("generate_code", self._generate_code)
        workflow.add_node("execute_code", self._execute_code)
        workflow.add_node("interpret_results", self._interpret_results)
        workflow.add_node("generate_pdf_report", self._generate_pdf_report)
        workflow.add_node("handle_error", self._handle_error)
        
        # Define the flow
        workflow.set_entry_point("understand_task")
        
        # Sequential flow with error handling
        workflow.add_conditional_edges(
            "understand_task",
            self._check_task_understanding,
            {
                "success": "understand_data",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "understand_data",
            self._check_data_understanding,
            {
                "success": "plan_analysis",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "plan_analysis",
            self._check_analysis_plan,
            {
                "success": "generate_code",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "generate_code",
            self._check_code_generation,
            {
                "success": "execute_code",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "execute_code",
            self._check_code_execution,
            {
                "success": "interpret_results",
                "error": "handle_error"
            }
        )

        workflow.add_edge("interpret_results", "generate_pdf_report")
        workflow.add_edge("generate_pdf_report", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    def _understand_task(self, state: AnalysisState) -> AnalysisState:
        """Understand the user's analysis request"""
        try:
            state["current_step"] = "understanding_task"
            logger.info("Understanding analysis task...")
            
            # Prepare data context if available
            data_context = ""
            if state.get("data_file_path"):
                try:
                    data_info = self.data_loader.get_file_info(state["data_file_path"])
                    data_context = f"Dataset: {data_info['filename']}, Size: {data_info['size_mb']:.2f}MB"
                except Exception as e:
                    logger.warning(f"Could not load data context: {e}")
                    data_context = "Data file provided but could not be analyzed"
            
            # Format prompt
            prompt = TASK_UNDERSTANDING_PROMPT_TEMPLATE.format(
                user_request=state["user_request"],
                data_context=data_context
            )
            
            # Get LLM response
            messages = [
                SystemMessage(content=TASK_UNDERSTANDING_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse response (simplified - in production, you might want JSON parsing)
            state["task_understanding"] = {
                "raw_response": response.content,
                "analysis_type": "comprehensive",
                "status": "completed"
            }
            
            logger.info("Task understanding completed")
            return state
            
        except Exception as e:
            logger.error(f"Task understanding error: {e}")
            state["error_message"] = f"Failed to understand task: {str(e)}"
            state["success"] = False
            return state
    
    def _understand_data(self, state: AnalysisState) -> AnalysisState:
        """Analyze and understand the dataset"""
        try:
            state["current_step"] = "understanding_data"
            logger.info("Understanding dataset...")

            # Handle both file-based and description-based data
            data_context = None

            if state.get("data_file_path"):
                # File-based data understanding
                data_context = self.data_loader.load_data_context(state["data_file_path"])

                prompt = DATA_UNDERSTANDING_PROMPT_TEMPLATE.format(
                    data_info=f"Shape: {data_context.shape}, Columns: {len(data_context.columns)}",
                    data_sample=str(data_context.sample_data)
                )
            elif state.get("data_description"):
                # Natural language data description
                prompt = f"""
                Data Description (provided by user):
                {state['data_description']}

                Please analyze this data description and provide insights about:
                1. Data structure and variables
                2. Expected data types and formats
                3. Potential data quality considerations
                4. Suitable analysis approaches
                5. Any limitations or assumptions
                """
            else:
                state["error_message"] = "No data file or data description provided"
                state["success"] = False
                return state

            # Get LLM response
            messages = [
                SystemMessage(content=DATA_UNDERSTANDING_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]

            response = self.llm.invoke(messages)

            state["data_understanding"] = {
                "raw_response": response.content,
                "data_context": data_context.__dict__ if data_context else None,
                "status": "completed"
            }
            
            logger.info("Data understanding completed")
            return state
            
        except Exception as e:
            logger.error(f"Data understanding error: {e}")
            state["error_message"] = f"Failed to understand data: {str(e)}"
            state["success"] = False
            return state
    
    def _plan_analysis(self, state: AnalysisState) -> AnalysisState:
        """Create detailed analysis plan"""
        try:
            state["current_step"] = "planning_analysis"
            logger.info("Planning analysis...")
            
            # Format prompt
            prompt = ANALYSIS_PLANNING_PROMPT_TEMPLATE.format(
                task_understanding=state["task_understanding"]["raw_response"],
                data_understanding=state["data_understanding"]["raw_response"]
            )
            
            # Get LLM response
            messages = [
                SystemMessage(content=ANALYSIS_PLANNING_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            state["analysis_plan"] = {
                "raw_response": response.content,
                "status": "completed"
            }
            
            logger.info("Analysis planning completed")
            return state
            
        except Exception as e:
            logger.error(f"Analysis planning error: {e}")
            state["error_message"] = f"Failed to plan analysis: {str(e)}"
            state["success"] = False
            return state
    
    def _generate_code(self, state: AnalysisState) -> AnalysisState:
        """Generate Python code for analysis"""
        try:
            state["current_step"] = "generating_code"
            logger.info("Generating analysis code...")
            
            # Format prompt based on data source
            if state.get("data_file_path"):
                data_context = f"File: {state['data_file_path']}"
            else:
                data_context = f"Data Description: {state.get('data_description', 'No data description provided')}"

            prompt = CODE_GENERATION_PROMPT_TEMPLATE.format(
                analysis_plan=state["analysis_plan"]["raw_response"],
                data_context=data_context
            )
            
            # Get LLM response
            messages = [
                SystemMessage(content=CODE_GENERATION_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Extract code from response (simplified)
            code_content = response.content
            if "```python" in code_content:
                code_start = code_content.find("```python") + 9
                code_end = code_content.find("```", code_start)
                if code_end != -1:
                    code_content = code_content[code_start:code_end].strip()
            
            state["generated_code"] = code_content
            
            logger.info("Code generation completed")
            return state

        except Exception as e:
            logger.error(f"Code generation error: {e}")
            state["error_message"] = f"Failed to generate code: {str(e)}"
            state["success"] = False
            return state

    def _execute_code(self, state: AnalysisState) -> AnalysisState:
        """Execute the generated analysis code"""
        try:
            state["current_step"] = "executing_code"
            logger.info("Executing analysis code...")

            if not state.get("generated_code"):
                state["error_message"] = "No code generated to execute"
                state["success"] = False
                return state

            # Prepare data context for execution
            data_context = {}
            if state.get("data_file_path"):
                data_context["data_file_path"] = state["data_file_path"]
            elif state.get("data_description"):
                data_context["data_description"] = state["data_description"]

            # Execute code
            execution_result = self.code_executor.execute_code(
                state["generated_code"],
                data_context
            )

            state["execution_result"] = execution_result

            if execution_result.get("success"):
                logger.info("Code execution completed successfully")
            else:
                logger.warning(f"Code execution had issues: {execution_result.get('error')}")

            return state

        except Exception as e:
            logger.error(f"Code execution error: {e}")
            state["error_message"] = f"Failed to execute code: {str(e)}"
            state["success"] = False
            return state

    def _interpret_results(self, state: AnalysisState) -> AnalysisState:
        """Interpret analysis results and provide insights"""
        try:
            state["current_step"] = "interpreting_results"
            logger.info("Interpreting analysis results...")

            # Format results for interpretation
            execution_result = state.get("execution_result", {})
            results_summary = {
                "stdout": execution_result.get("stdout", ""),
                "plots": execution_result.get("plots", []),
                "variables": execution_result.get("variables", {}),
                "success": execution_result.get("success", False)
            }

            # Format prompt
            prompt = RESULT_INTERPRETATION_PROMPT_TEMPLATE.format(
                analysis_results=json.dumps(results_summary, indent=2),
                original_request=state["user_request"]
            )

            # Get LLM response
            messages = [
                SystemMessage(content=RESULT_INTERPRETATION_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]

            response = self.llm.invoke(messages)

            # Set final outputs
            state["final_analysis"] = response.content
            state["recommendations"] = ["Review the analysis results", "Consider additional validation"]
            state["plots"] = execution_result.get("plots", [])
            state["success"] = True
            state["current_step"] = "completed"

            logger.info("Result interpretation completed")
            return state

        except Exception as e:
            logger.error(f"Result interpretation error: {e}")
            state["error_message"] = f"Failed to interpret results: {str(e)}"
            state["success"] = False
            return state

    def _handle_error(self, state: AnalysisState) -> AnalysisState:
        """Handle errors in the workflow"""
        try:
            state["current_step"] = "error_handling"
            error_msg = state.get("error_message", "Unknown error occurred")

            logger.error(f"Handling workflow error: {error_msg}")

            # Provide fallback analysis
            state["final_analysis"] = f"Analysis failed: {error_msg}. Please check your data and request."
            state["recommendations"] = [
                "Verify data file format and accessibility",
                "Simplify the analysis request",
                "Check for data quality issues"
            ]
            state["plots"] = []
            state["success"] = False

            return state

        except Exception as e:
            logger.error(f"Error handling failed: {e}")
            state["error_message"] = f"Critical error: {str(e)}"
            state["success"] = False
            return state

    # Conditional edge functions
    def _check_task_understanding(self, state: AnalysisState) -> str:
        """Check if task understanding was successful"""
        if state.get("task_understanding") and state["task_understanding"].get("status") == "completed":
            return "success"
        return "error"

    def _check_data_understanding(self, state: AnalysisState) -> str:
        """Check if data understanding was successful"""
        if state.get("data_understanding") and state["data_understanding"].get("status") == "completed":
            return "success"
        return "error"

    def _check_analysis_plan(self, state: AnalysisState) -> str:
        """Check if analysis planning was successful"""
        if state.get("analysis_plan") and state["analysis_plan"].get("status") == "completed":
            return "success"
        return "error"

    def _check_code_generation(self, state: AnalysisState) -> str:
        """Check if code generation was successful"""
        if state.get("generated_code") and len(state["generated_code"].strip()) > 0:
            return "success"
        return "error"

    def _check_code_execution(self, state: AnalysisState) -> str:
        """Check if code execution was successful"""
        if state.get("execution_result"):
            return "success"
        return "error"

    def run_analysis(self, user_request: str, data_file: str) -> Dict[str, Any]:
        """Run the complete analysis workflow with file input"""
        try:
            initial_state = AnalysisState(
                user_request=user_request,
                data_file_path=data_file,
                current_step="",
                task_understanding=None,
                data_understanding=None,
                analysis_plan=None,
                generated_code=None,
                execution_result=None,
                final_analysis=None,
                recommendations=None,
                plots=None,
                error_message=None,
                success=False
            )

            # Execute the graph
            result = self.graph.invoke(initial_state)

            # Return formatted result
            return self._format_result(result)

        except Exception as e:
            logger.error(f"Analysis workflow failed: {e}")
            return self._format_error_result(str(e))

    def run_analysis_with_description(self, user_request: str, data_description: str) -> Dict[str, Any]:
        """Run the complete analysis workflow with natural language data description"""
        try:
            initial_state = AnalysisState(
                user_request=user_request,
                data_file_path=None,  # No file, using description
                data_description=data_description,  # Add data description
                current_step="",
                task_understanding=None,
                data_understanding=None,
                analysis_plan=None,
                generated_code=None,
                execution_result=None,
                final_analysis=None,
                recommendations=None,
                plots=None,
                error_message=None,
                success=False
            )

            # Execute the graph
            result = self.graph.invoke(initial_state)

            # Return formatted result
            return self._format_result(result)

        except Exception as e:
            logger.error(f"Analysis workflow with description failed: {e}")
            return self._format_error_result(str(e))

    def _format_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format analysis result"""
        return {
            "success": result.get("success", False),
            "error": result.get("error_message"),
            "task_understanding": result.get("task_understanding", {}).get("raw_response", ""),
            "data_understanding": result.get("data_understanding", {}).get("raw_response", ""),
            "analysis_plan": result.get("analysis_plan", {}).get("raw_response", ""),
            "generated_code": result.get("generated_code", ""),
            "execution_result": result.get("execution_result", {}),
            "final_analysis": result.get("final_analysis", ""),
            "recommendations": result.get("recommendations", []),
            "plots": result.get("plots", []),
            "current_step": result.get("current_step", ""),
            "pdf_report_path": result.get("pdf_report_path", ""),
            "pdf_generated": result.get("pdf_generated", False),
            "pdf_error": result.get("pdf_error", "")
        }

    def _generate_pdf_report(self, state: AnalysisState) -> AnalysisState:
        """Generate PDF report from analysis results"""
        try:
            logger.info("Generating PDF report...")

            # Prepare analysis result for PDF generation
            analysis_result = {
                "success": state.get("success", True),
                "current_step": state.get("current_step", "completed"),
                "task_understanding": state.get("task_understanding", ""),
                "data_understanding": state.get("data_understanding", ""),
                "analysis_plan": state.get("analysis_plan", ""),
                "generated_code": state.get("generated_code", ""),
                "execution_result": state.get("execution_result", {}),
                "final_analysis": state.get("final_analysis", ""),
                "recommendations": state.get("recommendations", []),
                "plots": state.get("plots", [])
            }

            # Generate PDF report
            session_id = state.get("session_id", "unknown")
            pdf_path = self.pdf_generator.generate_report(analysis_result, session_id)

            # Update state with PDF information
            state["pdf_report_path"] = pdf_path
            state["pdf_generated"] = True
            state["current_step"] = "pdf_generated"

            logger.info(f"PDF report generated successfully: {pdf_path}")

        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}")
            state["pdf_generated"] = False
            state["pdf_error"] = str(e)
            # Don't fail the entire analysis for PDF generation issues

        return state

    def _format_error_result(self, error_msg: str) -> Dict[str, Any]:
        """Format error result"""
        return {
            "success": False,
            "error": error_msg,
            "task_understanding": "",
            "data_understanding": "",
            "analysis_plan": "",
            "generated_code": "",
            "execution_result": {},
            "final_analysis": f"Analysis failed: {error_msg}",
            "recommendations": ["Check data description and request format"],
            "plots": [],
            "current_step": "failed"
        }
