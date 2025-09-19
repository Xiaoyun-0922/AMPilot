"""
BLAST+ 工具的 MCP (Model Context Protocol) 包装函数
用于将 BlastTool 类的功能转换为 MCP 工具，供 LLM 直接调用
"""

import os
import json
from typing import Dict, List, Union, Optional, Any
import asyncio
from pydantic import BaseModel, Field

from ampilot.tools.sequence import BlastTool

from fastmcp import FastMCP

sequence_server = FastMCP(name="sequence_server")

_blast_tool = None

def get_blast_tool():
    """
    获取 BlastTool 实例（单例模式）
    
    返回:
    - BlastTool 实例
    """
    global _blast_tool
    if _blast_tool is None:
        _blast_tool = BlastTool()
    return _blast_tool

@sequence_server.tool
async def blast_search(
    sequence: str = Field(..., description="Query sequence or path to sequence file"),
    program: str = Field("blastp", description="BLAST program type (blastp, blastn, tblastn, etc.)"),
    database: str = Field("nr", description="Target database name"),
    evalue: float = Field(1e-5, description="E-value threshold"),
    max_hits: int = Field(50, description="Maximum number of results"),
    output_format: str = Field("json", description="Output format (json, table, alignment, xml, csv)"),
    threads: int = Field(1, description="Number of threads"),
    additional_params: Dict[str, Any] = Field(None, description="Other BLAST parameters")
) -> Dict[str, Any]:
    """
    Execute BLAST sequence search
    """
    # Get BlastTool instance
    tool = get_blast_tool()
    
    # Convert output format
    outfmt = _convert_output_format(output_format)

    # Execute search
    result = await tool.blast_search(
        sequence=sequence,
        program=program,
        database=database,
        evalue=evalue,
        max_hits=max_hits,
        outfmt=outfmt,
        threads=threads,
        additional_params=additional_params
    )
    
    # 格式化结果
    return _format_blast_result(result, output_format)

@sequence_server.tool
async def create_blast_db(
    input_file: str = Field(..., description="FASTA file containing sequences"),
    output_dir: str = Field("/Users/wang-work/AMPilot/ampilot/tools/sequence/test_data", description="Directory to store BLAST databases"),
    db_type: str = Field("protein", description="Database type ('protein' or 'nucleotide')"),
    db_name: str = Field("test_db", description="Database name"),
    title: Optional[str] = Field(None, description="Database title (optional)")
) -> Dict[str, Any]:
    """
    Create BLAST database
    """
    # Get BlastTool instance
    tool = get_blast_tool()
    
    # Convert database type
    dbtype = "prot" if db_type.lower() in ["protein", "prot", "p"] else "nucl"

    # Create database
    result = await tool.create_blast_db(
        input_file=input_file,
        output_dir=output_dir,
        dbtype=dbtype,
        dbname=db_name,
        title=title
    )
    
    # Format result
    if result.get("success", False):
        return {
            "success": True,
            "database_name": result["dbname"],
            "file_count": len(result["files"]),
            "files": result["files"],
            "message": "数据库创建成功"
        }
    else:
        return {
            "success": False,
            "error": result.get("error", "未知错误"),
            "message": result.get("message", "")
        }

def _convert_output_format(format_str: str) -> int:
    """
    将用户友好的格式名称转换为 BLAST+ 的输出格式代码
    
    参数:
    - format_str: 格式名称 (json, table, alignment, xml, etc.)
    
    返回:
    - 对应的 BLAST+ 输出格式代码
    """
    format_map = {
        "json": 15,        # BLAST JSON 格式
        "table": 6,        # 表格格式
        "alignment": 0,    # 传统 pairwise 格式
        "xml": 5,          # BLAST XML 格式
        "csv": 10,         # CSV 格式
    }
    
    return format_map.get(format_str.lower(), 15)  # 默认为 JSON


def _format_blast_result(result: Dict[str, Any], output_format: str) -> Dict[str, Any]:
    """
    格式化 BLAST 结果为对用户友好的格式
    
    参数:
    - result: 原始 BLAST 结果
    - output_format: 输出格式
    
    返回:
    - 格式化后的结果
    """
    # 处理错误情况
    if "error" in result:
        return {
            "success": False,
            "error": result["error"],
            "raw_output": result.get("raw_output", "")
        }
    
    # 如果是表格格式，解析为结构化数据
    if output_format.lower() == "table" and "raw_output" in result:
        hits = []
        columns = ["query_id", "subject_id", "percent_identity", "alignment_length", 
                  "mismatches", "gap_opens", "q_start", "q_end", "s_start", "s_end", 
                  "evalue", "bit_score"]
        
        for line in result["raw_output"].strip().split("\n"):
            values = line.split("\t")
            if len(values) >= len(columns):
                hit = {columns[i]: values[i] for i in range(len(columns))}
                hits.append(hit)
        
        return {
            "success": True,
            "format": "table",
            "hit_count": len(hits),
            "hits": hits
        }
    
    # 对于其他格式，返回原样结果
    return {
        "success": True,
        "format": output_format,
        "result": result
    }

if __name__ == "__main__":
    sequence_server.run()