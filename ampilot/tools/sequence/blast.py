import os
import subprocess
import tempfile
import json
from typing import Dict, List, Union, Optional, Any
import asyncio
import shutil
from pathlib import Path


class BlastTool:
    """封装BLAST工具的类，直接调用BLAST+命令行工具"""
    
    def __init__(self):
        """
        初始化BlastTool，检查系统中是否已安装BLAST+工具
        """
        self._tools_available: Dict[str, bool] = {}
        self._checked_installation = False
    
    async def _check_blast_installation_async(self):
        """
        异步检查BLAST+工具是否已安装并可用
        """
        if self._checked_installation:
            return

        essential_tools = ["blastp", "blastn", "blastx", "tblastn", "tblastx", "makeblastdb"]
        missing_tools = []
        
        async def check(tool):
            is_available = await self._is_command_available(tool)
            self._tools_available[tool] = is_available
            if not is_available:
                missing_tools.append(tool)

        await asyncio.gather(*(check(tool) for tool in essential_tools))
        
        if missing_tools:
            print(f"警告: 以下BLAST+工具未找到或不可用: {', '.join(missing_tools)}")
        self._checked_installation = True
    
    async def blast_search(self, 
                     sequence: str, 
                     program: str = "blastp", 
                     database: str = "nr", 
                     evalue: float = 1e-5, 
                     max_hits: int = 50, 
                     outfmt: int = 15,  # 15为JSON格式
                     threads: int = 1,
                     additional_params: Dict[str, Any] = None
                     ) -> Dict[str, Any]:
        """
        执行BLAST搜索
        
        参数:
        - sequence: 查询序列或序列文件路径
        - program: BLAST程序类型(blastp, blastn, tblastn等)
        - database: 目标数据库名称
        - evalue: E值阈值
        - max_hits: 最大结果数量
        - outfmt: 输出格式(0-17)
        - threads: 使用的线程数
        - additional_params: 其他BLAST参数
        
        返回:
        - BLAST搜索结果(字典格式)
        """
        # 创建临时文件存储序列
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False
                                         ) as temp_file:
            # 如果sequence不是文件路径，则假定是序列字符串
            if not os.path.isfile(sequence):
                # 简单FASTA格式写入
                temp_file.write(">query\n")
                temp_file.write(sequence)
                query_file = temp_file.name
            else:
                query_file = sequence
        
        # 检查程序是否可用
        if not await self._is_tool_available(program):
            return {"error": f"BLAST程序 {program} 不可用，请确保已正确安装"}
            
        # 构建命令
        cmd_args = [
            program,
            "-query", query_file,
            "-db", database,
            "-evalue", str(evalue),
            "-max_target_seqs", str(max_hits),
            "-outfmt", str(outfmt),
            "-num_threads", str(threads)
        ]
        
        # 添加额外参数
        if additional_params:
            for key, value in additional_params.items():
                cmd_args.extend([f"-{key}", str(value)])
        
        # 执行命令
        result = await self._run_command_async(cmd_args)
        
        # 清理临时文件
        if query_file != sequence:  # 如果使用了临时文件
            os.unlink(query_file)
        
        # 解析结果
        if outfmt == 15:  # JSON格式
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return {"error": "无法解析BLAST JSON输出", "raw_output": result}
        else:
            return {"raw_output": result}
    
    async def create_blast_db(self, 
                        input_file: str, 
                        dbtype: str = "prot", 
                        output_dir: Optional[str] = None,
                        dbname: Optional[str] = None, 
                        title: Optional[str] = None
                        ) -> Dict[str, Any]:
        """
        创建BLAST数据库
        
        参数:
        - input_file: 包含序列的FASTA文件
        - dbtype: 数据库类型 ('prot'或'nucl')
        - output_dir: 数据库输出目录(可选, 默认为当前目录)
        - dbname: 数据库名称(可选)
        - title: 数据库标题(可选)
        
        返回:
        - 创建结果信息
        """
        if not os.path.isfile(input_file):
            return {"error": f"找不到输入文件: {input_file}"}
        
        # 设置数据库名称
        if not dbname:
            dbname = os.path.splitext(os.path.basename(input_file))[0]

        # 如果提供了输出目录，则创建目录并组合输出路径
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            db_path = os.path.join(output_dir, dbname)
        else:
            db_path = dbname
        
        # 检查程序是否可用
        if not await self._is_tool_available("makeblastdb"):
            return {"error": "makeblastdb 工具不可用，请确保已正确安装"}
            
        # 构建命令
        cmd_args = [
            "makeblastdb",
            "-in", input_file,
            "-dbtype", dbtype,
            "-out", db_path
        ]
        
        if title:
            cmd_args.extend(["-title", title])

        print("Creating BLAST database with command:", " ".join(cmd_args))
        executed_command = " ".join(cmd_args)
        
        # 执行命令
        result = await self._run_command_async(cmd_args)
        
        # 检查数据库文件是否创建成功
        search_path = Path(output_dir) if output_dir else Path()
        db_files = list(search_path.glob(f"{dbname}.*"))
        
        # 如果数据库消息中包含"Adding sequences"，通常表示成功
        success = "Adding sequences" in result
        
        return {
            "success": success,
            "dbname": dbname,
            "files": [str(f) for f in db_files],
            "message": result,
            "executed_command": executed_command
        }
    
    async def _run_command_async(self, cmd_args, stdin_data=None, stdout_file=None):
        """
        异步执行命令行命令
        
        参数:
        - cmd_args: 命令参数列表
        - stdin_data: 标准输入数据(可选)
        - stdout_file: 标准输出文件对象(可选)
        
        返回:
        - 命令输出
        """
        # 创建子进程
        if stdout_file:
            # 如果提供了文件对象，则直接写入文件
            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdin=asyncio.subprocess.PIPE if stdin_data else None,
                stdout=stdout_file,
                stderr=asyncio.subprocess.PIPE
            )
            
            # 等待进程完成
            if stdin_data:
                stdin_bytes = stdin_data.encode() if isinstance(stdin_data, str) else stdin_data
                stdout, stderr = await process.communicate(stdin_bytes)
            else:
                stdout, stderr = await process.communicate()
                
            if process.returncode != 0:
                return f"命令执行失败(返回码 {process.returncode}): {stderr.decode()}"
            
            return ""  # 输出已经写入文件
        else:
            # 否则捕获输出
            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdin=asyncio.subprocess.PIPE if stdin_data else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # 等待进程完成
            if stdin_data:
                stdin_bytes = stdin_data.encode() if isinstance(stdin_data, str) else stdin_data
                stdout, stderr = await process.communicate(stdin_bytes)
            else:
                stdout, stderr = await process.communicate()
                
            if process.returncode != 0:
                return f"命令执行失败(返回码 {process.returncode}): {stderr.decode()}"
            
            return stdout.decode()
            
    @property
    def available_tools(self):
        """
        获取当前系统中可用的BLAST+工具列表
        
        返回:
        - 可用工具的字典，键为工具名，值为是否可用
        """
        return dict(self._tools_available)
    
    async def _is_tool_available(self, tool_name: str) -> bool:
        """
        异步检查指定工具是否可用
        
        参数:
        - tool_name: 工具名称
        
        返回:
        - 工具是否可用 (True/False)
        """
        # 如果从未检查过，先执行一次完整的异步检查
        if not self._checked_installation:
            await self._check_blast_installation_async()

        if tool_name in self._tools_available:
            return self._tools_available[tool_name]
        
        # 如果是列表中没有的工具，单独检查
        is_available = await self._is_command_available(tool_name)
        self._tools_available[tool_name] = is_available
        return is_available

    async def _is_command_available(self, command: str) -> bool:
        """异步检查单个命令是否可用"""
        try:
            process = await asyncio.create_subprocess_exec(
                command, "-version",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            return await process.wait() == 0
        except (FileNotFoundError, OSError):
            return False
