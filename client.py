from fastmcp import Client
from rich import print
import json

from client_llm import llm_client
from server_llm import llm_server


# 全局MCP客戶端
client = Client("/Users/wang-work/AMPilot/ampilot/tools/sequence/tools/blast_tools.py")


def parse_json_from_response(response: str) -> dict:
    """Safely parses a JSON object from a string, even with surrounding text."""
    try:
        # Find the start and end of the JSON object
        start_index = response.index("{")
        end_index = response.rindex("}") + 1
        return json.loads(response[start_index:end_index])
    except (ValueError, json.JSONDecodeError):
        return None



async def main():
    async with client:
        tools = await client.list_tools()
        tool_descriptions = [f"- {tool.name}: {tool.description} InputSchema: {tool.inputSchema}" for tool in tools]
        tools_text = "\n".join(tool_descriptions)
        
        # Initial task description
        task_des = "Now the fist task is to build the BLAST database, and we have the reference data ‘/Users/wang-work/AMPilot/ampilot/tools/sequence/test_data/ref.fasta’ which is the nucleotide reference." \
                   "the name of the db is 'test_client_db' and the other parameters are default."

        # --- Stage 1: Client LLM generates a draft tool call ---
        print("[bold yellow]Step 1: Client LLM is generating a draft tool call...[/bold yellow]")
        client_prompt = f"""
        Available tools:
{tools_text}
        My task is: {task_des}"""
        client_response = await llm_client(client_prompt, model="llama3.2:3b")
        print("Client LLM response:", client_response)

        tool_call_draft = parse_json_from_response(client_response)
        if not tool_call_draft:
            print("[bold red]Error: Failed to parse draft tool call from Client LLM.[/bold red]")
            return
        print("Parsed draft tool call:", tool_call_draft)

        # Conversation loop for validation
        conversation_history = f"Original Task: {task_des}\nProposed Tool Call: {json.dumps(tool_call_draft)}"
        max_retries = 3
        for i in range(max_retries):
            print(f"\n[bold yellow]Step 2 (Attempt {i+1}): Server LLM is validating the draft...[/bold yellow]")
            
            server_prompt = conversation_history
            server_response_str = await llm_server(server_prompt, model="llama3.2:3b")
            print("Server LLM response:", server_response_str)

            validation_result = parse_json_from_response(server_response_str)
            if not validation_result:
                print("[bold red]Error: Failed to parse validation result from Server LLM.[/bold red]")
                return

            # --- Stage 3: Act based on validation ---
            if validation_result.get("status") == "incomplete":
                print("\n[bold cyan]Step 3: Information is incomplete. Server LLM is asking for more input...[/bold cyan]")
                question = validation_result.get("question", "I need more information, but I'm not sure what to ask.")
                print(f"[bold magenta]Server's Question:[/bold magenta] {question}")
                
                # Simulate user providing an affirmative answer
                user_answer = "Yes, that is correct. The file exists and is a valid nucleotide sequence file. Please proceed."
                print(f"[bold green]Simulated User Answer:[/bold green] {user_answer}")

                # Update conversation history
                conversation_history += f"\nServer's Question: {question}\nYour Answer: {user_answer}"
                continue # Go back to the server LLM with the new info
            
            # If status is not "incomplete", we assume it's complete and break the loop
            print("\n[bold green]Step 3: Validation complete. Proceeding to execute the tool.[/bold green]")
            tool_name = tool_call_draft.get("tool")
            tool_input = tool_call_draft.get("input", {})
            print(f"Calling tool: [bold]{tool_name}[/bold] with input: {tool_input}")

            # --- Final Stage: Execute the tool ---
            result = await client.call_tool(tool_name, tool_input)
            print(f"\n[bold green]Tool result:[/bold green]\n{result}")
            return # Exit after successful execution

        print(f"\n[bold red]Execution failed after {max_retries} retries. Server LLM could not get a complete tool call.[/bold red]")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 






# async def main():
#     async with client:
#         tools = await client.list_tools()
#         print("[bold green]已注册的工具:[/bold green]")
#         for tool in tools:
#             print(f"  - [bold]{tool.name}[/bold]: {tool.description}")
#             if tool.inputSchema:
#                 print(f"    - 输入参数: {tool.inputSchema}")

#         # 测试1: 创建BLAST数据库
#         print("\n[bold blue]测试1: 创建BLAST数据库[/bold blue]")
#         db_result = await client.call_tool(
#             "create_blast_db",
#             {
#                 "input_file": "/Users/wang-work/AMPilot/ampilot/tools/sequence/test_data/ref.fasta",
#                 "db_type": "nucleotide",
#                 "db_name": "test_client_db",
#                 "title": "测试客户端数据库"
#             })
#         print("数据库创建结果:")
#         print(db_result)

#         # 测试2: BLAST搜索
#         print("\n[bold blue]测试2: BLAST序列搜索[/bold blue]")
#         result = await client.call_tool(
#             "blast_search",
#             {
#                 "sequence": "ACATCAGTGCAGTCAGTGGGCAGTCAGTGCATGTCAGTCAGTCAGTGTCAGTCGATCAGCGAGTCAGTGCATGTCAGTGCATGTACGTCATGTCAGTGCTAGTCAGTGCATGTCAGTGCATGCTATG",
#                 "program": "blastn",
#                 "database": "test_client_db",
#                 "evalue": 1e-3,
#                 "max_hits": 5,
#                 "output_format": "table",
#                 "threads": 1
#             })
#         print("BLAST搜索结果:")
#         print(result)

#         # 测试3: 获取BLAST工具信息 (暂时跳过，因为工具可能没有正确注册)
#         print("\n[bold blue]测试3: 获取BLAST工具信息[/bold blue]")
#         try:
#             info_result = await client.call_tool("get_blast_info", {})
#             print("BLAST工具信息:")
#             print(info_result)
#         except Exception as e:
#             print(f"获取工具信息失败: {e}")
#             print("跳过此测试")


# if __name__ == "__main__":

#     import asyncio
#     asyncio.run(main())