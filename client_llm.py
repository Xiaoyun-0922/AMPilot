
async def llm_client(message: str,
               model: str = "gemma:2b") -> str:
    """与本地LLM模型交互的简单封装函数"""
    from ollama import chat
    from ollama import ChatResponse

    response: ChatResponse = chat(model=model, messages=[
        {
            'role': 'system',
            'content': '''You are an expert assistant that helps users accomplish tasks by generating tool calls in JSON format.

**Your Mission:** First, think step-by-step about the user's request. Then, based on your thinking, generate a single, precise JSON object for the tool call.

**Rules:**
1.  **Think First**: In a `<thinking>` block, analyze the user's task, identify the best tool, and check the parameters.
    - **Analyze Task**: What is the user trying to achieve?
    - **Select Tool**: Which tool is the most appropriate for this task?
    - **Check Parameters**:
        - Are all **required** parameters present in the user's request? If not, what is missing?
        - Are there **optional** parameters? Note them down. It's good to use defaults unless the user specifies otherwise.
        - Are there parameters that require a user-defined name (like `db_name`)? If the user hasn't provided one, I should consider using a sensible default or noting that it needs to be defined.
        - **Crucially**: If the user explicitly says "use defaults" for other parameters, I MUST accept that and not ask about them. I will proceed using the default values.
2.  **Generate JSON**: After the `<thinking>` block, generate the JSON for the tool call.
    - The JSON object MUST be the final part of your response.
    - Do not wrap the JSON in markdown backticks.

**Example:**

*User Request:* "I want to search for a sequence in the 'nt' database and get the top 5 hits."
*Your Response (this is the format you must follow):*
<thinking>
1.  **Analyze Task**: The user wants to perform a sequence search.
2.  **Select Tool**: The `blast_search` tool is perfect for this.
3.  **Check Parameters**:
    - `sequence`: Required, but not provided by the user. I need to ask the user for the sequence.
    - `database`: Provided as 'nt'.
    - `max_hits`: Provided as 5.
    - `program`, `evalue`, `output_format`, `threads`, `additional_params` are optional and were not specified. I will use their default values.
</thinking>
{"tool": "blast_search", "input": {"sequence": {"unknown": true}, "database": "nt", "max_hits": 5}}'''
        },
        {
            'role': 'user',
            'content': message,
        },
    ],
    options={
        'temperature': 0
        })
    print(f"LLM raw Input: {message}")
    return response.message.content