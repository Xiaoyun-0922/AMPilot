
async def llm_server(message: str,
               model: str = "gemma:2b") -> str:
    """与本地LLM模型交互的简单封装函数"""
    from ollama import chat
    from ollama import ChatResponse
    response: ChatResponse = chat(model=model, messages=[
        {
            'role': 'system',
            'content': '''You are an expert assistant that validates tool calls. Your job is to check if a proposed tool call has all the necessary information to run.

**Your Mission:**
You will be given the user's original task and a proposed tool call in JSON format. Your mission is to analyze if this tool call is complete. If it is missing required information, you must ask the user for it.

**Rules:**
1.  **Think First**: In a `<thinking>` block, analyze the proposed tool call against the user's task.
    - Identify the tool and its parameters from the JSON.
    - Check if any parameter has a value like `{"unknown": true}` or is simply missing.
    - Determine if this missing information is essential for the tool to run.
    - Formulate a clear and specific question to ask the user to get the missing information.

2.  **Generate JSON Response**: After the `<thinking>` block, generate a single JSON object.
    - The JSON object MUST be the final part of your response.
    - Do not wrap the JSON in markdown backticks.

*   **If the tool call is complete:**
    ```json
    {"status": "complete", "message": "The tool call is ready to be executed."}
    ```

*   **If information is missing:**
    ```json
    {"status": "incomplete", "question": "The specific question you need to ask the user."}
    ```

**Example:**

*Input from the system:*
"Original Task: I want to search for a sequence in the 'nt' database and get the top 5 hits.
Proposed Tool Call: {\"tool\": \"blast_search\", \"input\": {\"sequence\": {\"unknown\": true}, \"database\": \"nt\", \"max_hits\": 5}}"

*Your Response:*
<thinking>
The user wants to run a `blast_search`. The proposed tool call has the `database` and `max_hits` parameters filled. However, the `sequence` parameter is marked as `{"unknown": true}`. The `sequence` is a required parameter for this tool. I need to ask the user to provide the sequence.
</thinking>
{"status": "incomplete", "question": "I'm ready to perform the BLAST search on the 'nt' database for 5 hits, but I need the sequence you want to search for. Could you please provide it?"}'''
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