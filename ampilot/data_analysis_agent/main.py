import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage

# Import our custom components
from tools import hypothesis_test_z_test, regression_analysis, visualize_data_curve
from graph import create_workflow
from prompts import SYSTEM_PROMPT

def main():
    """
    Main execution function.
    """
    load_dotenv()
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in .env file.")

    # Model and Tools Initialization ---
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "LangGraph Data Agent"),
        }
    )

    tools = [hypothesis_test_z_test, regression_analysis, visualize_data_curve]
    llm_with_tools = llm.bind_tools(tools)

    # Create the Agent Workflow ---
    app = create_workflow(llm_with_tools, tools)

    # Prepare Sample Data ---
    data = {
        'Time': range(1, 101),
        'GroupA_Score': [x + 5 + (i % 5) for i, x in enumerate(np.random.randn(100) * 10)],
        'GroupB_Score': [x + 8 + (i % 3) for i, x in enumerate(np.random.randn(100) * 10)],
        'Performance_Metric': [1.5 * t + 20 + np.random.randn() * 5 for t in range(1, 101)]
    }
    df = pd.DataFrame(data)
    csv_file_path = "sample_data.csv"
    df.to_csv(csv_file_path, index=False)
    print(f"Sample data file created at: '{csv_file_path}'")
    
    # Run Agent with User Questions ---
    questions = [
        f"Hello, my data file is at '{csv_file_path}'. Please compare 'GroupA_Score' and 'GroupB_Score' using a Z-test to see if they come from the same distribution. Summarize the findings.",
        f"I want to analyze the impact of the 'Time' variable on the 'Performance_Metric' variable. Please run a regression analysis using the file '{csv_file_path}' and explain the R-squared and coefficients in simple terms.",
        f"Please create a line chart for 'GroupA_Score' and 'GroupB_Score' from '{csv_file_path}'. Use the 'Time' column as the x-axis and save the chart as 'scores_over_time.png'. Let me know when it's done."
    ]
    
    for i, question in enumerate(questions):
        print("\n" + "="*50)
        print(f"🚀 Processing Request {i+1} 🚀")
        print(f"User Query: {question}")
        print("="*50 + "\n")
        
        inputs = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=question)
            ]
        }
        
        for output in app.stream(inputs, stream_mode="values"):
            last_message = output["messages"][-1]
            if isinstance(last_message, ToolMessage):
                 print(f"--- Tool Output ---\n{last_message.content}\n")
            elif hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                 print(f"--- LLM decided to call a tool ---")
            else:
                 print("--- LLM Final Response ---")
                 print(last_message.content)

if __name__ == "__main__":

    main()

