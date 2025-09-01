import json
import uuid

from retrieval_graph.graph import app


def run_agent_cli():
    """
    Runs an interactive command-line interface for the AMPilot agent
    using the graph's built-in memory (checkpointer).
    """
    print("=" * 50)
    print("      Welcome to the AMPilot CLI Assistant!     ")
    print("=" * 50)
    print("You can now ask questions about Antimicrobial Peptides.")
    print("Type 'quit' or 'exit' to end the session.")
    print("-" * 50)

    # Each conversation needs a unique ID when using a checkpointer.
    # We'll generate a random one for each CLI session.
    conversation_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": conversation_id}}

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                print("AMPilot: Goodbye!")
                break

            # The input to the graph is now just the single new message.
            # The checkpointer handles loading the past messages.
            inputs = {"messages": [("human", user_input)]}

            print("AMPilot: ", end="", flush=True)

            final_response_content = ""
            # Use .stream() to get the agent's response
            for chunk in app.stream(inputs, config=config, stream_mode="values"):
                # The streamed chunk is now the full AgentState
                final_message = chunk["messages"][-1]
                if final_message.content:
                    # Print the content of the latest message
                    print(final_message.content, end="", flush=True)
                    final_response_content = final_message.content

            # Attempt to parse and pretty-print the final JSON response
            try:
                json_start = final_response_content.find("{")
                json_end = final_response_content.rfind("}") + 1
                if json_start != -1 and json_end != -1:
                    json_str = final_response_content[json_start:json_end]
                    parsed_json = json.loads(json_str)
                    print("\r" + " " * len(final_response_content), end="\r")
                    print("AMPilot (Formatted JSON):")
                    print(json.dumps(parsed_json, indent=2))
                else:
                    print()
            except (json.JSONDecodeError, IndexError):
                print()

        except KeyboardInterrupt:
            print("\nAMPilot: Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            break


if __name__ == "__main__":
    run_agent_cli()
