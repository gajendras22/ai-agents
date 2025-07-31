import os
from typing import Annotated, TypedDict, List, Dict, Any, Optional, Sequence, Union
from langchain_core.tools import BaseTool
from langchain_core.messages import (
    HumanMessage, 
    AIMessage, 
    SystemMessage, 
    ToolMessage,
    BaseMessage
)
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END

# Define the structure of the agent's state, which includes the conversation messages
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "The messages in the conversation so far"]

# Define the DuckDuckGo search tool with a decorator for integration
@tool
def search_duckduckgo(query: str) -> str:
    """Search the web for information using DuckDuckGo."""
    print(f"[search_duckduckgo] - Invoking DuckDuckGo search with query: {query}")
    search_tool = DuckDuckGoSearchRun()
    result = search_tool.invoke(query)
    print(f"[search_duckduckgo] - Search result: {result}")
    return result

class Agent:
    def __init__(self, model, tools, system=""):
        """
        Initialize the Agent with a model, tools, and an optional system message.
        Sets up the state graph for managing the agent's workflow.
        """
        print(f"[Agent.__init__] - Initializing Agent with model: {model}, tools: {tools}, system message: {system}")
        self.system = system
        
        # Create a state graph to manage the agent's workflow
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_llm)  # Node for calling the LLM
        graph.add_node("action", self.take_action)  # Node for executing actions
        graph.add_conditional_edges(
            "llm",
            self.exists_action,  # Conditional check for tool calls
            {True: "action", False: END}  # Transition based on condition
        )
        graph.add_edge("action", "llm")  # Loop back to LLM after action
        graph.set_entry_point("llm")  # Set the entry point of the graph
        self.graph = graph.compile()  # Compile the graph for execution
        
        # Map tools by their names for easy access
        self.tools = {t.name: t for t in tools}
        
        # Bind the tools to the model
        self.model = model.bind_tools(tools)
        print(f"[Agent.__init__] - Agent initialized successfully.")

    def exists_action(self, state: AgentState) -> bool:
        """
        Check if the last message in the state contains any tool calls.
        Returns True if tool calls exist, otherwise False.
        """
        print(f"[Agent.exists_action] - Checking for tool calls in state: {state}")
        result = state['messages'][-1]  # Get the last message in the conversation
        has_tool_calls = hasattr(result, 'tool_calls') and len(result.tool_calls) > 0
        print(f"[Agent.exists_action] - Tool calls exist: {has_tool_calls}")
        return has_tool_calls

    def call_llm(self, state: AgentState) -> Dict[str, Any]:
        """
        Call the LLM with the current conversation messages.
        Adds a system message if not already present and invokes the model.
        """
        print(f"[Agent.call_llm] - Calling LLM with state: {state}")
        messages = state['messages']
        
        # Add the system message if it is defined and not already in the messages
        if self.system and not any(isinstance(msg, SystemMessage) for msg in messages):
            print(f"[Agent.call_llm] - Adding system message to messages.")
            messages = [SystemMessage(content=self.system)] + messages
        
        # Invoke the model with the messages
        message = self.model.invoke(messages)
        print(f"[Agent.call_llm] - LLM response: {message}")
        return {'messages': state['messages'] + [message]}

    def take_action(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute tool calls from the LLM's response.
        Handles errors gracefully and returns the results as ToolMessage objects.
        """
        print(f"[Agent.take_action] - Executing tool calls with state: {state}")
        tool_calls = state['messages'][-1].tool_calls  # Extract tool calls from the last message
        results = []
        
        # Process each tool call
        for t in tool_calls:
            print(f"[Agent.take_action] - Processing tool call: {t}")
            if not t['name'] in self.tools:
                # Handle case where the tool is not found
                print(f"[Agent.take_action] - Tool not found: {t['name']}")
                result = "Error: Tool not found. Please use one of the available tools."
            else:
                try:
                    # Invoke the tool with the provided arguments
                    print(f"[Agent.take_action] - Invoking tool: {t['name']} with args: {t['args']}")
                    result = self.tools[t['name']].invoke(t['args'])
                except Exception as e:
                    # Handle errors during tool execution
                    print(f"[Agent.take_action] - Error executing tool: {e}")
                    result = f"Error executing tool: {str(e)}"
            
            # Append the result as a ToolMessage
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        
        print(f"[Agent.take_action] - Tool execution results: {results}")
        print("[Agent.take_action] - Returning to the model.")
        return {'messages': state['messages'] + results}

def main():
    """
    Main function to initialize the agent and handle user interaction.
    Sets up the environment, initializes the LLM and tools, and starts a conversation loop.
    """
    # Setup API key for the Groq LLM
    print("[main] - Checking for GROQ_API_KEY in environment variables.")
    if not os.environ.get("GROQ_API_KEY"):
        api_key = input("Please enter your Groq API key: ")
        os.environ["GROQ_API_KEY"] = api_key
        print("[main] - GROQ_API_KEY set in environment variables.")

    # Initialize the Groq LLM with the API key
    print("[main] - Initializing ChatGroq LLM.")
    llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model_name="llama3-70b-8192",
    )
    print("[main] - ChatGroq LLM initialized.")

    # Define the tools available to the agent
    print("[main] - Defining tools.")
    tools = [search_duckduckgo]

    # Create a system message to guide the agent's behavior
    system_message = """You are a helpful research assistant that can search for information on the web.
    When you don't know the answer to a question, use the search_duckduckgo tool to find relevant information.
    Always provide thorough and accurate answers based on the search results."""
    print(f"[main] - System message defined: {system_message}")

    # Initialize the agent with the LLM, tools, and system message
    print("[main] - Initializing Agent.")
    agent = Agent(llm, tools, system=system_message)

    print("Welcome to the LangGraph Agent! (Type 'exit' to quit)")
    
    # Start a loop to continuously interact with the user until they type 'exit'
    while True:
        # Prompt the user for input
        user_input = input("\nYour question: ")
        print(f"[main] - User input received: {user_input}")
        
        # Check if the user wants to exit the program
        if user_input.lower() == 'exit':
            print("[main] - Exiting conversation loop.")
            break
        
        # Pass the user's input to the agent for processing
        print("[main] - Invoking agent with user input.")
        result = agent.graph.invoke({
            "messages": [HumanMessage(content=user_input)]  # Wrap the user input in a HumanMessage object
        })
        
        # Extract the agent's response from the result
        final_message = result["messages"][-1]  # The last message in the list is the agent's response
        
        # Display the agent's response to the user
        print(f"[main] - Agent response: {final_message.content}")
        print(f"\nAgent: {final_message.content}")

# Entry point of the script
if __name__ == "__main__":
    main()  # Call the main function to start the program
