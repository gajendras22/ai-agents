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

# Define the state structure
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "The messages in the conversation so far"]

# Define the search tool with proper decoration
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
        print(f"[Agent.__init__] - Initializing Agent with model: {model}, tools: {tools}, system message: {system}")
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_llm)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)
        print(f"[Agent.__init__] - Agent initialized successfully.")

    def exists_action(self, state: AgentState) -> bool:
        """Check if the last message contains any tool calls."""
        print(f"[Agent.exists_action] - Checking for tool calls in state: {state}")
        result = state['messages'][-1]
        has_tool_calls = hasattr(result, 'tool_calls') and len(result.tool_calls) > 0
        print(f"[Agent.exists_action] - Tool calls exist: {has_tool_calls}")
        return has_tool_calls

    def call_llm(self, state: AgentState) -> Dict[str, Any]:
        """Call the LLM with the current messages."""
        print(f"[Agent.call_llm] - Calling LLM with state: {state}")
        messages = state['messages']
        if self.system and not any(isinstance(msg, SystemMessage) for msg in messages):
            print(f"[Agent.call_llm] - Adding system message to messages.")
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        print(f"[Agent.call_llm] - LLM response: {message}")
        return {'messages': state['messages'] + [message]}

    def take_action(self, state: AgentState) -> Dict[str, Any]:
        """Execute tool calls from the LLM's response."""
        print(f"[Agent.take_action] - Executing tool calls with state: {state}")
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"[Agent.take_action] - Processing tool call: {t}")
            if not t['name'] in self.tools:
                print(f"[Agent.take_action] - Tool not found: {t['name']}")
                result = "Error: Tool not found. Please use one of the available tools."
            else:
                try:
                    print(f"[Agent.take_action] - Invoking tool: {t['name']} with args: {t['args']}")
                    result = self.tools[t['name']].invoke(t['args'])
                except Exception as e:
                    print(f"[Agent.take_action] - Error executing tool: {e}")
                    result = f"Error executing tool: {str(e)}"
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print(f"[Agent.take_action] - Tool execution results: {results}")
        print("[Agent.take_action] - Returning to the model.")
        return {'messages': state['messages'] + results}

def main():
    # Setup API key
    print("[main] - Checking for GROQ_API_KEY in environment variables.")
    if not os.environ.get("GROQ_API_KEY"):
        api_key = input("Please enter your Groq API key: ")
        os.environ["GROQ_API_KEY"] = api_key
        print("[main] - GROQ_API_KEY set in environment variables.")

    # Initialize the Groq LLM
    print("[main] - Initializing ChatGroq LLM.")
    llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model_name="llama3-70b-8192",
    )
    print("[main] - ChatGroq LLM initialized.")

    # Define the tools
    print("[main] - Defining tools.")
    tools = [search_duckduckgo]

    # Create system message
    system_message = """You are a helpful research assistant that can search for information on the web.
    When you don't know the answer to a question, use the search_duckduckgo tool to find relevant information.
    Always provide thorough and accurate answers based on the search results."""
    print(f"[main] - System message defined: {system_message}")

    # Initialize the agent
    print("[main] - Initializing Agent.")
    agent = Agent(llm, tools, system=system_message)

    # Simple conversation loop
    print("Welcome to the LangGraph Agent! (Type 'exit' to quit)")
    while True:
        user_input = input("\nYour question: ")
        print(f"[main] - User input received: {user_input}")
        if user_input.lower() == 'exit':
            print("[main] - Exiting conversation loop.")
            break
        
        # Process the query through the agent
        print("[main] - Invoking agent with user input.")
        result = agent.graph.invoke({
            "messages": [HumanMessage(content=user_input)]
        })
        
        # Display the final answer
        final_message = result["messages"][-1]
        print(f"[main] - Agent response: {final_message.content}")
        print(f"\nAgent: {final_message.content}")

if __name__ == "__main__":
    main()
