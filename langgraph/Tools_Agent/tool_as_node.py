from langgraph.prebuilt import ToolNode 
from langchain_core.tools import tool 
from langchain_core.messages import AIMessage, ToolCall 

# Define tools 
@tool 
def get_weather(location: str) -> str:
    """Call to get the current weather."""    
    if location.lower() in ["sf", "san francisco"]:        
        return "It's 60 degrees and foggy."    
    else:        
        return "It's 90 degrees and sunny." 
    
@tool 
def get_coolest_cities() -> str:    
    """Get a list of coolest cities."""    
    return "NYC, SF, Seattle, Portland" 

# Register tools to the ToolNode 
tool_node = ToolNode([get_weather, get_coolest_cities]) 

# Build the tool call message (with content as an empty string) 
tool_call_message = AIMessage(content="",  # Must be a valid string    
                              tool_calls=[ToolCall(id="1",name="get_weather",
                                                   args={"location": "San Francisco"},),
                                                     ToolCall(id="2",name="get_coolest_cities",args={},  ),    ] ) 
# Run the ToolNode 
state = {"messages": [tool_call_message]} 
result = tool_node.invoke(state) 

# Output results 
print("\nToolNode Output Messages:") 
for msg in result["messages"]:    
    print(msg)