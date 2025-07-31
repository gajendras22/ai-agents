from typing import TypedDict,List 
from langchain_core.messages import BaseMessage 
from langchain_core.runnables import Runnable 
from langchain_groq import ChatGroq 
from langgraph.checkpoint.memory import InMemorySaver 
from langgraph.graph import StateGraph,END 
from langchain_core.messages import HumanMessage 

class AgentState(TypedDict):    
    """A message that is sent to the agent."""    
    messages: List[BaseMessage]    
llm=ChatGroq(groq_api_key="Your API KEY", model="llama-3.3-70b-versatile") 
    
def llm_node(state:AgentState) -> AgentState:    
    """Call the model with the current state."""    
    response = llm.invoke(state["messages"])    
    return {"messages": state["messages"] + [response]}   

checkpointer = InMemorySaver()     
graph = StateGraph(AgentState) 
graph.add_node( "llm_node",llm_node,) 
graph.set_entry_point("llm_node") 
app = graph.compile() 
thread_id = "user-session-001"  

# Unique identifier for the session - user defined 
input_state={"messages": [HumanMessage(content="Hi, who are you?")]}  
result = app.invoke(input_state, thread_id=thread_id) 
print(result["messages"][-1].content)# Output the last message from the model 
new_input = {    "messages": result["messages"] + [HumanMessage(content="Explain OOP concepts")] } 
result = app.invoke(new_input, thread_id=thread_id) 
print(result["messages"][-1].content)# Output the last message from the model