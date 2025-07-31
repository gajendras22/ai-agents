from typing import TypedDict,List 
from langchain_core.messages import BaseMessage 
from langchain_core.runnables import Runnable 
from langchain_groq import ChatGroq 
from langgraph.graph import StateGraph,END 
from langchain_core.messages import HumanMessage    

class AgentState(TypedDict):    
    """A message that is sent to the agent."""    
    messages: List[BaseMessage]    
    
llm=ChatGroq(    groq_api_key="Your API KEY",    model="llama-3.3-70b-versatile") 

def call_model(state:AgentState) -> AgentState:    
    """Call the model with the current state."""    
    response = llm.invoke(state["messages"])    
    state["messages"].append(response)    
    return state 

builder = StateGraph(AgentState) 
builder.add_node( "respond",call_model,) 
builder.set_entry_point("respond") 
builder.add_edge("respond",END) 
graph = builder.compile() 
inputs={"messages": [HumanMessage(content="Hello, how are you?")]}  

for step in graph.stream(inputs):    
    print(step)