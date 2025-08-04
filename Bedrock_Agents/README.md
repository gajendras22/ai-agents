# Bedrock Agents Repository

This repository contains a collection of Bedrock agents designed to solve complex problems using an iterative approach. Each level introduces new capabilities, building upon the previous levels to create increasingly sophisticated agentic solutions. This approach is ideal for training individuals with no prior knowledge of Generative AI (GenAI) to gain familiarity and confidence in agentic frameworks.

## What are Bedrock Agents?

Bedrock agents are AI-driven systems built on Amazon Bedrock, designed to perform specific tasks by leveraging foundational models, memory, and tools. These agents can be customized to solve a wide range of problems, from simple question-answering to complex multi-agent workflows.

### Key components used to implement these levels

1. **Memory**:
   - Memory enables agents to retain context across interactions, making them capable of providing context-aware responses.
   - Example: Conversational memory allows agents to remember user preferences or prior queries.

2. **Prompt**:
   - Prompts define the behavior and instructions for the agent.
   - Example: "You are a helpful assistant. Your task is to answer the user's questions clearly and concisely."

3. **Action Groups**:
   - Action groups define a sequence of actions or tools that the agent can use to perform tasks.
   - Example: A web search action group integrates APIs like Tavily Search to fetch real-time information.

4. **Foundational Models**:
   - These are pre-trained models provided by Amazon Bedrock, such as Amazon Titan or Amazon Nova, used for tasks like text generation, embedding, or summarization.

5. **Knowledge Bases**:
   - Knowledge bases store proprietary information in a vectorized format, enabling agents to perform Retrieval Augmented Generation (RAG).
   - Example: A resume ingested into a knowledge base allows the agent to answer questions about the user's qualifications.

6. **Flows**:
   - Flows define the orchestration of multi-agent and multi-tool systems.
   - Example: A flow for the NotebookLM Clone agent might include steps for ingesting files, generating mind maps, and creating podcasts.
   
7. **Lambda Functions**
   - Lambda functions are used to extend the capabilities of Bedrock agents by integrating external tools and APIs. 
   - These functions are essential for enabling agents to interact with the real world and perform tasks beyond the capabilities of foundational models.

## Levels Overview

### **Level 1: Basic**
- A simple agent powered by an LLM to answer basic questions.

### **Level 2: Conversational Memory**
- Adds memory to manage conversation state and provide context-aware responses.

### **Level 3: Tools**
- Integrates web search capabilities using the Tavily Search API.

### **Level 4: Vector Store**
- Implements Retrieval Augmented Generation (RAG) using a vector store.

### **Level 5: Notebook LM mimic**
- Mimics Google NotebookLM with multi-agent, multi-tool capabilities.

## References

- For detailed descriptions of each level, visit the [Agentic AI Sample Problem](http://github.com/cladius/agentic-ai/blob/master/sample_problem.md).
- For more examples and advanced use cases, visit the [Amazon Bedrock Agent Samples](https://github.com/awslabs/amazon-bedrock-agent-samples).




