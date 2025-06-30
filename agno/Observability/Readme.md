Langfuse
--
langfuse.py is an example script demonstrating how to enable advanced observability and tracing for AI agent interactions using [Langfuse](https://langfuse.com/) and OpenTelemetry with the agno framework and Gemini language model.

Key Features:
1. Configures OpenTelemetry tracing to export agent traces to Langfuse for monitoring and analytics.
2. Instruments the agno framework so all agent actions and tool calls are traced and sent to Langfuse.
3. Creates a stock price agent using the Gemini model and YFinance tools, with debug mode enabled.
4. Runs a sample query ("What is the current price of Tesla?") and traces the full reasoning and tool usage to Langfuse.

⚙️ Setup

Create a .env file in the project root:

```
GOOGLE_API_KEY=your_google_api_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
```

▶️ Run the script

```python langfuse.py```

This script is ideal for developers who want to add observability, debugging, and analytics to their AI agent workflows using Langfuse and OpenTelemetry.

OUTPUT:
--
![image](https://github.com/user-attachments/assets/f488dad8-6bae-43d2-94e0-9b2d94aacb4d)
