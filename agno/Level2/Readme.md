**1. Built-in:**
--
*built-in.py* is an example script that demonstrates how to create a conversational AI agent with built-in memory using the agno framework and the Gemini language model.

Key Features :
- Built-in memory to include previous chat history in conversations.
- Contextual awareness by specifying how many previous responses to remember.
- A positive, polite assistant persona via a custom description.
- Creates a web-based playground app for interactive conversations with the agent.

⚙️ **Setup**

Create a .env file in the project root:

```GOOGLE_API_KEY=your-api-key-here```

▶️ **Run the script**

```python built-in.py```
   
   ![image](https://github.com/user-attachments/assets/4d0e8de5-207f-46bd-8f18-1660df049933)

**2. Built-in-cli:**
--
*built-in-cli.py* is a command-line example that demonstrates how to interact with a conversational AI agent with built-in memory using the agno framework and the Gemini language model.

⚙️ **Setup**

Create a .env file in the project root:

```GOOGLE_API_KEY=your-api-key-here```

▶️ **Run the script**

```python built-in-cli.py```


   ![image](https://github.com/user-attachments/assets/79b74ba1-279c-4dc1-ae75-9e0002b50cf1)
   ![image](https://github.com/user-attachments/assets/7e5c61a5-0f11-407f-8b3b-8ab463159594)

**3. Session-storage:**
--
*session-storage.py* is an example script that demonstrates how to create a conversational AI agent with persistent session memory using the agno framework and the Gemini language model.

Initializes an AI agent ("SessionStorageAgent") with:
- The Gemini model for language understanding.
- Persistent conversation memory using SQLite, so chat history is saved across sessions and script restarts.
- A fixed session ID to ensure continuity of the same conversation over multiple runs.
- Contextual awareness by including previous chat history in the model input.

⚙️ **Setup**

Create a .env file in the project root:

```GOOGLE_API_KEY=your-api-key-here```

▶️ **Run the script**

```python session-storage.py```

This file is a practical starting point for building persistent, context-aware conversational agents using agno, Gemini, and SQLite storage.

   ![image](https://github.com/user-attachments/assets/a2ea5401-b856-425f-b763-bf1b9dd7abcb)
   ![image](https://github.com/user-attachments/assets/26b5e17b-0f25-483e-9044-11b9016652bf)

**4. session-storage-cli:**
--
*session-storage-cli.py* is a command-line example that demonstrates how to create a conversational AI agent with persistent session memory using the agno framework and the Gemini language model.

⚙️ **Setup**

Create a .env file in the project root:

```GOOGLE_API_KEY=your-api-key-here```

▶️ **Run the script**

```python session-storage-cli.py```


   https://github.com/user-attachments/assets/0e7c44b3-3b6f-43b7-860c-f97a49143afa

**5. user-memory:**
--
*user-memory.py* is an advanced example demonstrating how to build a conversational AI agent with both persistent session memory and user-specific long-term memory using the agno framework and the Gemini language model.

This sets up two types of memory:
- User-specific memory: Stores long-term user information in a dedicated SQLite database.
- Session (conversation) memory: Stores chat history in a separate SQLite database for persistent context across sessions.
   
Initializes an AI agent ("UserMemoryAgent") with:
- The Gemini model for language understanding.
- Both user and agent memory enabled for richer, more personalized conversations.
- Contextual awareness by including previous chat history in the model input.
- Markdown formatting, streaming responses, and session summaries enabled.

⚙️ **Setup**

Create a .env file in the project root:

```GOOGLE_API_KEY=your-api-key-here```

▶️ **Run the script**

```python user-memory.py```

This file is a practical starting point for building highly personalized, context-aware conversational agents using agno, Gemini, and advanced memory management.

   ![image](https://github.com/user-attachments/assets/e02bad5a-22ff-41a4-b979-f56abe5a9129)
   ![image](https://github.com/user-attachments/assets/42252f89-8939-40ee-a6b0-b491c5381de0)

**6. user-memory-cli:**
--
*user-memory-cli.py* is an advanced command-line example that demonstrates how to build a conversational AI agent with both persistent session memory and user-specific long-term memory using the agno framework and the Gemini language model.

⚙️ **Setup**

Create a .env file in the project root:

```GOOGLE_API_KEY=your-api-key-here```

▶️ **Run the script**

```python user-memory-cli.py```

This file is a practical starting point for building highly personalized, context-aware conversational agents in a command-line environment using agno, Gemini, and advanced memory management.

   ![image](https://github.com/user-attachments/assets/c45a812f-72cc-46e5-922d-06934ce16135)
   ![image](https://github.com/user-attachments/assets/9fc6c99f-a395-48c0-b8e5-f081db019bba)
   ![image](https://github.com/user-attachments/assets/5738eac2-3891-45ac-a885-f6b954c6c215)
   ![image](https://github.com/user-attachments/assets/9907250f-51d2-4d71-8596-9e4979ec3347)
   ![image](https://github.com/user-attachments/assets/700be53e-c1c9-4087-83b1-e1a1e25a53f1)
   ![image](https://github.com/user-attachments/assets/8edc5b33-488b-41f4-a2fa-b54739fc9b76)






   
   
