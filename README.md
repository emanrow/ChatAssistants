# ChatAssistants

`ChatAssistants` is a Python module designed to standardize and manage the flow of messages for chat applications, particularly for systems utilizing Language Learning Models (LLMs) via REST APIs.

## Features

- Define and manage chat messages with unique IDs and roles (user, assistant, system).
- Manage conversations with a collection of chat messages.
- Support for system messages and chat exchanges (pairs of prompt and response).
- Serialization and deserialization of chat message objects to and from JSON.
- Interface for to format submissions to, and responses from, LLMs.

## Classes

- `ChatMessage`: Represents a single chat message with unique ID, role, and content.
- `SystemChatMessage`: Specialized `ChatMessage` with a role of "system".
- `ChatExchange`: Manages a pair of chat messages, one as a prompt and the other as a response.
- `ConversationThread`: Manages a conversation thread that consists of a system message followed by a series of chat exchanges.
- `AbstractChatAdapter` implementable interface for validating submissions to and from LLMs, including a provided `OpenAIAdapter` for OpenAI `ChatCompletion` submissions.

## Usage

```python
from ChatAssistants import ChatMessage, SystemChatMessage, ChatExchange, ConversationThread, AbstractChatAdapter
```

# Create individual messages
```python
user_message = ChatMessage(role="user", content="Hello, I am the user message.")
system_message = SystemChatMessage(content="System status: All systems go.")
```

# Create a conversation thread
```python
convo_thread = ConversationThread(system_message=system_message)
convo_thread.append(ChatExchange(prompt=user_message, response=system_message.to_chatmessage()))
```

# Serialize a conversation thread to JSON
```python
json_string = convo_thread.serialize()
```

# Deserialize a conversation thread from JSON
```python
convo_thread.deserialize(json_string)
```

# Installation
To install ChatAssistants, you can use the following pip command:
```python
pip install ChatAssistants
```

# Testing
The module comes with a suite of tests that can be run to ensure functionality remains consistent throughout changes.

# Contributing
Contributions to ChatAssistants are welcome! Please feel free to submit pull requests.

# License
