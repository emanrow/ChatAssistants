import unittest
from ChatAssistants import (ChatMessage, SystemChatMessage, ChatMessages,
                            ChatExchange, Conversation)
from adapters.OpenAIAdapter import OpenAIAdapter

class TestOpenAIAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = OpenAIAdapter()

    def test_from_chatmessage(self):
        # Test converting from ChatMessage to dictionary
        chat_message = ChatMessage(role="user", content="Hello, world!")
        result = self.adapter.from_chatmessage(chat_message)
        expected = {'role': 'user', 'content': 'Hello, world!'}
        self.assertEqual(result, expected)

    def test_to_chatmessage(self):
        # Test converting from dictionary to ChatMessage
        message_dict = {'role': 'user', 'content': 'Hello, world!'}
        result = self.adapter.to_chatmessage(message_dict)
        self.assertIsInstance(result, ChatMessage)
        self.assertEqual(result.role, 'user')
        self.assertEqual(result.content, 'Hello, world!')

        # Test with missing 'role' key
        with self.assertRaises(KeyError):
            self.adapter.to_chatmessage({'content': 'Missing role'})

        # Test with missing 'content' key
        with self.assertRaises(KeyError):
            self.adapter.to_chatmessage({'role': 'user'})

        # Test missing both 'role' and 'content' keys
        with self.assertRaises(KeyError):
            self.adapter.to_chatmessage({'lunch': 'tacos'})

    def test_from_systemchatmessage(self):
        # Test converting from SystemChatMessage to dictionary
        system_chat_message = SystemChatMessage(content="System message content")
        result = self.adapter.from_systemchatmessage(system_chat_message)
        expected = {'role': 'system', 'content': 'System message content'}
        self.assertEqual(result, expected)

    def test_to_systemchatmessage(self):
        # Test converting from dictionary to SystemChatMessage
        message_dict = {'role': 'system', 'content': 'System message content'}
        result = self.adapter.to_systemchatmessage(message_dict)
        self.assertIsInstance(result, SystemChatMessage)
        self.assertEqual(result.role, 'system')
        self.assertEqual(result.content, 'System message content')

        # Test with missing keys
        with self.assertRaises(KeyError):
            self.adapter.to_systemchatmessage({'content': 'Missing role'})
        with self.assertRaises(KeyError):
            self.adapter.to_systemchatmessage({'role': 'system'})

    def test_from_chatmessages(self):
        # Test converting from ChatMessages to list of dictionaries
        chat_messages = ChatMessages()
        chat_messages.add(ChatMessage(role="user", content="User message"))
        chat_messages.add(ChatMessage(role="assistant", content="Assistant message"))

        result = self.adapter.from_chatmessages(chat_messages)
        expected = [
            {'role': 'user', 'content': 'User message'},
            {'role': 'assistant', 'content': 'Assistant message'}
        ]
        self.assertEqual(result, expected)

    def test_to_chatmessages(self):
        # Test converting from list of dictionaries to ChatMessages
        messages_list = [
            {'role': 'user', 'content': 'User message'},
            {'role': 'assistant', 'content': 'Assistant message'}
        ]

        result = self.adapter.to_chatmessages(messages_list)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], ChatMessage)
        self.assertIsInstance(result[1], ChatMessage)
        self.assertEqual(result[0].role, 'user')
        self.assertEqual(result[1].role, 'assistant')

    def test_from_chatexchange(self):
        # Test converting from ChatExchange to list of dictionaries
        prompt = ChatMessage(role="user", content="Hello, world!")
        response = ChatMessage(role="assistant", content="Hello, user!")
        chat_exchange = ChatExchange(prompt=prompt, response=response)

        result = self.adapter.from_chatexchange(chat_exchange)
        expected = [
            {'role': 'user', 'content': 'Hello, world!'},
            {'role': 'assistant', 'content': 'Hello, user!'}
        ]
        self.assertEqual(result, expected)

    def test_to_chatexchange(self):
        # Test converting from list of dictionaries to ChatExchange
        prompt_dict = {'role': 'user', 'content': 'Hello, world!'}
        response_dict = {'role': 'assistant', 'content': 'Hello, user!'}
        prompt_and_response = [prompt_dict, response_dict]

        result = self.adapter.to_chatexchange(prompt_and_response)
        self.assertIsInstance(result, ChatExchange)
        self.assertEqual(result.prompt.role, 'user')
        self.assertEqual(result.prompt.content, 'Hello, world!')
        self.assertEqual(result.response.role, 'assistant')
        self.assertEqual(result.response.content, 'Hello, user!')

        # Test with incorrect list length
        with self.assertRaises(ValueError):
            self.adapter.to_chatexchange([prompt_dict])

    def test_from_conversationthread(self):
        # Test converting from ConversationThread to list of dictionaries
        system_message = SystemChatMessage(content="System message content")
        prompt = ChatMessage(role="user", content="User message")
        response = ChatMessage(role="assistant", content="Assistant message")
        chat_exchange = ChatExchange(prompt=prompt, response=response)
        conversation_thread = Conversation(system_message=system_message, 
                                                 chat_exchanges=[chat_exchange])

        result = self.adapter.from_conversation(conversation_thread)
        expected = [
            {'role': 'system', 'content': 'System message content'},
            {'role': 'user', 'content': 'User message'},
            {'role': 'assistant', 'content': 'Assistant message'}
        ]
        self.assertEqual(result, expected)

    def test_to_conversationthread(self):
        # Test converting from list of dictionaries to ConversationThread
        messages_list = [
            {'role': 'system', 'content': 'System message content'},
            {'role': 'user', 'content': 'User message'},
            {'role': 'assistant', 'content': 'Assistant message'}
        ]

        result = self.adapter.to_conversation(messages_list)
        self.assertIsInstance(result, Conversation)
        self.assertIsInstance(result.system_message, SystemChatMessage)
        self.assertEqual(len(result.chat_exchanges), 1)
        self.assertEqual(result.system_message.content, 'System message content')
        self.assertEqual(result.chat_exchanges[0].prompt.role, 'user')
        self.assertEqual(result.chat_exchanges[0].response.role, 'assistant')

        # Test with incorrect list length
        with self.assertRaises(ValueError):
            self.adapter.to_conversation([{'role': 'system', 'content': 'System message content'}])
        with self.assertRaises(ValueError):
            self.adapter.to_conversation(messages_list + [{'role': 'user', 'content': 'Another user message'}])

# More test methods can be added here...

if __name__ == '__main__':
    unittest.main()
