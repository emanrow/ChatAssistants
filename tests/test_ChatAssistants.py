import unittest
from ChatAssistants import (ChatMessages, ChatMessage, ChatExchange, SystemChatMessage,
                           Conversation, AbstractChatAdapter)

class TestChatMessages(unittest.TestCase):

    def setUp(self):
        # This method will be called before each test
        self.this_convo = ChatMessages()
        self.system_message = self.this_convo.create("system", "Hello, I am the system message.")
        self.user_message = self.this_convo.create("user", "Hello, I am the user message.")
        self.assistant_message = ChatMessage("assistant", "Hello, I am the assistant response.")
        self.this_convo.add(self.assistant_message)
        self.convo_list = self.this_convo.list()

    def test_message_creation(self):
        # Test if messages are created correctly
        self.assertEqual(self.system_message.role, "system")
        self.assertEqual(self.system_message.content, "Hello, I am the system message.")
        self.assertEqual(self.user_message.role, "user")
        self.assertEqual(self.assistant_message.role, "assistant")

    def test_message_list(self):
        # Test if the message list contains the correct messages
        self.assertEqual(len(self.convo_list), 3)  # Assuming 3 messages were added
        self.assertIn(self.system_message, self.convo_list)
        self.assertIn(self.user_message, self.convo_list)
        self.assertIn(self.assistant_message, self.convo_list)

    def test_conversion_to_system_message(self):
        # Test conversion of a ChatMessage to a SystemChatMessage
        for message in self.convo_list:
            if message.role == "system":
                system_message = SystemChatMessage.from_chatmessage(message)
                self.assertIsInstance(system_message, SystemChatMessage)
                self.assertEqual(system_message.content, message.content)

    def test_chatexchange_creation(self):
        # Test if a ChatExchange is created correctly
        chatexchange = ChatExchange(prompt = self.user_message, response = self.assistant_message)
        self.assertIsInstance(chatexchange, ChatExchange)
        self.assertEqual(chatexchange.prompt, self.user_message)
        self.assertEqual(chatexchange.response, self.assistant_message)

    def test_conversation_creation(self):
        # Test if a Conversation is created correctly
        convo = Conversation(system_message = self.system_message,
                             chat_exchanges = [ChatExchange(prompt = self.user_message,
                                                            response = self.assistant_message),
                                               ChatExchange(prompt = self.user_message,
                                                            response = self.assistant_message),
                                               ChatExchange(prompt = self.user_message,
                                                            response = self.assistant_message)])
        self.assertIsInstance(convo, Conversation)
        self.assertEqual(convo.system_message, self.system_message)
        self.assertEqual(convo.chat_exchanges[0].prompt, self.user_message)
        self.assertEqual(convo.chat_exchanges[2].response, self.assistant_message)

    def test_conversation_run(self):
        # Test if a Conversation runs correctly
        convo = Conversation(system_message = self.system_message,
                             chat_exchanges = [ChatExchange(prompt = self.user_message,
                                                            response = self.assistant_message),
                                               ChatExchange(prompt = self.user_message,
                                                            response = self.assistant_message),
                                               ChatExchange(prompt = self.user_message,
                                                            response = self.assistant_message)])
        convo.next_prompt = self.user_message
        class mock_adapter(AbstractChatAdapter):
            def __init__(self):
                pass

            def from_conversation(self, conversation: Conversation):
                return [{"role": "system", "content": "This is a mock system message."},
                        {"role": "user", "content": "This is a mock user message."},
                        {"role": "assistant", "content": "This is a mock assistant response."}]
            
            def to_chatmessage(self, message_dict: dict) -> ChatMessage:
                return ChatMessage(role = message_dict["role"], content = message_dict["content"])
            
            def llm_callback(self, conversation: Conversation, *args, **kwargs):
                return {"role": "assistant", "content": "This is a mock assistant response."}

        convo.run(adapter = mock_adapter)
        self.assertEqual(len(convo.chat_exchanges), 3)
        self.assertEqual(convo.chat_exchanges[0].prompt, self.user_message)
        self.assertEqual(convo.chat_exchanges[2].response, self.assistant_message)
        self.assertEqual(convo.chat_exchanges[3].prompt.content, "Hello, I am the user message.")
        self.assertEqual(convo.chat_exchanges[3].response.content, "This is a mock assistant response.")