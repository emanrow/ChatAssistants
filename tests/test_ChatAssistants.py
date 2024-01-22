import unittest
from ChatAssistants import ChatMessages, ChatMessage, ChatExchange, SystemChatMessage

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

    # Additional tests can be added here...