import json
import logging
from ChatAssistants import (AbstractChatAdapter, ChatMessage, ChatMessages, 
                            SystemChatMessage, ChatExchange, ConversationThread)

class OpenAIAdapter(AbstractChatAdapter):
    def from_chatmessage(self, chatmessage: ChatMessage):
        return chatmessage.to_dict(include_id = False)
    
    def to_chatmessage(self, message_dict: dict) -> ChatMessage:
        role = message_dict.get('role')
        content = message_dict.get('content')
        error_strings = []

        if role is None:
            error_strings.append("The message dictionary must contain a 'role' key.")
        
        if content is None:
            error_strings.append("The message dictionary must contain a 'content' key.")
        
        if error_strings:
            raise KeyError("\n".join(error_strings))

        return ChatMessage(role = role, content = content)

    def from_systemchatmessage(self, systemchatmessage: SystemChatMessage):
        # This is the same as from_chatmessage, but we'll include it
        # for completeness.
        return self.from_chatmessage(systemchatmessage)
    
    def to_systemchatmessage(self, message_dict: dict) -> SystemChatMessage:
        return SystemChatMessage.from_chatmessage(self.to_chatmessage(message_dict))

    def from_chatmessages(self, chatmessages: ChatMessages):
        return [message.to_dict(include_id = False) for message in chatmessages]

    def to_chatmessages(self, messages_list: list) -> ChatMessages:
        return [self.to_chatmessage(message) for message in messages_list]

    def from_chatexchange(self, chat_exchange: ChatExchange):
        return [self.from_chatmessage(chat_exchange.prompt),
                self.from_chatmessage(chat_exchange.response)]

    def to_chatexchange(self, prompt_and_response: list) -> ChatExchange:
        pr_len = len(prompt_and_response)
        if pr_len != 2:
            raise ValueError(f"The list should contain a prompt and a response, but has length {pr_len}.")
        
        prompt = self.to_chatmessage(prompt_and_response[0])
        response = self.to_chatmessage(prompt_and_response[1])

        return ChatExchange(prompt = prompt, response = response)
    
    def from_conversationthread(self, conversation_thread: ConversationThread):
        chatmessages_list = [conversation_thread.system_message]
        for chatexchange in conversation_thread.chat_exchanges:
            chatmessages_list.append(chatexchange.prompt)
            chatmessages_list.append(chatexchange.response)
            
        return [self.from_chatmessage(message) for message in chatmessages_list]
    
    def to_conversationthread(self, list_of_dicts: list) -> ConversationThread:
        convo_thread_len = len(list_of_dicts)

        if convo_thread_len < 3:
            raise ValueError(f"The list should contain an odd number of at least "
                             "3 messages, but it only has length {convo_thread_len}.")

        if convo_thread_len % 2 == 0:
            raise ValueError(f"The list should contain an odd number of at least "
                             "3 messages, but it has length {convo_thread_len}, "
                             "which is even.")

        system_chatmessage = SystemChatMessage.from_chatmessage(self.to_chatmessage(list_of_dicts[0]))

        chat_exchanges = []
        for prompt, response in zip(list_of_dicts[1::2], list_of_dicts[2::2]):
            chat_exchanges.append(self.to_chatexchange([prompt, response]))

        return ConversationThread(system_message = system_chatmessage,
                                  chat_exchanges = chat_exchanges)    
    