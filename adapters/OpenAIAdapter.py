import json
import logging
from ChatAssistants import (AbstractChatAdapter, ChatMessage, ChatMessages, 
                            SystemChatMessage, ChatExchange, ConversationThread)
import asyncio
from openai import OpenAI

openai_client = OpenAI()

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
    
    async def llm_callback(self, conversationthread: ConversationThread,
                     *cb_args, **cb_kwargs) -> dict:
        """
        This is the callback function that actually uses the LLM API to obtain
        a response.
        """
        model = cb_kwargs.get('model', 'gpt-3.5-turbo')
        frequency_penalty = cb_kwargs.get('frequency_penalty', 0.0)
        presence_penalty = cb_kwargs.get('presence_penalty', 0.0)
        temperature = cb_kwargs.get('temperature', 1.0)
        top_p = cb_kwargs.get('top_p', 1.0)
        max_tokens = cb_kwargs.get('max_tokens', 2048)
        response_format = cb_kwargs.get('response_format', None)
        openai_client.api_key = cb_kwargs.get('OPENAI_API_KEY', None)
        
        _response = await openai_client.chat.completions.create(
            model=model,
            response_format=response_format,
            temperature=1,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            max_tokens=max_tokens,
            messages=self.from_conversationthread(conversationthread)
        )

        return _response.choices[0].message.content
        # await asyncio.sleep(1)
        # return {'role': 'assistant', 'content': 'This is a placeholder response.'}
    
