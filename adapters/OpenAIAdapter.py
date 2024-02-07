import json
import logging
from ChatAssistants import (AbstractChatAdapter, ChatMessage, ChatMessages, 
                            SystemChatMessage, ChatExchange, Conversation,
                            ExcessTokenError)
# import asyncio
from openai import OpenAI
import tiktoken
from enum import StrEnum
import logging

openai_client = OpenAI()

class modelstr(StrEnum):
    GPT4 =          "gpt-4"
    GPT4VISION =    "gpt-4-vision-preview"
    GPT4PREV =      "gpt-4-1106-preview"
    GPT4TURBOPREV = "gpt-4-turbo-preview"
    GPT35TURBO =    "gpt-3.5-turbo-1106"
    GPT35TURBO16 =  "gpt-3.5-turbo-16k"
    DAVINCI =       "text-davinci-003"
    ADAEMBED =      "text-embedding-ada-002"
    ADA =           "text-ada-001"

class embedstr(StrEnum):
    CL100K = "cl100k_base"
    P50K =   "p50k_base"

model_to_encode = {modelstr.GPT4:          tiktoken.registry.get_encoding(embedstr.CL100K),
                   modelstr.GPT4VISION:    tiktoken.get_encoding(embedstr.CL100K),
                   modelstr.GPT4PREV:      tiktoken.get_encoding(embedstr.CL100K),
                   modelstr.GPT4TURBOPREV: tiktoken.get_encoding(embedstr.CL100K),
                   modelstr.GPT35TURBO:    tiktoken.get_encoding(embedstr.CL100K),
                   modelstr.GPT35TURBO16:  tiktoken.get_encoding(embedstr.CL100K),
                   modelstr.DAVINCI:       tiktoken.get_encoding(embedstr.P50K),
                   modelstr.ADAEMBED:      tiktoken.get_encoding(embedstr.CL100K),
                   modelstr.ADA:           tiktoken.get_encoding(embedstr.P50K)}

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
    
    def from_conversation(self, conversation: Conversation):
        chatmessages_list = [conversation.system_message]

        for chatexchange in conversation.chat_exchanges:
            chatmessages_list.append(chatexchange.prompt)
            chatmessages_list.append(chatexchange.response)

        if conversation.next_prompt is not None:
            chatmessages_list.append(conversation.next_prompt)
            
        return [self.from_chatmessage(message) for message in chatmessages_list]
    
    def to_conversation(self, list_of_dicts: list) -> Conversation:
        convo_len = len(list_of_dicts)

        if convo_len < 3:
            raise ValueError(f"The list should contain an odd number of at least "
                             "3 messages, but it only has length {convo_len}.")

        if convo_len % 2 == 0:
            raise ValueError(f"The list should contain an odd number of at least "
                             "3 messages, but it has length {convo_len}, "
                             "which is even.")
        
        first_message = self.to_chatmessage(list_of_dicts[0])
        if first_message.role != "system":
            raise ValueError(f"The first message should be a system message, but "
                             f"it's a {first_message.role} message.")
        
        system_chatmessage = SystemChatMessage.from_chatmessage(self.to_chatmessage(list_of_dicts[0]))

        if convo_len % 2 == 0:
            # There is an even number of messages, which means there is an
            # odd number excluding the system message. All but the last one
            # should be parts of prior exchanges.
            chat_exchange_list = list_of_dicts[1:-1]
            # The last message should be the next prompt.
            next_prompt = self.to_chatmessage(list_of_dicts[-1])
        else:
            # There is an odd number of messages, which means there is an
            # even number excluding the system message. All of them
            # should be parts of prior exchanges.
            chat_exchange_list = list_of_dicts[1:]
            next_prompt = None

        chat_exchanges = []
        for prompt, response in zip(chat_exchange_list[0::2], chat_exchange_list[1::2]):
            chat_exchanges.append(self.to_chatexchange([prompt, response]))

        if next_prompt is not None and next_prompt.role != "user":
            raise ValueError(f"The last message should be a user message, but "
                             f"it's a {next_prompt.role} message.")

        return Conversation(system_message = system_chatmessage,
                                  chat_exchanges = chat_exchanges,
                                  next_prompt = next_prompt)    

    def llm_callback(self, conversation: Conversation,
                     *cb_args, **cb_kwargs) -> dict:
        """
        This is the callback function that actually uses the LLM API to obtain
        a response.
        """
        model = cb_kwargs.get('model', modelstr.GPT35TURBO)
        frequency_penalty = cb_kwargs.get('frequency_penalty', 0.0)
        presence_penalty = cb_kwargs.get('presence_penalty', 0.0)
        temperature = cb_kwargs.get('temperature', 1.0)
        top_p = cb_kwargs.get('top_p', 1.0)
        max_prompt_tokens = cb_kwargs.get('max_prompt_tokens', 2048)
        max_response_tokens = cb_kwargs.get('max_response_tokens', 4096)
        response_format = cb_kwargs.get('response_format', None)
        image_b64 = cb_kwargs.get('image_b64', None)
        openai_client.api_key = cb_kwargs.get('OPENAI_API_KEY', None)
        
        # Make sure messages isn't more tokens than max_tokens
        messages = self.from_conversation(conversation)
        logging.info(f"messages: {messages}")
        messages_str=json.dumps(messages)
        
        tt_encoder = model_to_encode[model]
        submission_tokens = len(tt_encoder.encode(messages_str))
        logging.info(f"submission_tokens: {submission_tokens}")
        if submission_tokens > max_prompt_tokens:
            raise ExcessTokenError(f"Submission tokens ({submission_tokens}) is greater than max_tokens ({max_prompt_tokens}).")

        completions_kwargs = {"model": model,
                              "temperature": temperature,
                              "top_p": top_p,
                              "frequency_penalty": frequency_penalty,
                              "presence_penalty": presence_penalty,
                              "max_tokens": max_response_tokens}

        if model == modelstr.GPT4VISION:
            _image_url = {"url": f"data:image/jpeg;base64,{image_b64}"}
            messages[-1]["content"] = [{"type":"text","text":f"{messages[-1]['content']}"},
                                       {"type":"image","image_url":_image_url}]
        elif model == modelstr.GPT35TURBO or model == modelstr.GPT4TURBOPREV:
            completions_kwargs["response_format"] = response_format

        completions_kwargs["messages"] = messages
        
        _response = openai_client.chat.completions.create(**completions_kwargs)

        _actual_submission_tokens = _response.usage.prompt_tokens
        logging.info(f"actual_submission_tokens: {_actual_submission_tokens}")
        if _actual_submission_tokens != submission_tokens:
            logging.warning(f"Actual submission tokens ({_actual_submission_tokens}) "
                            f"is not equal to calculated submission tokens "
                            f"({submission_tokens}).")

        _response_role = _response.choices[0].message.role
        _response_content = _response.choices[0].message.content

        return {"role": _response_role, "content": _response_content, "raw_response": _response}
    
