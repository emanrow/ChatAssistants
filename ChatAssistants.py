from __future__ import annotations
import uuid
import json
import logging
from abc import ABC, abstractmethod
import enum
# import asyncio
import copy
from time import sleep, time


class ExcessTokenError(Exception):
    """Raise when the LLM callback returns an error indicating that
    the token limit has been exceeded. This will prevent successive
    futile attempts to complete API calls."""
    pass

class RunStatus(enum.Enum):
    UNSUBMITTED = 0
    PENDING = 1
    SUBMITTED = 2
    QUEUED = 3
    COMPLETED = 4
    ERROR = 5
    FAILED = 6

class ChatMessage:
    """A ChatMessage is akin to an OpenAI.client.beta.threads.message object.
    The object is, fundamentally, a dictionary with a unique ID, a role, and a content.
    
    This is formalized into a class to standardize inputs to the LLM through REST APIs.
    
    "role" can be either "user", "assistant", or "system".

    "content" can be any text string, inclusing JSON or other code snippets.
    TODO: Content should be .venvstored as a dict with a "type" key and a subordinate 
    "text" or "image_url" key.
    """
    VALID_ROLES = ["user", "assistant", "system"]
    
    def __init__(self, role, content):
        self.id = str(uuid.uuid4())
        self._role = None   # To make sure it exists
        self.role = role    # To run the setter
        self.content = content

    @property
    def role(self):
        return self._role
    
    @role.setter
    def role(self, new_role: str):
        logging.debug("Running ChatMessage.role.setter validation.")
        if new_role not in self.VALID_ROLES:
            raise ValueError(f"Invalid role: {new_role}. Role must be one of {self.VALID_ROLES}.")
        
        self._role = new_role

    def __str__(self):
        max_valid_user = max([len(role) for role in self.VALID_ROLES])
        return f"{self.role}:{' '*(max_valid_user+2-len(self.role))}{self.content}"
    
    def __repr__(self):
        content_str = self.content[:40]+"..." if len(self.content) > 44 else self.content
        return f"ChatMessage(id={self.id!r}, role = {self.role!r}, content = {content_str!r} )"
    
    def __json__(self):
        """Overrides default behavior of json.dumps() to serialize the object."""
        return json.dumps({"role": self.role, "content": self.content})
  
    def to_dict(self, include_id: bool = True) -> dict:
        if not include_id:
            return {"role": self.role, "content": self.content}
        
        return {"id": self.id, "role": self.role, "content": self.content}
    
    def update(self, role: str, content: str) -> None:
        self.role = role
        self.content = content
        return None
    
class ChatMessages:
    """ChatMessages is akin to the OpenAI.client.beta.threads.messages namespace.
    The object is, fundamentally, a list of ChatMessage objects.
    
    This is formalized into a class to standardize inputs to the LLM through REST APIs.

    The object manages the creation, removal and retrieval of ChatMessage objects, as well
    as their conversion to dicts and serialization and deserialization to and from JSON strings.
    """
    def __init__(self):
        self._messages = []

    def __iter__(self):
        """This method allows iteration over the ChatMessages object."""
        return iter(self._messages)

    def create(self, role: str, content: str) -> ChatMessage:
        new_message = ChatMessage(role, content)
        self._messages.append(new_message)
        return new_message
    
    def add(self, chat_message: ChatMessage) -> None:
        self._messages.append(chat_message)
        return None

    def remove(self, chat_message: ChatMessage) -> None:
        if chat_message in self._messages:
            self._messages.remove(chat_message)
            # Additional deletion logic if required
        else:
            raise ValueError("Message not found.")
        return chat_message
    
    def get(self, chat_message_id: str) -> ChatMessage:
        for message in self._messages:
            if message.id == chat_message_id:
                return message
        raise ValueError("Message not found.")
    
    def list(self) -> list:
        return self._messages
    
    def to_dict(self, include_id = True) -> dict:
        return [message.to_dict(include_id) for message in self._messages]

    def serialize(self) -> str:
        return json.dumps([message.to_dict() for message in self._messages])
    
    def deserialize(self, json_string: str) -> None:
        try:
            messages = json.loads(json_string)
            self._messages = [ChatMessage(**msg) for msg in messages]
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON string.")
        
class ChatExchange:
    """This is an intermediate helper object to manage a single pair of prompt and response
    ChatMessage objects."""
    def __init__(self, prompt: ChatMessage, response: ChatMessage):
        self.prompt = prompt
        self.response = response

    @property
    def prompt(self):
        return self._prompt
    
    @prompt.setter
    def prompt(self, new_prompt: ChatMessage):
        if new_prompt.role != "user":
            logging.error(f"Prompt must be a user message. Role was {new_prompt.role}.")
            raise ValueError("Prompt must be a user message.")
        self._prompt = new_prompt

    @property
    def response(self):
        return self._response
    
    @response.setter
    def response(self, new_response: ChatMessage):
        if new_response.role != "assistant":
            logging.error(f"Response must be an assistant message. Role was {new_response.role}.")
            raise ValueError("Response must be an assistant message.")
        self._response = new_response

    def __str__(self):
        return f"{self.prompt}\n{self.response}"
    
    def __repr__(self):
        return f"ChatExchange(prompt = {self.prompt!r}, response = {self.response!r})"
    
    def to_dict(self, include_id = True) -> dict:
        return {"prompt": self.prompt.to_dict(include_id), "response": self.response.to_dict(include_id)}
    
class SystemChatMessage(ChatMessage):
    """This is an intermediate helper object to manage a single system message.
    The object is, fundamentally, a ChatMessage object with a role of "system".
    
    This is formalized into a class to standardize inputs to the LLM through REST APIs.
    
    The object also adds the ability to convert to and from a standard ChatMessage object
    using the `from_chatmessage` and `to_chatmessage` methods."""
    def __init__(self, content: str):
        super().__init__(role="system", content=content)

    @ChatMessage.role.setter
    def role(self, new_role: str):
        logging.debug(f"Running SystemChatMessage.role.setter validation.")
        if new_role != "system":
            raise ValueError("Role of SystemChatMessage must be 'system'")
        # Passing type(self) as the second argument to super makes sense if you 
        # want to use super to look up a class attribute in an instance method, 
        # which is what we're doing here:
        super(SystemChatMessage, type(self)).role.fset(self, new_role)
        # self._role = new_role

    def __repr__(self) -> str:
        content_str = self.content[:34]+"..." if len(self.content) > 38 else self.content
        return (f"SystemChatMessage(id={self.id!r}, "
                f"role = {self.role!r}, content = {content_str!r} )")

    @classmethod
    def from_chatmessage(cls, chat_message: ChatMessage):
        if chat_message.role != "system":
            raise ValueError("ChatMessage should have a role of 'system' "
                             "to be converted to a SystemChatMessage.")
        return cls(content = chat_message.content)
    
    def to_chatmessage(self):
        return ChatMessage(role = "system", content = self.content)

class ConversationRun:
    """This is a state object for a submission to an LLM. It contains a 
    queryable unique run ID, run diagnostics, status, and
    the response from the LLM."""
    def __init__(self, max_attempts = 3, timeout = 60, adapter = None):
        self.id = str(uuid.uuid4())
        self.creation_time = time()
        self.submission_time = None
        self.completion_time = None
        self.duration = None
        self.attempts = 0
        self.max_attempts = max_attempts
        self.timeout = timeout
        self.status = RunStatus.UNSUBMITTED
        self.conversation: Conversation = None
        self.convo_snapshot: Conversation = None
        self.submission_list: list = None
        self.llm_callback = None
        self.cb_args = None
        self.cb_kwargs = None
        self.error_list = []
        self.raw_response: dict = None
        self.adapter: AbstractChatAdapter = adapter
        self.response: ChatMessage = None
        self._task = None

    def adapt_submission(self, tr: Conversation):
        if self.adapter is None:
            logging.warning("No adapter set in ConversationRun object. "
                            "Setting `submission_list` equal to `raw_submission_list`.")
            return None
        
        if self.conversation is None:
            logging.error("Cannot adapt submission: raw_submission_list is None.")
            return None

        try:
            self.submission_list = self.adapter.from_conversation(self.conversation)
        except Exception as e:
            logging.error(f"Error adapting submission using provided ChatAdapter: {e}")
            raise e

    def adapt_response(self):
        if self.raw_response is None:
            logging.error("Cannot adapt response: raw_response is None.")
            return None
        
        if self.adapter is None:
            logging.warning("No adapter set in ConversationRun object. "
                            "Setting `response` equal to `raw_response`.")
            self.response = self.raw_response
        
        try:
            self.response = self.adapter.to_chatmessage(self.raw_response)
        except Exception as e:
            logging.error(f"Error adapting response using provided ChatAdapter: {e}")
            raise e

    def __str__(self):
        return f"Run {self.id} status: {self.status}."
    
    def __repr__(self):
        return (f"ConversationRun(id = {self.id!r}, "
                f"status = {self.status!r}, "
                f"attempts = {self.attempts!r}, "
                f"max_attempts = {self.max_attempts!r}, "
                f"timeout = {self.timeout!r}, "
                f"raw_response = {self.raw_response!r}, "
                f"response = {self.response!r})")

class Conversation:
    """This is a payload object to manage a list of ChatExchange objects,
    prepended by a SystemChatMessage object. This can then be either appended
    with a single ChatMessage (prompt) object and passed to an LLM to obtain
    a response (completing the next ChatExchange), updated with the next ChatExchange,
    or serialized to a JSON string for storage or transmission."""
    def __init__(self, system_message: SystemChatMessage, chat_exchanges: list = None,
                 next_prompt: ChatMessage = None):
        self.system_message = system_message
        self.chat_exchanges = chat_exchanges
        self.next_prompt = next_prompt

    @property
    def system_message(self):
        return self._system_message
    
    @system_message.setter
    def system_message(self, new_system_message: SystemChatMessage):
        self._system_message = new_system_message

    @property
    def chat_exchanges(self):
        return self._chat_exchanges
    
    @chat_exchanges.setter
    def chat_exchanges(self, new_chat_exchanges: list):
        if new_chat_exchanges is None:
            self._chat_exchanges = []
            return None
        
        for chat_exchange in new_chat_exchanges:
            if not isinstance(chat_exchange, ChatExchange):
                raise ValueError("chat_exchanges must be a list of ChatExchange objects.")
        self._chat_exchanges = new_chat_exchanges

    @property
    def next_prompt(self):
        return self._next_prompt
    
    @next_prompt.setter
    def next_prompt(self, new_next_prompt: ChatMessage):
        if new_next_prompt is not None and new_next_prompt.role != "user":
            raise ValueError("next_prompt must be a user message.")
        self._next_prompt = new_next_prompt

    def run(self, max_attempts = 3, timeout = 60, adapter: AbstractChatAdapter = None, 
            *cb_args, **cb_kwargs) -> ConversationRun:
        """This method runs the Conversation through the LLM, obtains
        the response to complete the next ChatExchange, and appends the
        new ChatExchange to the list of ChatExchanges.
        
        The LLM callback function should be implemented to take whatever
        arguments """
        if self.next_prompt is None:
            raise ValueError("next_prompt must be set before running the Conversation.")

        # Packaging everything in a stateful ConversationRun object        
        _run_object = ConversationRun(max_attempts = max_attempts, 
                                            timeout = timeout)
        _run_object.cb_args = cb_args
        _run_object.cb_kwargs = cb_kwargs
        _run_object.adapter = adapter
        _run_object.conversation = self
        # _run_object.convo_snapshot = copy.deepcopy(self)

        # Broad strokes:
        # TODO: Refactor this all so that it is self-documenting
        # I.   Adapt the _submission_list to the LLM input format
        # II.  Submit the _submission_list to the LLM via the handler, which 
        #      returns nothing, immediately without waiting for the response.
        # III. Return the run object with the response and status set

        # Then, in the handler:
        # II(a). Adapt the LLM response to a ChatMessage object
        # II(b). Update the Conversation with the new ChatExchange
        # II(c). Update the run object with the response and status

        # I.   Adapt the _submission_list to the LLM input format
        # This isn't right... should be using the appropriate adapter to get the
        # correct format for the LLM input.
        _run_object.submission_list = _run_object.adapter.from_conversation(self)

        # II.  Submit the _submission_list to the LLM via the handler
        _run_object.status = RunStatus.PENDING
        # This isn't running because it's not awaited
        _run_object._task = self._handle_submission(_run_object)

        # III. Return the run object with the response and status set
        return _run_object

    def _handle_submission(self, ro: ConversationRun):
        # This is the asynchronous handler for the LLM submission.
        # Calling the run_oject `ro` just to save space
        _delay_time = 3

        # II(a). Adapt the LLM response to a ChatMessage object

        while ro.attempts < ro.max_attempts:
            ro.submission_time = time()
            ro.attempts += 1
            ro.status = RunStatus.SUBMITTED
            try:
                ro.raw_response = ro.adapter.llm_callback(self,
                                                          *ro.cb_args,
                                                          **ro.cb_kwargs)
            except ExcessTokenError as token_e:
                ro.status = RunStatus.FAILED
                logging.error(f"ExcessTokenError in LLM callback: {token_e}")
                ro.error_list.append(token_e)
                raise ExcessTokenError(token_e)
                break
            except Exception as e:
                ro.status = RunStatus.ERROR
                logging.error(f"Error in LLM callback attempt #{ro.attempts}: {e}")
                ro.error_list.append(e)
                if ro.attempts >= ro.max_attempts:
                    ro.status = RunStatus.FAILED
                    return ro
                sleep(_delay_time)
                pass
            else:
                # Submission was successful: Snapshot the conversation and return
                ro.convo_snapshot = copy.deepcopy(ro.conversation)
                # II(b). Update the Conversation with the new ChatExchange
                ro.adapt_response()
                # TODO: This needs better validation
                _new_exchange = ChatExchange(prompt = self.next_prompt, 
                                             response = ro.response)
                self.chat_exchanges.append(_new_exchange)

                # II(c). Update the run object with the response and status
                # TODO: This needs better validation
                ro.status = RunStatus.COMPLETED
                ro.completion_time = time()
                ro.duration = ro.completion_time - ro.creation_time
                return ro

    def append(self, chat_exchange: ChatExchange) -> None:
        if not isinstance(chat_exchange, ChatExchange):
            raise ValueError("chat_exchange must be a ChatExchange object.")
        self._chat_exchanges.append(chat_exchange)
        return None

    def to_dict(self, include_id = True) -> dict:
        return {"system_message": self.system_message.to_dict(include_id),
                "chat_exchanges": [exchange.to_dict(include_id) for exchange in self.chat_exchanges]}

    def serialize(self) -> str:
        return json.dumps(self.to_dict())

    def deserialize(self, json_string: str) -> None:
        try:
            convo = json.loads(json_string)
            self.system_message = SystemChatMessage(**convo["system_message"])
            self.chat_exchanges = [ChatExchange(**exchange) for exchange in convo["chat_exchanges"]]
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON string.")

    def __str__(self):
        return f"{self.system_message}\n" + "\n".join([str(exchange) for exchange in self.chat_exchanges])

    def __repr__(self):
        return f"Conversation(system_message = {self.system_message!r}, chat_exchanges = {self.chat_exchanges!r})"
    
class AbstractChatAdapter(ABC):
    @abstractmethod
    def from_chatmessage(self, chatmessage: ChatMessage):
        pass

    @abstractmethod
    def to_chatmessage(self) -> ChatMessage:
        pass

    @abstractmethod
    def from_systemchatmessage(self, systemchatmessage: SystemChatMessage):
        pass

    @abstractmethod
    def to_systemchatmessage(self) -> SystemChatMessage:
        pass

    @abstractmethod
    def from_chatmessages(self, chatmessages: ChatMessages):
        pass

    @abstractmethod
    def to_chatmessages(self) -> ChatMessages:
        pass

    @abstractmethod
    def from_chatexchange(self, chatexchange: ChatExchange):
        pass

    @abstractmethod
    def to_chatexchange(self) -> ChatExchange:
        pass

    @abstractmethod
    def from_conversation(self, conversation: Conversation):
        pass

    @abstractmethod
    def to_conversation(self) -> Conversation:
        pass

    @abstractmethod
    def llm_callback(self, conversation: Conversation, 
                     *args, **kwargs) -> dict:
        """
        This method should handle the communication with the LLM, process the response,
        and return a value that can be adapted to a ChatMessage object.
        
        A sane way to do this would be to design llm_callback to use the adapter method
        from_conversation to convert the Conversation to the LLM input format,
        and to return the LLM response as a dict that can be adapted to a ChatMessage
        object with to_chatmessage. But you do you.
        """
        pass


