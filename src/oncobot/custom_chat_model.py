import time
from typing import Dict, AsyncGenerator, Generator, Optional, Union, List
from jinja2.exceptions import TemplateError
from abc import ABC, abstractmethod

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)
from huggingface_hub import login
from threading import Thread
import torch
import openai
import replicate
import groq

from src.utils.logger_config import logger
from src.utils.env_config import settings
from .chat_templates import CHAT_TEMPLATES


# Base class for all custom chat models. All custom chat models should inherit from this class
class BaseChat(ABC):
    def __init__(self):
        logger.info("BaseChat initialized.")

    @abstractmethod
    def invoke(self, current_conversation: List[Dict[str, str]]) -> str:
        pass

    @abstractmethod
    def stream(
        self, current_conversation: List[Dict[str, str]]
    ) -> Generator[str, None, None]:
        pass

    @abstractmethod
    async def ainvoke(self, current_conversation: List[Dict[str, str]]) -> str:
        pass

    @abstractmethod
    async def astream(
        self, current_conversation: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        pass


class DummyChat(BaseChat):
    def __init__(
        self,
        default_message: str = "This is a dummy chat message.",
    ):
        logger.info("DummyChat initialized.")
        self.default_message = default_message

    def invoke(self, current_conversation: List[Dict[str, str]]) -> str:
        return "DummyChat invoke: " + self.default_message

    def stream(
        self, current_conversation: List[Dict[str, str]]
    ) -> Generator[str, None, None]:
        yield "DummyChat stream: " + self.default_message

    async def ainvoke(
        self, current_conversation: List[Dict[str, str]], sleep_time=3
    ) -> str:
        time.sleep(sleep_time)
        return "DummyChat ainvoke: " + self.default_message

    async def astream(
        self,
        current_conversation: List[Dict[str, str]],
        num_repeats=3,
        stream: bool = True,
    ) -> AsyncGenerator[str, None]:
        import asyncio

        async def async_generator():
            await asyncio.sleep(1)
            if stream:
                for _ in range(num_repeats):
                    await asyncio.sleep(0.01)
                    yield "DummyChat astream: " + self.default_message
            else:
                yield ("DummyChat astream: " + self.default_message) * num_repeats

        return async_generator()


# The async methods in this class are just to have the same async syntax as the other remotely called custom chat models. Not actually async.
class CustomChatHuggingFace(BaseChat):
    default_generation_kwargs = {
        # "truncation": True,
        # "max_length": 512,
        "max_new_tokens": 128,
        "num_return_sequences": 1,
        "do_sample": True,
        "temperature": 0.1,
        "top_k": 50,
        "top_p": 0.95,
    }

    def __init__(
        self,
        checkpoint: str = "meta-llama/Llama-2-7b-chat-hf",
        device: Optional[str] = None,
        model=None,
        tokenizer=None,
        max_chat_length: int = 2048,
        generation_kwargs: Optional[Dict[str, Union[int, bool, float]]] = None,
    ):
        self._huggingface_login()
        self.device = self._determine_device() if not device else device
        self.checkpoint = checkpoint
        try:
            self.tokenizer = (
                AutoTokenizer.from_pretrained(self.checkpoint)
                if not tokenizer
                else tokenizer
            )
            self.tokenizer.chat_template = CHAT_TEMPLATES.get(
                self.checkpoint, self.tokenizer.default_chat_template
            )
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
        try:
            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    self.checkpoint,
                    device_map="auto" if self.device == "cuda" else self.device,
                )
                if not model
                else model
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        self.max_chat_length = max_chat_length  # NOTE: can modify this to be the context length of the model. Also depends on how much memory is available (GPU or CPU)
        logger.debug(f"Chat max length: {self.max_chat_length} tokens")
        if self.max_chat_length > self.model.config.max_position_embeddings:
            logger.warning(
                f"Max chat length is greater than model's max position embeddings: {self.model.config.max_position_embeddings}"
            )

        self.generation_kwargs = generation_kwargs or self.default_generation_kwargs
        if self.generation_kwargs.get("temperature") == 0:
            self.generation_kwargs["do_sample"] = False

        # add end-of-sequence tokens for Llama 3 Instruct models
        if (
            self.checkpoint == "meta-llama/Meta-Llama-3-8B-Instruct"
            or self.checkpoint == "meta-llama/Meta-Llama-3-70B-Instruct"
        ):
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ]
            self.generation_kwargs["eos_token_id"] = terminators  # type: ignore

        logger.info(f"{checkpoint} initialized on device {self.device}.")
        if self.device == "cuda":
            logger.debug("Number of GPUs: {}".format(torch.cuda.device_count()))
            logger.debug(
                "Memory footprint: {:.2f}GB".format(
                    self.model.get_memory_footprint() / 1024**3
                )
            )

    def _huggingface_login(self):
        token = settings.hf_token.get_secret_value()
        login(token=token)

    def _determine_device(self):
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def invoke(self, current_conversation: List[Dict[str, str]]) -> str:
        try:
            chat_history = self.tokenizer.apply_chat_template(
                current_conversation,
                tokenize=False,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        except TemplateError as e:
            logger.error(f"Error applying chat template: {e}")
            raise

        tokenized_chat_history = self.tokenizer(
            text=chat_history,  # type: ignore
            return_tensors="pt",  # type: ignore
        )

        # truncate chat history to max_chat_length before moving to device
        tokenized_chat_history["input_ids"] = tokenized_chat_history["input_ids"][  # type: ignore
            :, -self.max_chat_length :
        ]
        tokenized_chat_history["attention_mask"] = tokenized_chat_history[  # type: ignore
            "attention_mask"
        ][:, -self.max_chat_length :]

        tokenized_chat_history = tokenized_chat_history.to(self.device)
        prompt_size = tokenized_chat_history["input_ids"].shape[1]  # type: ignore
        try:
            generated_tokens = self.model.generate(
                **tokenized_chat_history,
                **self.generation_kwargs,
            )
            # skip the prompt tokens and decode the rest
            response = self.tokenizer.decode(
                generated_tokens[:, prompt_size:][0], skip_special_tokens=True
            )
            return response
        except Exception as e:
            logger.error(
                f"Error generating tokens: {e}",
                exc_info=True,
                stack_info=True,
            )
            raise
        finally:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def stream(
        self, current_conversation: List[Dict[str, str]]
    ) -> Generator[str, None, None]:
        streamer = TextIteratorStreamer(
            self.tokenizer,  # type: ignore
            skip_prompt=True,
            skip_special_tokens=True,  # type: ignore
        )
        chat_history = self.tokenizer.apply_chat_template(
            current_conversation,
            tokenize=False,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        tokenized_chat_history = self.tokenizer(
            text=chat_history,  # type: ignore
            return_tensors="pt",  # type: ignore
        ).to(self.device)

        tokenized_chat_history["input_ids"] = tokenized_chat_history["input_ids"][  # type: ignore
            :, -self.max_chat_length :
        ]
        tokenized_chat_history["attention_mask"] = tokenized_chat_history[  # type: ignore
            "attention_mask"
        ][:, -self.max_chat_length :]

        # start the generation in a separate thread
        thread = Thread(
            target=self.model.generate,
            kwargs={
                **tokenized_chat_history,
                "streamer": streamer,
                **self.generation_kwargs,
            },
        )
        thread.start()

        def generator():
            try:
                for new_text in streamer:
                    yield new_text
            except Exception as e:
                logger.error(
                    f"Error in stream generator: {e}",
                    exc_info=True,
                    stack_info=True,
                )
                raise
            finally:
                thread.join()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        return generator()

    async def ainvoke(self, current_conversation: list[dict[str, str]]) -> str:
        try:
            return self.invoke(current_conversation)
        except Exception as e:
            logger.error(
                f"Error in ainvoke generator: {e}",
            )
            raise

    async def astream(
        self, current_conversation: list[dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        async def async_generator():
            try:
                for new_text in self.stream(current_conversation):
                    yield new_text
            except Exception as e:
                logger.error(
                    f"Error in astream generator: {e}",
                    exc_info=True,
                    stack_info=True,
                )
                raise

        return async_generator()


class CustomChatOpenAI(BaseChat):
    def __init__(self, model_name: str = "gpt-3.5-turbo", base_url: str | None = None):
        self.api_key = self._get_openai_api_key()
        self.client = openai.OpenAI(api_key=self.api_key, base_url=base_url)
        self.async_client = openai.AsyncOpenAI(api_key=self.api_key)
        self.model_name = model_name
        logger.info("CustomChatOpenAI initialized.")

    def _get_openai_api_key(self):
        openai_api_key = settings.openai_api_key.get_secret_value()
        return openai_api_key

    def _handle_api_error(self, e: Exception):
        """Centralized error handling for OpenAI API errors."""
        if isinstance(e, openai.APITimeoutError):
            logger.error("The request took too long: %s", e)
        elif isinstance(e, openai.APIConnectionError):
            logger.error(
                "Request could not reach the OpenAI API servers or establish a secure connection: %s",
                e,
            )
        elif isinstance(e, openai.AuthenticationError):
            logger.error("API key or token is invalid, expired, or revoked: %s", e)
        elif isinstance(e, openai.InternalServerError):
            logger.error("OpenAI API internal server error: %s", e)
        elif isinstance(e, openai.RateLimitError):
            logger.error("Rate limit reached: %s", e)
        else:
            logger.error("An unexpected error occurred: %s", e)

    def invoke(self, current_conversation: List[Dict[str, str]]) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=current_conversation,  # type: ignore
            )
            return completion.choices[0].message.content or ""
        except Exception as e:
            self._handle_api_error(e)
            raise

    def stream(
        self, current_conversation: List[Dict[str, str]]
    ) -> Generator[str, None, None]:
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=current_conversation,  # type: ignore
                stream=True,
            )
            for chunk in completion:
                token = chunk.choices[0].delta.content or ""
                yield token
        except Exception as e:
            self._handle_api_error(e)
            raise

    async def ainvoke(self, current_conversation: List[Dict[str, str]]) -> str:
        try:
            completion = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=current_conversation,  # type: ignore
            )
            return completion.choices[0].message.content or ""
        except Exception as e:
            self._handle_api_error(e)
            raise

    async def astream(
        self, current_conversation: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        async def async_generator():
            try:
                completion = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=current_conversation,  # type: ignore
                    stream=True,
                )
                async for chunk in completion:
                    token = chunk.choices[0].delta.content or ""
                    yield token
            except Exception as e:
                self._handle_api_error(e)
                raise

        return async_generator()


class CustomChatLlamaReplicate(BaseChat):
    default_system_prompt = "You are a helpful assistant."

    def __init__(self, model_name: str = "meta/llama-2-70b-chat"):
        self.model_name = model_name
        logger.info("CustomChatLlamaReplicate initilized.")

    def _handle_error(self, error: Exception, context: str = ""):
        if context:
            logger.error(f"Error during {context}: {error}")
        else:
            logger.error(f"Error: {error}")

    def process_chat(self, chat: List[Dict[str, str]]):
        system_prompt = (
            chat[0]["content"]
            if chat[0]["role"] == "system"
            else self.default_system_prompt
        )
        parts = []

        for turn in chat:
            if turn["role"] == "system":
                system_prompt = turn["content"]
                continue

            if turn["role"] == "user":
                if system_prompt != "":
                    parts.append(
                        f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{turn['content']} [/INST]"
                    )
                    system_prompt = ""
                else:
                    parts.append(f"<s>[INST] {turn['content']} [/INST]")

            if turn["role"] == "assistant":
                parts.append(f" {turn['content']} </s>")

        return "".join(parts)

    def invoke(self, current_conversation: List[Dict[str, str]]) -> str:
        try:
            output = replicate.run(
                self.model_name,
                input={
                    "prompt": self.process_chat(current_conversation),
                    "max_new_tokens": 512,
                    "temperature": 0.75,
                },
            )
            return "".join(output)
        except Exception as e:
            self._handle_error(e, context="invoke")
            raise

    def stream(
        self, current_conversation: List[Dict[str, str]]
    ) -> Generator[str, None, None]:
        try:
            output = replicate.stream(
                self.model_name,
                input={
                    "prompt": self.process_chat(current_conversation),
                    "max_new_tokens": 512,
                    "temperature": 0.75,
                },
            )
            for chunk in output:
                if chunk.event.value == "output":
                    yield chunk.data
                elif chunk.event.value == "error":
                    logger.error(f"Error in Replicate stream: {chunk.data}")
                elif chunk.event.value == "done":
                    logger.info(f"Replicate stream done: {chunk.data or 'successful'}")
        except Exception as e:
            self._handle_error(e, context="stream")
            raise

    async def ainvoke(self, current_conversation: List[Dict[str, str]]) -> str:
        try:
            output = await replicate.async_run(
                self.model_name,
                input={
                    "prompt": self.process_chat(current_conversation),
                    "max_new_tokens": 512,
                    "temperature": 0.75,
                },
            )
            response = ""
            async for chunk in output:
                if chunk.event.value == "output":
                    response += chunk.data
                elif chunk.event.value == "error":
                    logger.error(f"Error in Replicate stream: {chunk.data}")
                elif chunk.event.value == "done":
                    logger.info(f"Replicate stream done: {chunk.data or 'successful'}")
            return response
        except Exception as e:
            self._handle_error(e, context="ainvoke")
            raise

    async def astream(
        self, current_conversation: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        async def async_generator():
            try:
                output = await replicate.async_stream(
                    self.model_name,
                    input={
                        "prompt": self.process_chat(current_conversation),
                        "max_new_tokens": 512,
                        "temperature": 0.75,
                    },
                )
                async for chunk in output:
                    if chunk.event.value == "output":
                        yield chunk.data
                    elif chunk.event.value == "error":
                        logger.error(f"Error in Replicate stream: {chunk.data}")
                    elif chunk.event.value == "done":
                        logger.info(
                            f"Replicate stream done: {chunk.data or 'successful'}"
                        )
            except Exception as e:
                self._handle_error(e, context="astream")
                raise

        return async_generator()


class CustomChatGroq(BaseChat):
    default_generation_kwargs = {
        "temperature": 0.5,
        "max_tokens": 1024,
        "top_p": 0.9,
        "stop": [],
    }

    def __init__(
        self,
        model_name: str = "gemma-7b-it",
        generation_kwargs: Optional[dict] = None,
    ):
        self.api_key = self._get_groq_api_key()
        self.client = groq.Groq(api_key=self.api_key)
        self.async_client = groq.AsyncGroq(api_key=self.api_key)
        self.model_name = model_name
        self.generation_kwargs = generation_kwargs or self.default_generation_kwargs
        logger.info("CustomChatGroq initialized.")

    def _get_groq_api_key(self):
        groq_api_key = settings.groq_api_key.get_secret_value()
        return groq_api_key

    def _handle_api_error(self, e: Exception):
        """Centralized error handling for Groq API errors."""
        if isinstance(e, groq.APITimeoutError):
            logger.error("The request took too long: %s", e)
        elif isinstance(e, groq.APIConnectionError):
            logger.error(
                "Request could not reach the Groq API servers or establish a secure connection: %s",
                e,
            )
        elif isinstance(e, groq.AuthenticationError):
            logger.error("API key or token is invalid, expired, or revoked: %s", e)
        elif isinstance(e, groq.InternalServerError):
            logger.error("Groq API internal server error: %s", e)
        elif isinstance(e, groq.RateLimitError):
            logger.error("Rate limit reached: %s", e)
        else:
            logger.error("An unexpected error occurred: %s", e)

    def invoke(self, current_conversation: List[Dict[str, str]]) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=current_conversation,  # type: ignore
                **self.generation_kwargs,
            )
            return completion.choices[0].message.content or ""
        except Exception as e:
            self._handle_api_error(e)
            raise

    def stream(
        self, current_conversation: List[Dict[str, str]]
    ) -> Generator[str, None, None]:
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=current_conversation,  # type: ignore
                **self.generation_kwargs,
                stream=True,
            )
            for chunk in completion:
                token = chunk.choices[0].delta.content or ""
                yield token
        except Exception as e:
            self._handle_api_error(e)
            raise

    async def ainvoke(self, current_conversation: List[Dict[str, str]]) -> str:
        try:
            completion = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=current_conversation,  # type: ignore
                **self.generation_kwargs,
            )
            return completion.choices[0].message.content or ""
        except Exception as e:
            self._handle_api_error(e)
            raise

    async def astream(
        self, current_conversation: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        async def async_generator():
            try:
                completion = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=current_conversation,  # type: ignore
                    **self.generation_kwargs,
                    stream=True,
                )
                async for chunk in completion:
                    token = chunk.choices[0].delta.content or ""
                    yield token
            except Exception as e:
                self._handle_api_error(e)
                raise

        return async_generator()
