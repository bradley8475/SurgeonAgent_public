import os
import json

from openai import OpenAI

from datetime import datetime

from config.load_config import YAMLConfig
from tools import get_tool_registry

import logging

logger = logging.getLogger(__name__)


class LLM:
    def __init__(
        self,
        name: str,
        llm_config: YAMLConfig,
        global_config: YAMLConfig,
    ):
        self.name = name
        self.config = llm_config
        self.logging_path = f"{global_config.logging.path}/{self.name}"
        self.model_name = llm_config.model_name

        os.makedirs(self.logging_path, exist_ok=True)

        api_key_env = getattr(llm_config, "api_key_env", None)
        if api_key_env:
            self.api_key = os.environ.get(api_key_env)
            if not self.api_key:
                raise RuntimeError(f"api_key_env={api_key_env!r} not set in environment / .env")
        else:
            self.api_key = (
                os.environ.get("self-api-key")
                or os.getenv("DASHSCOPE_API_KEY")
                or os.getenv("DASHSCOPE-API-KEY")
                or os.getenv("LLM_API_KEY")
            )
        base_url = getattr(llm_config, "base_url", None) or "https://idealab.alibaba-inc.com/api/openai/v1"
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)

        self.tool_registry = get_tool_registry()
        self.tools = self.tool_registry.get_tool_schemas(self.config.tools)
        logger.info(f"Created LLM instance for {self.name}")
        logger.info(f"Tools: {[t['function']['name'] for t in self.tools]}")

        self.context = []

        with open(self.config.system_prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

        self.context.append({"role": "system", "content": self.system_prompt, "cache_control": {"type": "ephemeral" }})

    def reset_context(self):
        """Reset context to just the system prompt — for per-sample eval loops."""
        self.context = [
            {"role": "system", "content": self.system_prompt, "cache_control": {"type": "ephemeral"}}
        ]

    def generate_response(self, messages: list[dict], max_retries=3):
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.context.extend(messages)

        with open(f"{self.logging_path}/{date_str}_input.json", "w", encoding="utf-8") as f:
            json.dump(self.context, f, indent=4, ensure_ascii=False)

        for i in range(max_retries):
            try:
                raw_json = None
                kwargs = {"model": self.model_name, "messages": self.context}
                if self.tools:
                    kwargs["tools"] = self.tools
                max_tokens = getattr(self.config, "max_tokens", None)
                if max_tokens:
                    kwargs["max_tokens"] = max_tokens
                response_raw = self.client.chat.completions.create(**kwargs)
                raw_json = response_raw.model_dump()
                message = response_raw.choices[0].message
                message_json = message.model_dump()
                self.context.append(
                    {
                        "role": "assistant",
                        "content": message_json["content"],
                        "tool_calls": message_json["tool_calls"],
                        # "reasoning": message_json["reasoning"], # gpt says this prompts the model to respond again.
                    }
                )
                break
            except Exception as e:
                logger.error(f"Error generating response: {e}.")
                if raw_json is not None:
                    logger.error(f"Raw response JSON: {raw_json}")
                if i == max_retries - 1:
                    raise e
                else:
                    logger.info(f"Retrying... ({i + 1}/{max_retries})")
                    continue

        with open(f"{self.logging_path}/{date_str}_response.json", "w", encoding="utf-8") as f:
            json.dump(message_json, f, indent=4, ensure_ascii=False)

        return message

    def run(self, messages: list[dict], max_rounds: int | None = None):
        """Iterative agent loop. Stops when the model emits a message with no tool calls,
        or when max_rounds is reached (raises RuntimeError so the caller can decide).
        max_rounds=None (default) means unbounded (legacy behavior).
        """
        rounds = 0
        next_messages = messages
        while True:
            if max_rounds is not None and rounds >= max_rounds:
                raise RuntimeError(f"LLM.run exceeded max_rounds={max_rounds} for agent {self.name!r}")
            rounds += 1
            message = self.generate_response(next_messages)

            if message.tool_calls is None or len(message.tool_calls) == 0:
                return message

            tool_messages = []
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                tool_response = self.tool_registry.call_tool(name, args)
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_response.content,
                    }
                )
                if len(tool_response.additional_messages) > 0:
                    if name == "retrieve_reference":
                        image_description = "The images below show the reference garment designs corresponding to the JSON specifications above."
                    else:
                        image_description = "These are the simulation results from the garment design model."
                    tool_messages.append(
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": image_description,
                                },
                                *tool_response.additional_messages,
                            ],
                        }
                    )
            next_messages = tool_messages
