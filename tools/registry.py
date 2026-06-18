import inspect
import logging
from typing import Any, get_type_hints

from utils.image import ImageInterface

logger = logging.getLogger(__name__)


class ToolResponse:
    def __init__(
        self,
        content: str | None = None,
        images: list[ImageInterface] | None = None,
    ):
        if content is None and images is None:
            raise ValueError("ToolResponse must have either content or images")

        if content is None:
            self.content = "Success! Please check the images."
        else:
            self.content = content

        self.additional_messages = []

        if images is None:
            self.images = []
        else:
            self.images = images

        for image in self.images:
            self.additional_messages.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image.uri_data},
                }
            )

    def __repr__(self):
        def _teal(x: str) -> str:
            return f"\033[96m{x}\033[0m"

        def _indent(x: str, n: int) -> str:
            return "\n".join(" " * n + line for line in x.splitlines())

        def _truncate(x: str, n: int) -> str:
            return x[:n] + "..." if len(x) > n else x

        # collect images (truncate URI data for readability)
        if self.images:
            img_str = "\n".join(
                f"{_truncate(image.uri_data, 100)}" for image in self.images
            )
        else:
            img_str = "(none)"

        content_str = _truncate(self.content, 400)

        return (
            "ToolResponse(\n"
            f"{_indent(_teal('content:'), 4)}\n{_indent(content_str, 8)}\n"
            f"{_indent(_teal('images:'), 4)}\n{_indent(img_str, 8)}\n"
            ")"
        )


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, dict[str, Any]] = {}

    def register(self, func):
        """Register a tool function"""
        schema = self._generate_schema(func)

        # Store the original function directly
        self._tools[func.__name__] = {
            "schema": schema,
            "callable": func,
        }
        return func

    def _generate_schema(self, func) -> dict[str, Any]:
        """Generate OpenAI-compatible tool schema"""
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            param_type = type_hints.get(param_name, str)
            properties[param_name] = {"type": self._to_json_type(param_type.__name__)}

            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": func.__doc__ or "",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def _to_json_type(self, type_name: str) -> str:
        if type_name == "str":
            return "string"
        elif type_name == "int":
            return "integer"
        elif type_name == "float":
            return "number"
        elif type_name == "bool":
            return "boolean"
        elif type_name in ("list", "tuple"):
            return "array"
        elif type_name == "dict":
            return "object"
        else:
            return type_name

    def get_tool_schemas(self, names: list[str]) -> list[dict[str, Any]]:
        result = []
        for name in names:
            if name not in self._tools:
                logger.warning(f"Tool {name} not found")
                continue
            result.append(self._tools[name]["schema"])
        return result

    def call_tool(self, name: str, args: dict[str, Any]) -> ToolResponse:
        if name not in self._tools:
            logger.warning(f"Tool {name} not found")
            return ToolResponse(content=f"ERROR: Tool {name} not found")

        func = self._tools[name]["callable"]
        logger.info(f"Calling tool {name} with args {args}")

        try:
            result = func(**args)
            if not isinstance(result, ToolResponse):
                raise ValueError(f"Tool {name} did not return a ToolResponse")
            return result
        except Exception as e:
            logger.error(f"Tool {name} failed: {str(e)}")
            return ToolResponse(content=f"ERROR: {str(e)}")
