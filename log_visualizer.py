#!/usr/bin/env python3
"""
Log Visualizer Server

A Flask web server to visualize agent conversation logs with tabs for each agent,
showing only the final input and final response for each agent.

Example: python log_visualizer.py logs_single_agent
       python log_visualizer.py --file logs/l1_designer/l1_designer/20260519_210105_input.json
"""

import argparse
import json
import os
import re
from pathlib import Path

SESSION_NAME_RE = re.compile(r"^\d{8}_\d{6}$")

from flask import Flask, jsonify, render_template

app = Flask(__name__)


def find_log_sessions(logs_dir):
    """Find all log session directories."""
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        return []

    sessions = []
    for item in logs_path.iterdir():
        if item.is_dir() and SESSION_NAME_RE.match(item.name):
            sessions.append(item.name)

    return sorted(sessions, reverse=True)


def load_single_file(file_path):
    """Load a single input/response log file (auto-pairs with sibling)."""
    file_path = Path(file_path)
    if not file_path.exists():
        return {}

    stem = file_path.stem
    parent = file_path.parent
    agent_name = parent.name

    if stem.endswith("_input"):
        timestamp = stem[: -len("_input")]
        input_file = file_path
        response_file = parent / f"{timestamp}_response.json"
    elif stem.endswith("_response"):
        timestamp = stem[: -len("_response")]
        response_file = file_path
        input_file = parent / f"{timestamp}_input.json"
    else:
        timestamp = stem
        input_file = file_path
        response_file = parent / f"{stem}_response.json"

    agent_data = {}
    if input_file.exists():
        agent_data["final_input"] = json.load(open(input_file, "r"))
        agent_data["input_file"] = input_file.name
    if response_file.exists():
        agent_data["final_response"] = json.load(open(response_file, "r"))
        agent_data["response_file"] = response_file.name

    if not agent_data:
        return {}

    return {f"{agent_name} - {timestamp}": agent_data}


def _collect_agent_data(agent_dir):
    """Return one agent's final input/response payload, or None."""
    input_files = sorted(agent_dir.glob("*_input.json"))
    if not input_files:
        return None

    final_input_file = input_files[-1]
    timestamp = final_input_file.stem.replace("_input", "")
    final_response_file = agent_dir / f"{timestamp}_response.json"

    agent_data = {
        "final_input": json.load(open(final_input_file, "r")),
        "input_file": final_input_file.name,
    }
    if final_response_file.exists():
        agent_data["final_response"] = json.load(open(final_response_file, "r"))
        agent_data["response_file"] = final_response_file.name
    return agent_data


def get_agent_logs(session_dir):
    """Get logs for all agents in a session.

    Supports two layouts:
      A) <session>/<agent_name>/<ts>_input.json   (nested, original)
      B) <session>/<ts>_input.json                (flat, single agent — uses
         the parent directory name as the agent name)
    """
    session_path = Path(session_dir)
    if not session_path.exists():
        return {}

    agents = {}

    # Layout A: nested agent subdirs
    for agent_dir in session_path.iterdir():
        if agent_dir.is_dir():
            data = _collect_agent_data(agent_dir)
            if data is not None:
                agents[agent_dir.name] = data

    # Layout B: flat — session dir itself holds the *_input.json files
    if not agents and any(session_path.glob("*_input.json")):
        data = _collect_agent_data(session_path)
        if data is not None:
            agents[session_path.parent.name or session_path.name] = data

    return agents


def format_content(content):
    """Format content for display, handling both strings and lists."""
    if isinstance(content, str):
        return content.strip()
    elif isinstance(content, list):
        formatted_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_content = item.get("text", "").strip()
                    if text_content:
                        formatted_parts.append(text_content)
                elif item.get("type") == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    if image_url.startswith("data:image"):
                        formatted_parts.append(
                            f'<img src="{image_url}" style="max-width: 300px; max-height: 300px;" alt="Base64 Image">'
                        )
                    else:
                        formatted_parts.append(
                            f'<img src="{image_url}" style="max-width: 300px; max-height: 300px;" alt="Image">'
                        )
                else:
                    formatted_parts.append(str(item).strip())
            else:
                formatted_parts.append(str(item).strip())
        return "<br>".join(part for part in formatted_parts if part)
    else:
        return str(content).strip()


def format_tool_calls(tool_calls):
    """Format tool calls for display."""
    if not tool_calls:
        return None

    formatted = []
    for call in tool_calls:
        call_info = f"<strong>Tool:</strong> {call.get('function', {}).get('name', 'Unknown')}<br>"
        if "function" in call and "arguments" in call["function"]:
            args = call["function"]["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            call_info += (
                f"<strong>Arguments:</strong> <pre>{json.dumps(args, indent=2)}</pre>"
            )
        formatted.append(call_info)

    return "<hr>".join(formatted)


def process_agents_for_display(agents):
    """Convert raw agent log data into the structure session.html expects."""
    processed_agents = {}
    for agent_name, data in agents.items():
        processed_agents[agent_name] = {
            "input_file": data.get("input_file", ""),
            "response_file": data.get("response_file", ""),
            "messages": [],
        }

        final_input = data.get("final_input")
        if isinstance(final_input, list):
            for msg in final_input:
                processed_agents[agent_name]["messages"].append({
                    "role": msg.get("role", "unknown"),
                    "content": format_content(msg.get("content", "")),
                    "tool_calls": format_tool_calls(msg.get("tool_calls")),
                })

        if "final_response" in data:
            final_response = data["final_response"]
            processed_agents[agent_name]["messages"].append({
                "role": final_response.get("role", "assistant"),
                "content": format_content(final_response.get("content", "")),
                "tool_calls": format_tool_calls(final_response.get("tool_calls")),
            })

    return processed_agents


@app.route("/")
def index():
    """Main page — single-file mode if --file was provided, else session list."""
    if SINGLE_FILE:
        return view_file()

    logs_dir = Path(LOGS_DIR)
    sessions = find_log_sessions(logs_dir)
    return render_template("index.html", sessions=sessions)


@app.route("/file")
def view_file():
    """View a single log file specified via --file."""
    if not SINGLE_FILE:
        return "未提供 --file 参数", 404

    agents = load_single_file(SINGLE_FILE)
    processed_agents = process_agents_for_display(agents)
    return render_template(
        "session.html",
        session_id=Path(SINGLE_FILE).name,
        agents=processed_agents,
    )


@app.route("/session/<session_id>")
def view_session(session_id):
    """View a specific log session."""
    session_dir = Path(LOGS_DIR) / session_id
    agents = get_agent_logs(session_dir)
    processed_agents = process_agents_for_display(agents)
    return render_template(
        "session.html", session_id=session_id, agents=processed_agents
    )


@app.route("/api/sessions")
def api_sessions():
    """API endpoint to get available sessions."""
    logs_dir = Path(LOGS_DIR)
    sessions = find_log_sessions(logs_dir)
    return jsonify(sessions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="日志可视化工具")
    parser.add_argument(
        "logs_dir", nargs="?", default="logs", help="包含日志文件的目录 (默认: logs)"
    )
    parser.add_argument(
        "--file", dest="file", default=None,
        help="单文件模式：直接查看一个 _input.json / _response.json (会自动配对兄弟文件)",
    )
    parser.add_argument("--port", type=int, default=5003, help="服务器运行端口")
    args = parser.parse_args()

    global LOGS_DIR, SINGLE_FILE
    LOGS_DIR = args.logs_dir
    SINGLE_FILE = args.file

    os.makedirs("templates", exist_ok=True)

    if SINGLE_FILE:
        print(f"启动日志可视化服务器，单文件模式: {SINGLE_FILE}")
    else:
        print(f"启动日志可视化服务器，日志目录: {LOGS_DIR}")
    print(f"服务器地址: http://localhost:{args.port}")
    app.run(debug=True, host="0.0.0.0", port=args.port)
