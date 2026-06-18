import base64
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys

import requests

from tools.pattern_validator import validate_pattern_json
from tools.registry import ToolRegistry, ToolResponse
from utils.image import ImageInterface
from utils.vector_index import ClipFaissIndex

from cad.api import load_pattern, visualize_pattern, simulate_pattern
from cad.core import Pattern
from pathlib import Path

logger = logging.getLogger(__name__)

WORKDIR = os.path.join(os.path.dirname(__file__), "..", "workspace")


tool_registry = ToolRegistry()


def get_tool_registry():
    return tool_registry


# Tool definitions - all raise errors, no try/catch
@tool_registry.register
def run_command(command: str) -> ToolResponse:
    """Run a terminal command and receive the output"""
    pyexec = shlex.quote(sys.executable)

    shim = f"""
    PYEXEC={pyexec}
    python() {{ "$PYEXEC" "$@"; }}
    python3() {{ "$PYEXEC" "$@"; }}
    pip() {{ "$PYEXEC" -m pip "$@"; }}
    pip3() {{ "$PYEXEC" -m pip "$@"; }}
    """

    full = f"{shim}\n{command}"
    result = subprocess.run(
        ["bash", "-lc", full],
        cwd=WORKDIR,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )

    output = result.stdout + result.stderr if result.stderr else result.stdout
    return ToolResponse(content=output)


@tool_registry.register
def view_image(path: str) -> ToolResponse:
    """
    View the image.
    IMPORTANT: If you don't have the picture path (end with png/jpg/jpeg), don't make wild guesses, don't use this tool.
    """
    print(path)
    image_data = ImageInterface(file_path=os.path.join(WORKDIR, path))
    return ToolResponse(images=[image_data])


@tool_registry.register
def read_file(path: str) -> ToolResponse:
    """Read the content of a file"""
    file_path = os.path.join(WORKDIR, path)

    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    return ToolResponse(content=content)


@tool_registry.register
def simulate(json_path: str) -> ToolResponse:
    """
    进行服装设计仿真和可视化

    功能：
    1. 加载JSON数据创建Pattern对象
    2. 生成平面展开可视化图像
    3. 生成仿真可视化图像
    4. 保存输入的JSON文件
    5. 将所有图像放置在同一个输出文件夹中
    """

    json_file_path = os.path.join(WORKDIR, json_path)

    # 直接使用JSON文件所在的目录作为输出目录，不创建额外文件夹
    output_folder = os.path.dirname(json_file_path)
    if not output_folder:  # 如果JSON在根目录
        output_folder = WORKDIR

    with open(json_file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    try:
        # 直接加载JSON数据创建Pattern对象
        from cad.core import Pattern

        pattern = Pattern.load(json_data)
        logger.info("✓ 成功加载JSON数据创建Pattern对象")

        all_images = []

        # 平面展开可视化
        try:
            visualize_pattern(pattern, output_folder)
            logger.info("✓ 平面展开可视化完成")
        except Exception as viz_e:
            logger.error(f"平面展开可视化失败: {str(viz_e)}")

        # 仿真可视化
        try:
            simulate_pattern(pattern, output_folder)
            logger.info("✓ 仿真可视化完成")
        except Exception as sim_e:
            logger.error(f"仿真可视化失败: {str(sim_e)}")

        # 收集输出文件夹中的所有图像文件
        if os.path.exists(output_folder):
            for filename in os.listdir(output_folder):
                if filename.lower().endswith((".png", ".jpg", ".jpeg", ".svg")):
                    image_path = os.path.join(output_folder, filename)
                    all_images.append(ImageInterface(file_path=image_path))

        result_message = "仿真和可视化完成！\n"
        result_message += f"输出目录: {output_folder}\n"
        result_message += f"生成图像数量: {len(all_images)} 张"

        return ToolResponse(images=all_images, content=result_message)

    except Exception as e:
        error_message = f"仿真失败: {str(e)}"
        logger.error(error_message)
        return ToolResponse(content=error_message)

    # 注释掉的传统仿真代码
    # spec_data = process_specification(json_data)
    # # 运行传统仿真服务
    # try:
    #     reference_image = ImageInterface(file_path=os.path.join(WORKDIR, "target.jpg"))
    #     rpc_client = xmlrpc_client.ServerProxy("http://localhost:8051", allow_none=True)
    #     result = rpc_client.run_simulation(spec_data, reference_image.base64_data)
    #     # ... 传统仿真处理逻辑 ...


@tool_registry.register
def retrieve_reference(prompt: str, topk: int = 1) -> ToolResponse:
    """Retrieve reference garment specification JSON examples and render images from local vector index.

    The returned specifications demonstrate professional garment construction patterns:
    - Garments often use split-panel designs (e.g., front torso split into left_ftorso/right_ftorso)
    - Split-panel designs typically have translation X=0 because each panel handles one side
    - When creating single-panel designs, translation values MUST be recalculated based on the panel's (0,0) origin position

    IMPORTANT Examples for translation calculation:
    - Body panels: Split at X=0 → Single panel needs X offset (e.g., [-19.66, 91.4, 30.0])
    - When referencing panel attachment points, adhere to ergonomic principles and garment pattern-making standards rather than relying solely on example coordinates.

    Processing workflow:
    1. Retrieves JSON specifications and images from local vector index using text prompt
    2. Uses parse_gcd to load patterns from JSON
    3. Applies pattern.compile() to generate processed specifications
    4. Returns compiled JSON and retrieved images
    
    Args:
        prompt: Text description for searching (e.g., "off-shoulder long-sleeve sweater")
        topk: Number of results to retrieve (default: 1)
    
    Returns both compiled JSON specifications and corresponding retrieved images directly to the model."""
    
    # 验证输入
    if not prompt or not prompt.strip():
        return ToolResponse(
            content="错误: 必须提供文本描述（prompt）"
        )
    
    # 初始化向量索引
    try:
        index = ClipFaissIndex()
    except Exception as e:
        return ToolResponse(
            content=f"错误: 无法初始化向量索引: {str(e)}\n请确保已构建向量索引（运行 utils/build_vector_index.py）"
        )
    
    # 使用文本查询
    try:
        search_results = index.search(prompt, k=topk)
        logger.info(f"使用文本提示词 '{prompt}' 进行向量索引搜索，找到 {len(search_results)} 个结果")
    except Exception as e:
        return ToolResponse(
            content=f"错误: 向量索引搜索失败: {str(e)}"
        )
    
    if not search_results:
        return ToolResponse(
            content="未找到匹配的结果。请检查向量索引是否已正确构建。"
        )

    # 创建主输出文件夹
    base_folder = os.path.join(WORKDIR, "retrieve_reference_output")
    os.makedirs(base_folder, exist_ok=True)

    images = []
    all_specs = []  # 收集所有规格数据
    parsed_patterns = []  # 收集解析后的pattern
    output_folders = []  # 记录每个pattern的输出文件夹

    # CAD API现在可以正常使用
    logger.info("✓ CAD模块可用")

    for i, result in enumerate(search_results):
        # 为每个召回的JSON创建独立文件夹
        pattern_folder = os.path.join(base_folder, f"pattern_{i}")
        os.makedirs(pattern_folder, exist_ok=True)
        output_folders.append(pattern_folder)

        json_path = result["json_path"]
        score = result.get("score", 0.0)
        
        # 从JSON路径推断图像路径（在同一目录下）
        json_dir = os.path.dirname(json_path)
        image_path = os.path.join(json_dir, "render_front.png")
        
        garment_spec = None

        # 读取JSON文件
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    garment_spec = json.load(f)
                logger.info(f"成功读取JSON文件: {json_path} (相似度: {score:.4f})")
            except Exception as e:
                logger.error(f"读取JSON文件失败 {json_path}: {str(e)}")
                continue
        else:
            logger.warning(f"JSON文件不存在: {json_path}")
            continue

        # 复制图像文件到输出文件夹
        if os.path.exists(image_path):
            try:
                output_image_path = os.path.join(pattern_folder, "render_front.png")
                shutil.copy2(image_path, output_image_path)
                images.append(ImageInterface(file_path=output_image_path))
                logger.info(f"召回的图片: {output_image_path}")
            except Exception as e:
                logger.warning(f"复制图像文件失败 {image_path}: {str(e)}")
        else:
            logger.warning(f"图像文件不存在: {image_path}")

        if garment_spec:
            all_specs.append(garment_spec)

            try:
                # 数据集已是 arrangement/uvr 新格式，直接 Pattern.load 即可
                # （旧版走 parse_gcd 是因为 GarmentCode 数据集是 translation/rotation 旧格式，需要人体模型反向映射）
                pattern = Pattern.load(garment_spec)
                logger.info(f"成功解析第{i}个pattern")

                # 生成处理后的规格数据并保存
                specification = None  # 初始化变量
                try:
                    specification = pattern.compile()
                    parsed_patterns.append(specification)

                    pattern_json_path = os.path.join(pattern_folder, "pattern.json")
                    with open(pattern_json_path, "w", encoding="utf-8") as f:
                        json.dump(specification, f, indent=2, ensure_ascii=False)
                    logger.info(f"成功编译并保存第{i}个pattern的JSON文件")
                except Exception as compile_e:
                    logger.error(f"第{i}个pattern编译失败: {str(compile_e)}")
                    parsed_patterns.append(None)
                    specification = None  # 确保失败时为None

                # 直接使用编译后的JSON
                if specification:  # 确保有处理后的规格数据
                    logger.info(f"第{i}个pattern编译完成")
                else:
                    logger.warning(f"第{i}个pattern没有处理后的规格数据")

            except Exception as e:
                logger.error(f"处理第{i}个pattern时发生异常: {str(e)}")
                parsed_patterns.append(None)

    # 构建返回内容，显示处理后的JSON代码和召回的图片供模型参考
    result_message = (
        f"成功处理 {len(all_specs)} 个召回的pattern，包含 {len(images)} 张图片（召回的参考图片）\n"
    )

    # 只添加处理后的JSON规格（隐藏原始JSON）
    if parsed_patterns and any(p is not None for p in parsed_patterns):
        result_message += "=== 参考JSON规格 ===\n"
        for i, pattern in enumerate(parsed_patterns):
            if pattern is not None:
                result_message += f"\n--- Pattern {i + 1} 规格 ---\n"
                result_message += json.dumps(pattern, indent=2, ensure_ascii=False)
                result_message += "\n"
    else:
        result_message += "⚠️ 没有成功处理的JSON规格数据\n"

    result_message += f"\n输出文件夹: {base_folder}\n"
    result_message += f"各pattern文件夹: {output_folders}"

    return ToolResponse(content=result_message, images=images)


@tool_registry.register
def validate_json_pattern(json_path: str) -> ToolResponse:
    """Validate garment pattern JSON file format according to coder specifications"""
    file_path = os.path.join(WORKDIR, json_path)
    is_valid, errors = validate_pattern_json(file_path)

    if is_valid:
        result = f"JSON pattern is valid: {json_path} \n"
    else:
        result = f"JSON pattern is NOT valid: {json_path} \nErrors:\n" + "\n".join(
            f"- {e}" for e in errors
        )

    return ToolResponse(content=result)


@tool_registry.register
def visualize_pattern_tool(
    json_path: str, output_folder: str = "visualization"
) -> ToolResponse:
    """
    平面展开可视化工具
    从JSON文件加载Pattern并生成平面展开可视化图像
    """
    json_file_path = os.path.join(WORKDIR, json_path)
    output_path = os.path.join(WORKDIR, output_folder)

    try:
        # 读取JSON数据
        with open(json_file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        # 使用CAD API加载pattern
        pattern = load_pattern(json_data)

        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)

        # 进行平面展开可视化
        viz_images = visualize_pattern(pattern, output_path)
        logger.info("成功生成平面展开可视化")

        # 收集生成的图像文件
        images = []
        for image_path in viz_images:
            images.append(ImageInterface(file_path=image_path))

        result_message = f"成功生成平面展开可视化，输出保存在: {output_folder}"
        if images:
            result_message += f"\n生成了 {len(images)} 张图像"

        return ToolResponse(content=result_message, images=images)

    except Exception as e:
        error_message = f"平面展开可视化失败: {str(e)}"
        logger.error(error_message)
        return ToolResponse(content=error_message)


@tool_registry.register
def simulate_pattern_tool(
    json_path: str, output_folder: str = "simulation"
) -> ToolResponse:
    """
    仿真可视化工具
    从JSON文件加载Pattern并生成仿真可视化图像
    """
    json_file_path = os.path.join(WORKDIR, json_path)
    output_path = os.path.join(WORKDIR, output_folder)

    try:
        # 读取JSON数据
        with open(json_file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        # 使用CAD API加载pattern
        pattern = load_pattern(json_data)

        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)

        # 进行仿真可视化
        sim_images = simulate_pattern(pattern, output_path)
        logger.info("成功生成仿真可视化")

        # 收集生成的图像文件
        images = []
        for image_path in sim_images:
            images.append(ImageInterface(file_path=image_path))

        result_message = f"成功生成仿真可视化，输出保存在: {output_folder}"
        if images:
            result_message += f"\n生成了 {len(images)} 张图像"

        return ToolResponse(content=result_message, images=images)

    except Exception as e:
        error_message = f"仿真可视化失败: {str(e)}"
        logger.error(error_message)
        return ToolResponse(content=error_message)


@tool_registry.register
def manage_todolist(
    action: str,
    task_id: str = None,
    task: str = None,
    tasks: list = None,
    status: str = None,
) -> ToolResponse:
    """
    Manage a persistent todolist for tracking workflow progress and ensuring no critical steps are missed.

    Actions:
    - "add": Add a single task (requires task parameter)
    - "add_batch": Add multiple tasks (requires tasks parameter as list of strings)
    - "list": Show all tasks with their current status
    - "update": Update task status (requires task_id and status parameters)
    - "complete": Mark task as completed (requires task_id)
    - "remove": Remove a task (requires task_id)
    - "clear": Clear all tasks (use with caution)
    - "init_workflow": Initialize common garment design workflow

    Status options: "pending", "in_progress", "completed", "blocked"

    Example usage:
    - manage_todolist("add", task="Generate initial JSON pattern")
    - manage_todolist("add_batch", tasks=["Task 1", "Task 2", "Task 3"])
    - manage_todolist("init_workflow")  # Quick start for garment design
    - manage_todolist("update", task_id="1", status="in_progress")
    - manage_todolist("complete", task_id="1")
    - manage_todolist("list")
    """

    todolist_file = os.path.join(WORKDIR, "todolist.json")

    # Load existing todolist or create empty one
    if os.path.exists(todolist_file):
        with open(todolist_file, "r", encoding="utf-8") as f:
            todolist = json.load(f)
    else:
        todolist = {"tasks": [], "next_id": 1}

    def save_todolist():
        with open(todolist_file, "w", encoding="utf-8") as f:
            json.dump(todolist, f, indent=2, ensure_ascii=False)

    def find_task_by_id(tid):
        for i, t in enumerate(todolist["tasks"]):
            if str(t["id"]) == str(tid):
                return i, t
        return None, None

    def format_task_list():
        if not todolist["tasks"]:
            return "📝 Todo List is empty"

        result = "📝 Current Todo List:\n" + "=" * 40 + "\n"

        status_icons = {
            "pending": "⏳",
            "in_progress": "🔄",
            "completed": "✅",
            "blocked": "🚫",
        }

        for task in todolist["tasks"]:
            icon = status_icons.get(task["status"], "❓")
            result += (
                f"{icon} [{task['id']}] {task['description']} ({task['status']})\n"
            )

        # Summary stats
        stats = {}
        for task in todolist["tasks"]:
            stats[task["status"]] = stats.get(task["status"], 0) + 1

        result += "\n📊 Summary: " + " | ".join(
            [f"{status}: {count}" for status, count in stats.items()]
        )
        return result

    # Handle different actions
    if action == "add":
        if not task:
            return ToolResponse(
                content="❌ Error: Task description is required for 'add' action"
            )

        new_task = {
            "id": todolist["next_id"],
            "description": task,
            "status": "pending",
            "created_at": str(__import__("datetime").datetime.now()),
        }

        todolist["tasks"].append(new_task)
        todolist["next_id"] += 1
        save_todolist()

        return ToolResponse(content=f"✅ Added task [{new_task['id']}]: {task}")

    elif action == "add_batch":
        if not tasks or not isinstance(tasks, list):
            return ToolResponse(
                content="❌ Error: tasks parameter must be a list of strings for 'add_batch' action"
            )

        added_tasks = []
        for task_desc in tasks:
            if isinstance(task_desc, str) and task_desc.strip():
                new_task = {
                    "id": todolist["next_id"],
                    "description": task_desc.strip(),
                    "status": "pending",
                    "created_at": str(__import__("datetime").datetime.now()),
                }
                todolist["tasks"].append(new_task)
                added_tasks.append(f"[{todolist['next_id']}] {task_desc.strip()}")
                todolist["next_id"] += 1

        save_todolist()
        return ToolResponse(
            content=f"✅ Added {len(added_tasks)} tasks:\n" + "\n".join(added_tasks)
        )

    elif action == "init_workflow":
        # Predefined garment design workflow
        workflow_tasks = [
            "Observe target image & decompose panels (front, back, sleeves, etc.)",
            "Retrieve and analyze reference JSON using retrieve_reference()",
            "Plan 2D panel topology (vertices, edges, stitches)",
            "Generate initial JSON",
            "Run simulation and check for errors",
            "Analyze final placement view (last simulation image)",
            "Quantify positional deviations (cm/% values)",
            "Apply corrections to JSON (translation/rotation/vertices)",
            "Re-run simulation and validate improvements",
            "Generate final report with corrections applied",
        ]

        added_count = 0
        for task_desc in workflow_tasks:
            new_task = {
                "id": todolist["next_id"],
                "description": task_desc,
                "status": "pending",
                "created_at": str(__import__("datetime").datetime.now()),
            }
            todolist["tasks"].append(new_task)
            todolist["next_id"] += 1
            added_count += 1

        save_todolist()
        return ToolResponse(
            content=f"🚀 Initialized garment design workflow with {added_count} tasks!\n\n"
            + format_task_list()
        )

    elif action == "list":
        return ToolResponse(content=format_task_list())

    elif action == "update":
        if not task_id or not status:
            return ToolResponse(
                content="❌ Error: Both task_id and status are required for 'update' action"
            )

        idx, found_task = find_task_by_id(task_id)
        if not found_task:
            return ToolResponse(content=f"❌ Error: Task with ID {task_id} not found")

        valid_statuses = ["pending", "in_progress", "completed", "blocked"]
        if status not in valid_statuses:
            return ToolResponse(
                content=f"❌ Error: Status must be one of: {', '.join(valid_statuses)}"
            )

        old_status = found_task["status"]
        todolist["tasks"][idx]["status"] = status
        todolist["tasks"][idx]["updated_at"] = str(
            __import__("datetime").datetime.now()
        )
        save_todolist()

        return ToolResponse(
            content=f"✅ Updated task [{task_id}]: {old_status} → {status}"
        )

    elif action == "complete":
        if not task_id:
            return ToolResponse(
                content="❌ Error: task_id is required for 'complete' action"
            )

        idx, found_task = find_task_by_id(task_id)
        if not found_task:
            return ToolResponse(content=f"❌ Error: Task with ID {task_id} not found")

        todolist["tasks"][idx]["status"] = "completed"
        todolist["tasks"][idx]["completed_at"] = str(
            __import__("datetime").datetime.now()
        )
        save_todolist()

        return ToolResponse(
            content=f"✅ Completed task [{task_id}]: {found_task['description']}"
        )

    elif action == "remove":
        if not task_id:
            return ToolResponse(
                content="❌ Error: task_id is required for 'remove' action"
            )

        idx, found_task = find_task_by_id(task_id)
        if not found_task:
            return ToolResponse(content=f"❌ Error: Task with ID {task_id} not found")

        removed_task = todolist["tasks"].pop(idx)
        save_todolist()

        return ToolResponse(
            content=f"✅ Removed task [{task_id}]: {removed_task['description']}"
        )

    elif action == "clear":
        task_count = len(todolist["tasks"])
        todolist["tasks"] = []
        save_todolist()

        return ToolResponse(content=f"✅ Cleared all {task_count} tasks from todolist")

    else:
        valid_actions = [
            "add",
            "add_batch",
            "init_workflow",
            "list",
            "update",
            "complete",
            "remove",
            "clear",
        ]
        return ToolResponse(
            content=f"❌ Error: Invalid action '{action}'. Valid actions: {', '.join(valid_actions)}"
        )
