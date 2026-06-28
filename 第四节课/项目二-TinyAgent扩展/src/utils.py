# ┌─────────────────────────────────────────────────────────────────────┐
# │ 项目二扩展版：Tiny-Agent + 更多工具                                  │
# │ 在原始 happy-llm Tiny-Agent 基础上添加：                             │
# │   - 计算器（加减乘除幂）                                              │
# │   - 文件读取器                                                        │
# │   - 字符串处理工具（反转/统计/大小写转换）                             │
# │   - 随机数生成器                                                      │
# └─────────────────────────────────────────────────────────────────────┘

import inspect
from datetime import datetime

def function_to_json(func) -> dict:
    """
    核心函数：将 Python 函数自动转为 OpenAI Function Calling 的 JSON Schema
    
    原理:
        - inspect.signature(func) 获取函数的参数列表和类型注解
        - func.__doc__ 作为函数的描述（LLM 据此判断何时调用）
        - 类型注解 str→"string", float→"number", int→"integer" 等
        - 没有默认值的参数自动标记为 required
    """
    # Python 类型 → JSON Schema 类型映射
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(f"无法获取函数 {func.__name__} 的签名: {str(e)}")

    # 构建参数信息
    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(f"未知的类型注解 {param.annotation}，参数名: {param.name}: {str(e)}")
        parameters[param.name] = {"type": param_type}

    # 找出必需参数（没有默认值的）
    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }
