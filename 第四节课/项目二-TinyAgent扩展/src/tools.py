#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 扩展版工具集合：在原始 7 个工具基础上新增 8 个工具，共计 15 个

import datetime
import random
import os
import wikipedia
import requests


# ========== 原始工具（保留） ==========

def get_current_datetime() -> str:
    """获取真实的当前日期和时间"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def add(a: float, b: float) -> str:
    """计算两个浮点数的和"""
    return str(a + b)


def compare(a: float, b: float) -> str:
    """比较两个浮点数的大小"""
    return f"{a} > {b}" if a > b else f"{b} > {a}" if b > a else f"{a} == {b}"


def count_letter_in_string(a: str, b: str) -> str:
    """统计字符串中某个字母/字符的出现次数。a是被搜索的字符串，b是要统计的字符"""
    count = a.lower().count(b.lower())
    return f"字符 '{b}' 在字符串中出现了 {count} 次"


def search_wikipedia(query: str) -> str:
    """在维基百科中搜索指定查询的前三个页面摘要"""
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[:3]:
        try:
            wiki_page = wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"页面: {page_title}\n摘要: {wiki_page.summary}")
        except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError):
            pass
    return "\n\n".join(summaries) if summaries else "维基百科没有搜索到合适的结果"


def get_current_temperature(latitude: float, longitude: float) -> str:
    """获取指定经纬度位置的当前温度（数据来源：Open-Meteo 免费天气 API）"""
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m',
        'forecast_days': 1,
    }
    response = requests.get("https://api.open-meteo.com/v1/forecast", params=params)
    if response.status_code != 200:
        return f"天气 API 请求失败，状态码: {response.status_code}"

    results = response.json()
    current_utc_time = datetime.datetime.now(datetime.UTC)
    time_list = [datetime.datetime.fromisoformat(t).replace(tzinfo=datetime.timezone.utc)
                 for t in results['hourly']['time']]
    temperatures = results['hourly']['temperature_2m']
    closest_idx = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))
    return f"现在温度是 {temperatures[closest_idx]}°C"


# ========== 扩展工具一：计算器 ==========

def calculator(expression: str) -> str:
    """计算数学表达式。参数 expression 是一个数学表达式字符串，如 '3 + 5 * 2'。支持 + - * / 和括号"""
    try:
        result = eval(expression, {"__builtins__": None}, {})
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算失败: {str(e)}"


def power(base: float, exponent: float) -> str:
    """计算 base 的 exponent 次方"""
    return f"{base} 的 {exponent} 次方 = {base ** exponent}"


def sqrt(number: float) -> str:
    """计算一个数的平方根"""
    import math
    if number < 0:
        return f"错误：不能对负数 {number} 开平方"
    return f"{number} 的平方根 = {math.sqrt(number)}"


# ========== 扩展工具二：文件读取器 ==========

def read_file(file_path: str) -> str:
    """读取指定文件的内容。请提供文件的完整路径。仅支持 .txt 和 .md 文件"""
    if not os.path.exists(file_path):
        return f"错误：文件 '{file_path}' 不存在"
    if not file_path.endswith(('.txt', '.md')):
        return f"错误：不支持的文件类型，仅支持 .txt 和 .md"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # 截断到 2000 字符防止 token 爆炸
        if len(content) > 2000:
            content = content[:2000] + f"\n\n... (文件共 {len(content)} 字符，此处仅显示前 2000)"
        return content
    except Exception as e:
        return f"读取失败: {str(e)}"


# ========== 扩展工具三：字符串处理 ==========

def reverse_string(text: str) -> str:
    """反转字符串。例如输入 'hello' 返回 'olleh'"""
    return f"反转结果: {text[::-1]}"


def word_count(text: str) -> str:
    """统计文本中的字符数、单词数和行数"""
    char_count = len(text)
    word_count = len(text.split())
    line_count = text.count('\n') + 1
    return f"字符数: {char_count}, 单词数: {word_count}, 行数: {line_count}"


def to_uppercase(text: str) -> str:
    """将文本转换为大写"""
    return text.upper()


# ========== 扩展工具四：随机数生成器 ==========

def random_number(min_val: float, max_val: float) -> str:
    """生成 min_val 到 max_val 之间的随机浮点数"""
    result = random.uniform(min_val, max_val)
    return f"随机数: {result:.4f}"


def random_integer(min_val: int, max_val: int) -> str:
    """生成 min_val 到 max_val 之间的随机整数（包含两端）"""
    result = random.randint(min_val, max_val)
    return f"随机整数: {result}"
