# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import time
import json
import re
import numpy as np
import pandas as pd
import io
import sys
import requests
import random
import ast
import os
from collections import defaultdict


def extract_python_code(prediction):
    """使用正则表达式提取三反引号包裹的Python代码"""
    pattern = r"```python(.*?)```"
    code_blocks = re.findall(pattern, prediction, re.DOTALL)
    code = "\n".join(code_blocks)
    return code


def execute_python_code(code):
    # python exec API Server
    BASE_URL = ""
    code_data={"code": code}
    max_try = 5
    while max_try>0:
        try:
            response = requests.post(f"{BASE_URL}/python", json=code_data)
            return response.json()['result'], response.json()['error_message']
        except Exception as e:
            max_try -= 1
            print("Warning! exec python api error.............\n{}".format(e))
            time.sleep(3)
    error_message = f"调用api失败"
    return None, error_message


def json_default(obj):
    """自定义 JSON 序列化函数"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    else:
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def extract_filepaths_from_code(code: str) -> list:
    """Extract file paths from pd.read_csv calls in the code using AST"""
    
    # Note: Code cleaning is now handled by the caller (compute_score)
    # We operate directly on the provided code string.
    
    paths = []
    
    class FilePathVisitor(ast.NodeVisitor):
        def __init__(self):
            self.paths = []
            self.assignments = {} # variable_name -> string_value

        def _resolve_node_value(self, node):
            """Helper to resolve a node to a string value if possible."""
            try:
                if isinstance(node, ast.Constant): # python 3.8+
                    return node.value
                elif isinstance(node, ast.Str): # python < 3.8
                    return node.s
                elif isinstance(node, ast.Name):
                    return self.assignments.get(node.id, None)
                elif isinstance(node, ast.JoinedStr):
                    # Try to resolve f-string: f"foo" or f"{var}"
                    parts = []
                    for val in node.values:
                        res = self._resolve_node_value(val)
                        if res is not None:
                            parts.append(str(res))
                        else:
                            return None # Cannot resolve full string
                    return "".join(parts)
                elif isinstance(node, ast.FormattedValue):
                    return self._resolve_node_value(node.value)
                elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                    # Handle string concatenation: "a" + "b"
                    left = self._resolve_node_value(node.left)
                    right = self._resolve_node_value(node.right)
                    if left is not None and right is not None:
                        return str(left) + str(right)
            except Exception:
                return None
            return None

        def visit_Assign(self, node):
            # Handle var = ...
            val = self._resolve_node_value(node.value)
            if val is not None:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.assignments[target.id] = val
            self.generic_visit(node)
            
        def visit_AnnAssign(self, node):
             # Handle var: type = ...
             if isinstance(node.target, ast.Name) and node.value:
                 val = self._resolve_node_value(node.value)
                 if val is not None:
                     self.assignments[node.target.id] = val
             self.generic_visit(node)

        def visit_Call(self, node):
            # Check for pd.read_csv(...) or pandas.read_csv(...)
            is_read_csv = False
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == 'read_csv':
                    # Check if called on 'pd' or 'pandas' (heuristic)
                    if isinstance(node.func.value, ast.Name):
                        if node.func.value.id in ['pd', 'pandas']:
                            is_read_csv = True
            
            if is_read_csv:
                # Check args[0]
                found_arg = None
                if node.args:
                    found_arg = node.args[0]
                
                # Check keywords (filepath_or_buffer)
                if not found_arg:
                    for keyword in node.keywords:
                        if keyword.arg == 'filepath_or_buffer':
                            found_arg = keyword.value
                            break
                
                if found_arg:
                    val = self._resolve_node_value(found_arg)
                    if val is not None:
                        self.paths.append(val)
                    elif isinstance(found_arg, ast.Name):
                         # If variable is not found in local assignments
                         # print(f"[DEBUG] Variable '{found_arg.id}' used in read_csv but not found in assignments")
                         pass
            
            self.generic_visit(node)

    try:
        # print(f"[DEBUG] Parsing code for AST:\n{code}")
        tree = ast.parse(code)
        visitor = FilePathVisitor()
        visitor.visit(tree)
        paths = visitor.paths
    except Exception as e:
        print(f"AST parsing failed: {e}, falling back to regex")
        pass
        
    # Always run regex as backup/supplement
    # Regex for direct string: pd.read_csv("path") or pd.read_csv(f"path") or pd.read_csv(r"path")
    # Allow optional f or r or fr prefix before quotes
    pattern_direct = r"pd\.read_csv\s*\(\s*[frFR]*['\"]([^'\"]+)['\"]"
    paths.extend(re.findall(pattern_direct, code))
    
    # Regex for simple variable assignment (heuristic fallback)
    # var = 'path' ... pd.read_csv(var)
    # 1. Find all pd.read_csv(var_name)
    pattern_var_usage = r"pd\.read_csv\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)"
    used_vars = re.findall(pattern_var_usage, code)
    for var in used_vars:
        # 2. Find last assignment to this var: var = 'path'
        # Handle var = f"..." or var = r"..."
        pattern_assign = r"{}\s*=\s*[frFR]*['\"]([^'\"]+)['\"]".format(re.escape(var))
        assigned_vals = re.findall(pattern_assign, code)
        if assigned_vals:
            paths.append(assigned_vals[-1]) # Take the last one found

    final_paths = list(set(paths))
    print(f"[DEBUG] Extracted paths: {final_paths} from code length {len(code)}")
    return final_paths

def compute_filepath_reward(code_filepaths, gt_filepaths) -> float:
    """
    Compute reward based on overlap between used filepaths and ground truth filepaths.
    Max score: 0.5 (all GT paths used)
    Min score: 0.0 (no GT paths used)
    """
    if not gt_filepaths:
        return 0.5  # If no files are required, give full score
    
    if not code_filepaths:
        return 0.0

    # Normalize paths: use basenames to avoid relative/absolute path mismatches
    # e.g. "data/table.csv" vs "/tmp/data/table.csv" should match
    code_paths_set = set(os.path.basename(p.strip()) for p in code_filepaths)
    gt_paths_set = set(os.path.basename(p.strip()) for p in gt_filepaths)
    
    # Calculate intersection based on BASENAMES
    matched = code_paths_set.intersection(gt_paths_set)
    recall = len(matched) / len(gt_paths_set) if gt_paths_set else 0.0
    
    return 0.5 * recall

def code_format_eval(prediction):
    code = extract_python_code(prediction)
    if code and "print(" in code and "pd.read_csv" in code:
        return True
    else:
        return False

def code_exec_result(code):
    print('-'* 50)
    # print("Cleaned Code:\n{}".format(code)) # Printed in compute_score now
    result, error_message = execute_python_code(code)
    print("Exec result:{}".format(result))
    print("Exec error:{}".format(error_message))
    print('-'* 50)
    return result, error_message


def api_reward_model(question, model_output, ref_answer):
    from openai import OpenAI

    prompt = "你是一个评判助手，你的任务是根据问题和提供的标准答案来评估其他答案的正确性，判断的标准是，其他答案跟标准答案在关键结果上是否一致，" \
             "如果一致输出1，否则输出0，除此外不要输出其他内容。\n问题：{}\n标准答案：\n{}\n其他答案：\n{}"
    if "</think>" in model_output:
        content = prompt.format(question, ref_answer, model_output.split("</think>")[-1])
    else:
        content = prompt.format(question, ref_answer, model_output)

    openai_api_key = ""
    openai_api_base = ""
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base, )

    def predict(query):
        max_try = 5
        max_tokens = 8192
        while max_try>0:
            try:
                model_name = client.models.list().data[0].id
                print("model_name", model_name)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=query,
                    temperature=0.3,
                    top_p=0.95,
                    max_tokens=max_tokens,
                    extra_body={
                        "repetition_penalty": 1.01,
                        "skip_special_tokens": False,
                        "spaces_between_special_tokens": False
                    },
                )
                max_try = -1
                return response.choices[0].message.model_dump()["content"]
            except:
                max_try -= 1
                time.sleep(60)
                print("Warning! reward api error.............")
        print("***inference failed***")
        return ''

    messages = [{"role": "user", "content": content}]
    answer = predict(messages)
    # print("reward model output: {}".format(answer))
    return answer


def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.

    Args:
        processed_str: Processed response string from the model

    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    # print("\n[Structure Validation]")
    validation_passed = True

    # check code format
    if validation_passed:
        processed_str = processed_str.split("</think>")[-1]
        validation_passed = code_format_eval(processed_str)

    return validation_passed


def compute_score(solution_str, ground_truth, extra_info=None) -> float:
    """
    get rule based score
    :param solution_str:
    :param ground_truth:
    :param prompt_str:
    :param format_reward:
    :return: Float score
    """

    question = extra_info["question"]
    gt_filepath_list = extra_info.get("filepath_list", [])

    format_score = -1.0
    execute_score = 0.0
    answer_score = 0.0
    codebleu_score = 0.0
    filepath_score = 0.0

    print("Solution:\n{}".format(solution_str))

    # Validate response structure
    # 时间戳、step、各分项指标、服务调用时间、格式[Question]
    format_correct = validate_response_structure(solution_str)

    # new_samples = []
    # critic_sample, rethink_sample, compare_sample = None, None, None
    code_str = None
    if format_correct:  # format_correct
        format_score = -0.5
        try:
            code_str = solution_str.split("</think>")[-1].replace("<|im_end|>", "").replace("<｜end▁of▁sentence｜>", "").replace("<_end>", "").replace("<|endoftext|>", "").strip()
            
            # 1. Clean Code
            cleaned_code = extract_python_code(code_str)
            print('-'* 50)
            print("Cleaned Code:\n{}".format(cleaned_code))

            # 2. Compute filepath reward
            # Use cleaned code if available, otherwise fallback to raw code (for recall)
            path_source = cleaned_code if cleaned_code else code_str
            if not cleaned_code:
                print("[DEBUG] No code blocks found, extracting paths from raw text.")
                
            code_filepaths = extract_filepaths_from_code(path_source)
            filepath_score = compute_filepath_reward(code_filepaths, gt_filepath_list)
            
            # 3. Execute
            exec_result, error_message = code_exec_result(cleaned_code)
            
            if exec_result:  # execute_correct
                execute_score = 0.5
                if '1' in api_reward_model(question, exec_result, ground_truth): # answer correct
                # if random.random() > 0.5: # for quick debug
                    answer_score = 3.0
                    codebleu_score = 0.0   # 正确答案，codebleu设为1
                else:
                    answer_score = 0.0
                    codebleu_score = 0.0 
            else:
                execute_score = 0.0
                codebleu_score = 0.0

        except Exception as e:
            print(e)
    else:
        format_score = -1.0
        codebleu_score = 0.0
        filepath_score = 0.0

    total_score = format_score + execute_score + answer_score + codebleu_score + filepath_score
    # Max score breakdown: -0.5 (format) + 0.5 (exec) + 3.0 (answer) + 0.5 (filepath) = 3.5
    # We require strictly correct execution and file usage for ACC.
    acc = total_score >= 3.0 
    # dict_scores = {'format': format_score, 'execute': execute_score, 'answer': answer_score}

    return {'score': total_score, 'format': format_score, 'execute': execute_score, 'answer': answer_score,'codebleu': codebleu_score, 'filepath': filepath_score, 'acc': acc}  # total_score, dict_scores