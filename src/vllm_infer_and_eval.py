import json
from openai import AsyncOpenAI, OpenAI
import asyncio
import argparse
from tqdm import tqdm
import time
import re
import numpy as np
import pandas as pd
import io
import sys
import yaml
import time
import vllm 
from transformers import AutoTokenizer


'''
python vllm_infer_and_eval.py --judge_model_name "openchat" --judge_api_key "EMPTY" \
--judge_api_url "http://10.244.14.102:8000/v1" \
--input_file "/gemini/space/private/panchangzai/table2text/TablePipeline/data/yllm_infer/table_area_result_deepseek-R1-gwen32B-test.jsonl" \
--output_file "/gemini/space/private/panchangzai/table2text/TablePipeline/data/vllm_infer/table_area_result_deepseek-R1-gwen32B-test-output.jsonl"
'''

class LLM:
    def __init__(self, model_config):
        """
        初始化LLM对象，设置模型名称、API密钥和基础URL。
        """
        self.model_name = model_config['model_name']
        self.api_key = model_config['api_key']
        self.base_url = model_config['base_url']
        self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_response(self, prompt: str):
        """
        根据任务类型生成对应的响应。
        task_type: str - 任务类型 ('report', 'extraction', 'qa')
        query: str - 查询内容或分析角度
        data: str - 表格数据
        """
        if isinstance(prompt, str):
            messages = [{"role": "system", "content": ''},
                        {"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            print('wrong input format!')

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0,
                messages=messages
            )
            time.sleep(1)
            # 逐步输出结果
            result = json.loads(completion.model_dump_json())
            response_content = result['choices'][0]['message']['content']
            return response_content

        except Exception as e:
            print(f"stream_response 调用出错: {e}")
            return ''

    async def generate_response_async(self, prompt: str):
        """
        根据任务类型生成对应的响应。
        task_type: str - 任务类型 ('report', 'extraction', 'qa')
        query: str - 查询内容或分析角度
        data: str - 表格数据
        """
        if isinstance(prompt, str):
            messages = [{"role": "system", "content": ''},
                        {"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            print('wrong input format!')

        try:
            completion = await self.async_client.chat.completions.create(
                model=self.model_name,
                temperature=0,
                messages=messages
            )

            # 逐步输出结果
            result = json.loads(completion.model_dump_json())
            response_content = result['choices'][0]['message']['content']
            return response_content

        except Exception as e:
            print(f"stream_response 调用出错: {e}")
            return ''


def extract_python_code(prediction):
    """使用正则表达式提取三反引号包裹的Python代码"""
    pattern = r"```python(.*?)```"
    code_blocks = re.findall(pattern, prediction, re.DOTALL)
    code = "\n".join(code_blocks)
    return code


def execute_python_code(code):
    """执行Python代码并获取最后的输出作为答案"""
    # if "file_path =" in code:
    #     code = code.replace("file_path =", "file_path ='/gemini/space/private/panchangzai/table2text/Logic-RL/data/table/MiMoTable_with_json' + ")
    # else:
    #     code = code.replace("pd.read_csv('", """pd.read_csv('/gemini/space/private/panchangzai/table2text/Logic-RL/data/table/MiMoTable_with_json""")
    # 地址需要修改
    # code = code.replace("/gemini-1/space/space/private/pengjiaxin/projects/Logic-RL/data/table/MiMoTable_with_json",
    #                     "D:\PycharmProjects\TableTaskEval\data\MiMoTable_with_json")
    print("Cleaned code:{}".format(code))
    local_path_clean = "/gemini-1/space/space/private/swc/TableTaskEval/data/origin_table"
    code = f"import os\nos.chdir('{local_path_clean}')\n" + code
    try:
        # 重定向标准输出
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        # 执行代码
        local_vars = {}
        exec(code, local_vars)

        # 获取标准输出结果
        output = buffer.getvalue().strip()
        sys.stdout = old_stdout

        if output:
            return output, None
        else:
            return None, "执行代码后没有任何输出。"

    except Exception as e:
        sys.stdout = old_stdout
        error_message = f"代码执行失败: {e}"
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


def code_format_eval(prediction):
    code = extract_python_code(prediction)
    if code and "print(" in code and "pd.read_csv" in code:
        return True

    else:
        return False


def code_exec_result(prediction):
    prediction = extract_python_code(prediction)
    print("\nCleaned code:{}".format(prediction))
    result, error_message = execute_python_code(prediction)
    print("Exec result:{}".format(result))
    print("Exec error:{}".format(error_message))
    return result, error_message


def api_reward_model(llm, question, model_output, ref_answer):

    prompt = "你是一个评判助手，你的任务是根据问题和提供的标准答案来评估其他答案的正确性，判断的标准是，其他答案跟标准答案在关键结果上是否一致，" \
             "如果一致输出1，否则输出0，除此外不要输出其他内容。\n问题：{}\n标准答案：\n{}\n其他答案：\n{}"
    if "</think>" in model_output:
        content = prompt.format(question, ref_answer, model_output.split("</think>")[-1])
    else:
        content = prompt.format(question, ref_answer, model_output)

    def predict(query):
        max_try = 5
        max_tokens = 8192
        while max_try>0:
            try:
                max_try = -1
                result = llm.generate_response(query)
                return result
            except:
                max_tokens -= 1000
                max_try -= 1
                time.sleep(60)
                print("Warning! reward api error.............")
        print("***inference failed***")
        return ''

    messages = [{"role": "user", "content": content}]
    answer = predict(messages)
    print("llm judge output: {}".format(answer))
    return answer


def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.

    Args:
        processed_str: Processed response string from the model

    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)

        print(f"  {tag_str}: count={count}, position={pos}")

        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    positions['final_end'] = len(processed_str)-1
    positions['think_length'] = positions['think_end'] - positions['think_start']
    positions['answer_length'] = positions['final_end'] - positions['think_start']

    # Verify tag order
    # if positions['think_start'] > positions['think_end'] or \
    #         (positions['answer_length'] > positions['think_length']):
    if positions['think_start'] > positions['think_end']:
        print("  [Error] Incorrect tag order: Expected <think>...</think>")
        validation_passed = False
    else:
        print("  Tag sequence validation passed")

    # check code format
    if validation_passed:
        processed_str = processed_str.split("</think>")[-1]
        validation_passed = code_format_eval(processed_str)

    return validation_passed


def compute_score(llm, solution_str, ground_truth, question) -> float:
    """

    :param solution_str:
    :param ground_truth:
    :return:
    """
    # print("\n" + "=" * 80)
    # print(" Processing New Sample ".center(80, '='))

    # try:
    #     if "<｜Assistant｜>" in solution_str:
    #         question = solution_str.split("输入问题：\n")[1].split("<｜Assistant｜>")[0]
    #     else:
    #         question = solution_str.split("输入问题：\n")[1].split("<|im_end|>")[0]
    # except Exception as e:
    #     print('切分question报错': e)

    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<_bot>" in solution_str:
        solution_str = solution_str.split("<_bot>", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    elif "<｜Assistant｜>" in solution_str:
        solution_str = solution_str.split("<｜Assistant｜>", 1)[1]
    else:
        print("[Error] Failed to locate model response header")
        solution_str = solution_str

    print(f"\n[Question]\n{question}")
    print(f"\n[Model Response]\n{solution_str}")
    print("********")
    print(f"\n[Ground Truth]\n{ground_truth}")

    # Validate response structure
    format_correct = validate_response_structure(solution_str)

    execute_suceess = False
    answer_correct = False
    exec_result, error_message = '', ''
    # if format_correct:
    try:
        code_str = solution_str.split("</think>")[-1].replace("<|im_end|>", "").replace("<｜end▁of▁sentence｜>", "").strip()
        exec_result, error_message = code_exec_result(code_str)
        if exec_result:
            judge_output = api_reward_model(llm, question, exec_result, ground_truth)
            answer_correct = True if '1' in judge_output else False
            execute_suceess = True
        else:
            execute_suceess = False

    except Exception as e:
        print(e)
    
    return format_correct, execute_suceess, answer_correct, exec_result, error_message


def apply_chat_template_design(prompt, tokenizer):

    # # no think模式
    new_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        enable_thinking=True,
        tokenize=False
    )

    # prompt + think
    # reasoning_output = prompt.split('<think>')[1].split('</think>')[0]
    # origin_prompt = prompt.split('<think>')[0]
    # new_prompt = tokenizer.apply_chat_template(
    #     [{"role": "user", "content": origin_prompt}],
    #     add_generation_prompt=True,
    #     enable_thinking=True,
    #     tokenize=False
    # ) + '<think>' + reasoning_output + '</think>\n\n'

    return new_prompt


def vllm_batch_infer(prompts, model_path):

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 模板拼接
    formatted_prompts = [apply_chat_template_design(prompt, tokenizer) for prompt in prompts]
    print(formatted_prompts[0])

    llm = vllm.LLM(
        model=model_path,                # 模型路径或HuggingFace ID
        tensor_parallel_size=2,                # GPU并行数（单卡设为1）
        max_model_len=40960,                    # 最大上下文长度
        gpu_memory_utilization=0.95            # 显存利用率控制[2,7](@ref)
    )

    # 2. 定义采样参数
    sampling_params = vllm.SamplingParams(
        temperature=0.6,                      # 温度系数（0-2）
        top_p=0.95,                           # 核采样概率
        top_k=20,
        max_tokens=8192,                      # 生成最大token数
        presence_penalty=1.2                  # 重复惩罚系数[2,7](@ref)
    )

    outputs = llm.generate(formatted_prompts, sampling_params)
    # outputs = llm.generate(['<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n<think>我觉得应该回复你好</think>'], sampling_params)

    return [output.outputs[0].text for output in outputs]


# 推理改成离线模式
def vllm_infer_and_eval(input_file, output_file, input_key, output_key, llm_infer_model_path, llm_judge):
    """
    从输入的 JSONL 文件中读取数据，调用 OpenAI API 处理 prompt 部分，
    并将结果保存到新的 JSONL 文件中。
    """
    print('推理模型名称：', llm_infer_model_path)
    # matrix = []

    start_time = time.perf_counter()

    prompts = []
    with open(input_file, "r", encoding="utf-8") as infile:
        for ix, line in tqdm(enumerate(infile)):
            data = json.loads(line)
            prompts.append(data[input_key])

    results = vllm_batch_infer(prompts, llm_infer_model_path)
    infer_time = time.perf_counter() - start_time

    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for ix, line in tqdm(enumerate(infile)):
            print("\n" + "=" * 80)
            print(f"Processing New Sample {ix}".center(80, '='))

            result = results[ix]
            data = json.loads(line)
            data[output_key] = result
            ground_truth = data['gold_truth']
            question = data['question']
            print(f"\n[Question]\n{question}")
            solution_str = result
            data['format_correct'], data['execute_suceess'], data['answer_correct'], data['exec_result'], data['error_message'] = compute_score(llm_judge, solution_str, ground_truth, question)
            # matrix.append([data[key] for key in ['spreedsheetpath_list', 'question', 'input', 'output', 'question_difficulty', 
                                                #  'table_difficulty', 'execute_suceess', 'answer_correct', 'exec_result', 'error_message']])  # 表格路径、问题、输入、输出、问题类型、表格类型、format成功、执行成功、答案正确
            print(f"\n[Is Format Correct]\n{data['format_correct']}")
            print(f"\n[Is Execute Suceess]\n{data['execute_suceess']}")
            print(f"\n[Execute Result]\n{data['exec_result']}")
            print(f"\n[Error Message]\n{data['error_message']}")
            print(f"\n[Ground Truth]\n{ground_truth}")
            print(f"\n[Is Answer Correct]\n{data['answer_correct']}")
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
    eval_time = time.perf_counter() - start_time - infer_time

    with open(output_file, "r", encoding="utf-8") as outfile:
        total_num = 0
        acc_count = [0, 0, 0]
        for ix, line in tqdm(enumerate(outfile)):
            total_num += 1
            data = json.loads(line)
            acc_count[0] += data['format_correct']
            acc_count[1] += data['execute_suceess']
            acc_count[2] += data['answer_correct']
        print(f'Format Acc: {acc_count[0]/total_num}')
        print(f'Execute Suceess Rate: {acc_count[1]/total_num}')
        print(f'Answer Acc: {acc_count[2]/total_num}')
    print(infer_time, eval_time)


# async def vllm_infer_batch(input_file, output_file, input_key, output_key, batch_size, llm):
#     """
#     从输入的 JSONL 文件中读取数据，调用 OpenAI API 处理 prompt 部分，
#     并将结果保存到新的 JSONL 文件中。
#     """
#     batch = []
#     with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
#         for line in tqdm(infile):
#             data = json.loads(line)
#             # data = {input_key: '你好'}
#             batch.append(data)
#             if len(batch) == batch_size:
#                 tasks = [llm.generate_response(data[input_key]) for data in batch]
#                 results = await asyncio.gather(*tasks)
#                 for result, data in zip(results, batch):
#                     data[output_key] = result  # 输出结果添加到output_key中，其他字段不变
#                     outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
#                 outfile.flush()  # 将缓存内容存入文件中
#                 batch = []
#         if batch:  # 处理最后一个批次
#             tasks = [llm.generate_response(data[input_key]) for prompt in batch]
#             results = await asyncio.gather(*tasks)
#             for result, data in zip(results, batch):
#                 data[output_key] = result
#                 outfile.write(json.dumps(data, ensure_ascii=False) + "\n")


def main():

    parser = argparse.ArgumentParser(description="Infer Tables.")
    parser.add_argument("--input_file", type=str, default=r"/gemini-1/space/space/private/zhangjie/table-reasoning/data/v6_0521/test/v6_test_0521.jsonl", help="Input JSONL file path")
    # parser.add_argument("--input_file", type=str, default=r"/gemini-1/space/space/private/pcz/projects/TableTaskEval/data/rl_v6_train_three_part_format_50.json", help="Input JSONL file path")  # 测试集合地址：输入jsonl
    # parser.add_argument("--input_file", type=str, default=r"/gemini-1/space/space/private/pcz/projects/TableTaskEval/data/rl_v6_train_three_part_format_50.json", help="Input JSONL file path")  # 测试集合地址：输入jsonl
    # parser.add_argument("--output_file", type=str, default=r"/gemini-1/space/space/private/pcz/projects/TableTaskEval/output/rl_v6_train_three_part_format_50_output.json", help="Output JSONL file path")  # 输出地址：输出jsonl
    parser.add_argument("--output_file", type=str, default=r"/gemini-1/space/space/private/pcz/projects/TableTaskEval/output/qwen3-8b-instruct-sft.json", help="Output JSONL file path")  # 输出地址：输出jsonl
    parser.add_argument("--input_key", type=str, default='input', help="Input key in JSONL")  # 测试集中prompt在jsonl文件对应的key
    parser.add_argument("--output_key", type=str, default='prediction', help="Output key in JSONL")  # 输出jsonl文件中，推理内容对应的key
    parser.add_argument("--llm_infer_model_path", type=str, default='/gemini-1/space/space/private/pcz/projects/ckpts/qwen3-8b-instruct-sft-qwen3-32b-v6/checkpoint-240', help="Output key in JSONL")  # 输出jsonl文件中，推理内容对应的key

    args = parser.parse_args()

    with open('model_config.yaml', 'r', encoding='utf-8') as file:
        model_config = yaml.safe_load(file)
        judge_model_config = model_config['judge_model']
        # infer_model_config = model_config['infer_model']

    # llm_infer = LLM(infer_model_config)
    llm_judge = LLM(judge_model_config)
    vllm_infer_and_eval(args.input_file, args.output_file, args.input_key, args.output_key, args.llm_infer_model_path, llm_judge)
    
    # vllm_infer_and_eval(args.input_file, args.output_file, args.input_key, args.output_key, llm_infer, llm_judge)

if __name__ == '__main__':
    main()
