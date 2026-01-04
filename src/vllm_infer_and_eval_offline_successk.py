import json
import os
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
import builtins
import sys

# Try to import vllm and transformers, but don't fail if not present (for API-only environments)
try:
    import vllm
    from transformers import AutoTokenizer
except ImportError:
    vllm = None
    AutoTokenizer = None
    print("Warning: vllm or transformers not installed. Local inference will not work.")

'''
python vllm_infer_and_eval.py --judge_model_name "openchat" --judge_api_key "EMPTY" --judge_api_url "http://10.244.14.102:8000/v1" --input_file "/gemini/space/private/panchangzai/table2text/TablePipeline/data/yllm_infer/table_area_result_deepseek-R1-gwen32B-test.jsonl" --output_file "/gemini/space/private/panchangzai/table2text/TablePipeline/data/vllm_infer/table_area_result_deepseek-R1-gwen32B-test-output.jsonl"
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
        # self.model_name = self.client.models.list().data[0].id
        self.total_tokens = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

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
            return ''

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0,
                messages=messages
            )
            # time.sleep(1)
            # 逐步输出结果
            result = json.loads(completion.model_dump_json())

            # Record usage
            if hasattr(completion, 'usage') and completion.usage:
                self.total_tokens['prompt_tokens'] += completion.usage.prompt_tokens
                self.total_tokens['completion_tokens'] += completion.usage.completion_tokens
                self.total_tokens['total_tokens'] += completion.usage.total_tokens

            response_content = result['choices'][0]['message']['content']
            return response_content

        except Exception as e:
            print(f"generate_response 调用出错: {e}")
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
            return ''

        try:
            completion = await self.async_client.chat.completions.create(
                model=self.model_name,
                temperature=0,
                messages=messages
            )

            # 逐步输出结果
            result = json.loads(completion.model_dump_json())

            # Record usage
            if hasattr(completion, 'usage') and completion.usage:
                self.total_tokens['prompt_tokens'] += completion.usage.prompt_tokens
                self.total_tokens['completion_tokens'] += completion.usage.completion_tokens
                self.total_tokens['total_tokens'] += completion.usage.total_tokens

            response_content = result['choices'][0]['message']['content']
            return response_content

        except Exception as e:
            print(f"generate_response_async 调用出错: {e}")
            return ''


def extract_python_code(prediction):
    """使用正则表达式提取三反引号包裹的Python代码"""
    pattern = r"```python(.*?)```"
    code_blocks = re.findall(pattern, prediction, re.DOTALL)
    code = "\n".join(code_blocks)
    return code


def execute_python_code(code, table_dir):
    """执行Python代码并获取最后的输出作为答案"""
    # if "file_path =" in code:
    #     code = code.replace("file_path =", "file_path ='/gemini/space/private/panchangzai/table2text/Logic-RL/data/table/MiMoTable_with_json' + ")
    # else:
    #     code = code.replace("pd.read_csv('", """pd.read_csv('/gemini/space/private/panchangzai/table2text/Logic-RL/data/table/MiMoTable_with_json""")
    # 地址需要修改
    # code = code.replace("/gemini-1/space/space/private/pengjiaxin/projects/Logic-RL/data/table/MiMoTable_with_json",
    #                     "D:\PycharmProjects\TableTaskEval\data\MiMoTable_with_json")
    print("Cleaned code:{}".format(code))
    class ExitIntercepted(Exception):
        pass
    def fake_exit(*args, **kwargs):
        raise ExitIntercepted("exit() or sys.exit() was called.")
    # 替换 exit 和 sys.exit
    builtins.exit = fake_exit
    sys.exit = fake_exit
    # local_path_clean = "/gemini/space/private/swc/TableTaskEval/data/origin_table"
    
    # Use the provided table directory
    local_path_clean = os.path.abspath(table_dir)
    full_code = f"import os\nos.chdir(r'{local_path_clean}')\n" + code
    try:
        # 重定向标准输出
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        # 执行代码
        local_vars = {'__name__': '__main__'}
        exec(full_code, local_vars)

        # 获取标准输出结果
        output = buffer.getvalue().strip()
        sys.stdout = old_stdout

        if output:
            return output, None
        else:
            return None, "执行代码后没有任何输出。"
    except ExitIntercepted as e:
        sys.stdout = old_stdout
        error_message = f"代码执行失败, 遇到exit(): {e}"
        return None, error_message
    except Exception as e:
        sys.stdout = old_stdout
        print(f"DEBUG: Full code causing error:\n{full_code}")
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


def code_exec_result(prediction, table_dir):
    prediction = extract_python_code(prediction)
    print("\nCleaned code:{}".format(prediction))
    result, error_message = execute_python_code(prediction, table_dir)
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
        max_tokens = 8000
        while max_try>0:
            try:
                max_try = -1
                result = llm.generate_response(query)
                return result
            except:
                # max_tokens -= 1000
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


def compute_score(llm, solution_str, ground_truth, question, table_dir) -> float:
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
        if "</think>" in solution_str:
            code_str = solution_str.split("</think>")[-1].replace("<|im_end|>", "").replace("<｜end▁of▁sentence｜>", "").strip()
        else:
            code_str = solution_str.replace("<|im_end|>", "").replace("<｜end▁of▁sentence｜>", "").strip()
            
        exec_result, error_message = code_exec_result(code_str, table_dir)
        if exec_result:
            judge_output = api_reward_model(llm, question, exec_result, ground_truth)
            answer_correct = True if '1' in judge_output else False
            execute_suceess = True
        else:
            execute_suceess = False

    except Exception as e:
        print(e)
    
    return format_correct, execute_suceess, answer_correct, exec_result, error_message


# def apply_chat_template_design(prompt, tokenizer):

#     # # think or no think模式
#     # new_prompt = tokenizer.apply_chat_template(
#     #     [{"role": "user", "content": prompt}],
#     #     add_generation_prompt=True,
#     #     enable_thinking=True,
#     #     tokenize=False
#     # )
def apply_chat_template_design(prompt, tokenizer):
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            enable_thinking=True,
            tokenize=False
        )
    except Exception as e:
        print("[Warning] 使用 chat_template 失败，fallback 到手动拼接")
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


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


def vllm_batch_infer(prompts, model_path, sampling_config):
    if vllm is None:
        raise ImportError("vllm is not installed, cannot run local inference.")
        
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 定义最大输入 token 长度（必须 < max_model_len）
    max_prompt_tokens = 31000

    def truncate_prompt(prompt_str):
        input_ids = tokenizer(prompt_str, return_tensors='pt').input_ids[0]
        if len(input_ids) > max_prompt_tokens:
            input_ids = input_ids[:max_prompt_tokens]
            return tokenizer.decode(input_ids, skip_special_tokens=True)
        return prompt_str
    # 模板拼接
    # formatted_prompts = [apply_chat_template_design(prompt, tokenizer) for prompt in prompts]
    # 应用模板并截断
    formatted_prompts = [
        truncate_prompt(apply_chat_template_design(prompt, tokenizer))
        for prompt in prompts
    ]
    print(formatted_prompts[0])

    llm_engine = vllm.LLM(
        model=model_path,                # 模型路径或HuggingFace ID
        tensor_parallel_size=8,                # GPU并行数（单卡设为1）
        max_model_len=sampling_config['max_model_len'],                    # 最大上下文长度
        gpu_memory_utilization=0.85            # 显存利用率控制[2,7](@ref)
    )

    # 2. 定义采样参数
    sampling_params = vllm.SamplingParams(
        temperature=sampling_config['temperature'],                      # 温度系数（0-2）
        top_p=sampling_config['top_p'],                           # 核采样概率
        top_k=sampling_config['top_k'],
        max_tokens=8000,                      # 生成最大token数
        presence_penalty=sampling_config['presence_penalty']                  # 重复惩罚系数[2,7](@ref)
    )

    outputs = llm_engine.generate(formatted_prompts, sampling_params)
    # outputs = llm.generate(['<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n<think>我觉得应该回复你好</think>'], sampling_params)

    return [output.outputs[0].text for output in outputs]

async def api_infer_batch(prompts, llm_client, max_concurrency=10):
    """
    Batch inference using API (Async) with Semaphore.
    """
    sem = asyncio.Semaphore(max_concurrency)
    
    async def worker(prompt, pbar):
        async with sem:
            try:
                res = await llm_client.generate_response_async(prompt)
            except Exception as e:
                print(f"Error in api_infer_batch: {e}")
                res = ""
            finally:
                pbar.update(1)
                if hasattr(llm_client, 'total_tokens'):
                    pbar.set_postfix(tokens=llm_client.total_tokens['total_tokens'])
            return res

    pbar = tqdm(total=len(prompts), desc="API Inference")
    tasks = [worker(prompt, pbar) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    pbar.close()
    
    return results


def infer_and_eval(input_file, output_file, input_key, output_key, infer_mode, llm_infer_source, llm_judge, sampling_config, table_dir, k=1, max_concurrency=10):
    """
    从输入的 JSONL 文件中读取数据，调用 vLLM 或 API 处理 prompt 部分，
    并将结果按样本ID分组保存到新的 JSONL 文件中。
    k: 每个样本的推理次数
    """
    if infer_mode == 'local':
        print('推理模型路径：', llm_infer_source)
    else:
        print('推理模型API配置：', llm_infer_source.model_name)
   
    start_time = time.perf_counter()

    # 读取所有样本并建立ID映射
    samples = []
    id_map = {}  # 存储ID到样本索引的映射
    with open(input_file, "r", encoding="utf-8") as infile:
        for idx, line in enumerate(infile):
            data = json.loads(line)
            samples.append(data)
            sample_id = data.get('id', f"sample_{idx}")  # 使用显式ID或生成唯一ID
            id_map[sample_id] = idx

    # 构建k倍推理列表
    expanded_prompts = []
    expanded_sample_ids = []  # 存储每个prompt对应的样本ID
    expanded_sample_indexes = []  # 存储每个prompt对应的样本索引
    
    for sample in samples:
        sample_id = sample.get('id', f"sample_{samples.index(sample)}")
        for _ in range(k):
            expanded_prompts.append(sample[input_key])
            expanded_sample_ids.append(sample_id)
            expanded_sample_indexes.append(samples.index(sample))

    # 批量推理
    if infer_mode == 'local':
        results = vllm_batch_infer(expanded_prompts, llm_infer_source, sampling_config)
    else:
        # API inference
        results = asyncio.run(api_infer_batch(expanded_prompts, llm_infer_source, max_concurrency))
        
    infer_time = time.perf_counter() - start_time

    # 按样本ID分组存储结果
    grouped_results = {}
    for sample_id, sample_idx, result in zip(expanded_sample_ids, expanded_sample_indexes, results):
        print("*" * 50)
        print(f"[Raw Model Output for {sample_id}]:")
        print(result)
        print("*" * 50)

        if sample_id not in grouped_results:
            # 初始化该ID的结果组
            grouped_results[sample_id] = {
                "original": samples[sample_idx].copy(),  # 原始样本数据
                "inferences": []  # 存储该ID的所有推理结果
            }
        
        # 评估单个推理结果
        ground_truth = samples[sample_idx]['gold_truth']
        question = samples[sample_idx]['question']
        format_correct, execute_success, answer_correct, exec_result, error_message = compute_score(
            llm_judge, result, ground_truth, question, table_dir
        )
        
        # 保存推理结果
        inference_data = {
            output_key: result,
            "format_correct": format_correct,
            "execute_success": execute_success,
            "answer_correct": answer_correct,
            "exec_result": exec_result,
            "error_message": error_message
        }
        grouped_results[sample_id]["inferences"].append(inference_data)

        print(f"\n[Sample ID: {sample_id}]")
        print(f"[Is Format Correct] {format_correct}")
        print(f"[Is Execute Success] {execute_success}")
        print(f"[Execute Result] {exec_result}")
        print(f"[Error Message] {error_message}")
        print(f"[Ground Truth] {ground_truth}")
        print(f"[Is Answer Correct] {answer_correct}")

    # 创建失败结果文件
    failure_output_file = output_file.replace(".jsonl", "_failures.jsonl")

    # 打印 Token 统计信息
    if infer_mode != 'local' and hasattr(llm_infer_source, 'total_tokens'):
        print("\n" + "="*50)
        print("[Infer Model Token Usage]")
        print(f"Prompt Tokens: {llm_infer_source.total_tokens['prompt_tokens']}")
        print(f"Completion Tokens: {llm_infer_source.total_tokens['completion_tokens']}")
        print(f"Total Tokens: {llm_infer_source.total_tokens['total_tokens']}")
        print("="*50 + "\n")

    if hasattr(llm_judge, 'total_tokens'):
        print("\n" + "="*50)
        print("[Judge Model Token Usage]")
        print(f"Prompt Tokens: {llm_judge.total_tokens['prompt_tokens']}")
        print(f"Completion Tokens: {llm_judge.total_tokens['completion_tokens']}")
        print(f"Total Tokens: {llm_judge.total_tokens['total_tokens']}")
        print("="*50 + "\n")
    
    # 处理结果并评估
    success_count = 0
    failure_count = 0
    total_samples = len(grouped_results)
    
    with open(output_file, "w", encoding="utf-8") as outfile,\
        open(failure_output_file, "w", encoding="utf-8") as failfile:
        
        for sample_id, group in grouped_results.items():
            # 检查是否有任意一次推理正确
            any_correct = any(inference["answer_correct"] for inference in group["inferences"])
            group["original"]["success@k"] = any_correct
            group["original"]["k_inferences"] = group["inferences"]
            
            # 更新统计
            if any_correct:
                success_count += 1
            else:
                failure_count += 1
                # 将失败结果写入失败文件
                failfile.write(json.dumps(group["original"], ensure_ascii=False) + "\n")
            
            # 写入输出文件
            outfile.write(json.dumps(group["original"], ensure_ascii=False) + "\n")
    
    # 打印总体指标
    success_rate = success_count / total_samples if total_samples > 0 else 0
    failure_rate = failure_count / total_samples if total_samples > 0 else 0
    print(f"\n[Final Summary]")
    print(f"Total samples: {total_samples}")
    print(f"Success@k (k={k}): {success_count} ({success_rate:.2%})")
    print(f"Failure@k (k={k}): {failure_count} ({failure_rate:.2%})")
    print(f"Failure cases saved to: {failure_output_file}")

    eval_time = time.perf_counter() - start_time - infer_time
    print(f"Inference time: {infer_time:.2f}s")
    print(f"Evaluation time: {eval_time:.2f}s")


def main():

    parser = argparse.ArgumentParser(description="Infer Tables.")
    parser.add_argument("--infer_mode", type=str, choices=['local', 'api'], default='local', help="Inference mode: 'local' (vLLM) or 'api'")
    parser.add_argument("--input_file", type=str, default=r"D:\repos\ReasonTabQA\src\en_717_20percent_test_sampled.jsonl", help="Input JSONL file path")
    parser.add_argument("--output_file", type=str, default=r"D:\repos\ReasonTabQA\output\output_en_717_20percent_test_sampled_gpt-5.2.jsonl", help="Output JSONL file path")  # 输出地址：输出jsonl
    parser.add_argument("--input_key", type=str, default='input', help="Input key in JSONL")  # 测试集中prompt在jsonl文件对应的key
    parser.add_argument("--output_key", type=str, default='prediction', help="Output key in JSONL")  # 输出jsonl文件中，推理内容对应的key
    parser.add_argument("--llm_infer_model_path", type=str, default="/gemini/space/private/zhangjie/lxy/model/QWQ-32B", help="Path to local model (for local mode)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--max_model_len", type=int,default=32768)
    parser.add_argument("--k", type=int, default=1)
    # parser.add_argument("--table_dir", type=str, default=r"D:\repos\TableTaskEval_origin_table\table_v6_v7", help="Root directory containing table files")
    parser.add_argument("--table_dir", type=str, default=r"D:\repos\TableTaskEval_origin_table\group_2_tables_english_withmapping_final_0708_csv", help="Root directory containing table files")
    parser.add_argument("--max_concurrency", type=int, default=10, help="Maximum concurrency for API inference")
    # parser.add_argument("--table_dir", type=str, default=r"D:\repos\KDD_open_data\en\table_preprocessed", help="Root directory containing table files")

    args = parser.parse_args()
    sampling_config = {
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'presence_penalty': args.presence_penalty,
        'max_model_len':args.max_model_len
    }
    
    # Load config
    try:
        with open('model_config.yaml', 'r', encoding='utf-8') as file:
            model_config = yaml.safe_load(file)
            judge_model_config = model_config['judge_model']
    except FileNotFoundError:
        print("Error: model_config.yaml not found. Please ensure it exists.")
        sys.exit(1)

    llm_judge = LLM(judge_model_config)
    
    if args.infer_mode == 'local':
        llm_infer_source = args.llm_infer_model_path
    else:
        if 'infer_model' not in model_config:
            print("Error: 'infer_model' configuration missing in model_config.yaml for API mode.")
            sys.exit(1)
        infer_model_config = model_config['infer_model']
        llm_infer_source = LLM(infer_model_config)

    infer_and_eval(args.input_file, args.output_file, args.input_key, args.output_key, args.infer_mode, llm_infer_source, llm_judge, sampling_config, args.table_dir, args.k, args.max_concurrency)
    

if __name__ == '__main__':
    main()
