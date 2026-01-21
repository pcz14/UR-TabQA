# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
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

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
import json
import time
import re
import requests
from codebleu import calc_codebleu

def extract_python_code(prediction):
    """使用正则表达式提取三反引号包裹的Python代码"""
    pattern = r"```python(.*?)```"
    code_blocks = re.findall(pattern, prediction, re.DOTALL)
    code = "\n".join(code_blocks)
    return code

@register("dapo")
class DAPORewardManager:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    def __call__(self, data: DataProto, return_dict: bool = False, step: int=0):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]
            
        print("\n\n===== Starting Reward Calculation =====")
        print(f"Processing batch with {len(data)} samples")
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        # 第一步：收集同一个prompt的所有response信息
        prompt_groups = defaultdict(list)
        print("\n=== Step 1: Grouping responses by prompt ===")
        
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            
            # 存储这个response的信息，稍后处理
            prompt_groups[prompt_str].append(i)
            #print(f"  Sample {i} -> Prompt: '{prompt_str[:50]}...'")
        
        print(f"\nFound {len(prompt_groups)} unique prompts in batch")
        # for prompt, indices in prompt_groups.items():
        #     print(f"  Prompt: '{prompt[:50]}...' has {len(indices)} responses")
        
        # 第二步：处理每个response并收集正确代码
        print("\n=== Step 2: Calculating initial scores and collecting correct codes ===")
        all_scores = []  # 存储每个response的分数信息
        correct_codes_by_prompt = defaultdict(list)  # 按prompt存储正确代码
        
        # 第一次遍历：计算分数并收集正确代码
        for prompt_str, indices in prompt_groups.items():
            print(f"\nProcessing prompt: '{prompt_str[:50]}...'")
            for i in indices:
                data_item = data[i]
                
                # 提取response
                response_ids = data_item.batch['responses']
                prompt_length = data_item.batch['prompts'].shape[-1]
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]
                response_str = self.tokenizer.decode(valid_response_ids)
                
                # 提取其他必要信息
                ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            
                data_source = data_item.non_tensor_batch['data_source']
                extra_info = data_item.non_tensor_batch.get('extra_info', None)
                
                print(f"\n  Calculating score for sample {i}...")
                print(f"  Response: '{response_str[:50]}...'")
                print(f"  Ground truth: '{ground_truth[:50]}...'")
                
                # 使用compute_score计算分数
                dict_scores = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
                
                
                # 保存分数信息
                all_scores.append({
                    "index": i,
                    "total_score": dict_scores['score'],
                    "dict_scores": dict_scores,
                    "valid_response_length": valid_response_length,
                    "prompt_str": prompt_str,
                    "response_str": response_str
                })
                
                print(f"  Initial scores for sample {i}:")
                print(f"    Score: {dict_scores['score']:.2f}")
                print(f"    Format: {dict_scores['format']:.2f}")
                print(f"    Execute: {dict_scores['execute']:.2f}")
                print(f"    Answer: {dict_scores['answer']:.2f}")
                print(f"    CodeBLEU: {dict_scores['codebleu']:.4f}")
                print(f"    Filepath: {dict_scores.get('filepath', 0.0):.2f}")
                
                # 如果答案正确，保存代码
                # 修改：使用严格的 acc (Total=3.5) 作为判断标准，确保路径和答案都正确才算 Correct
                if dict_scores.get('acc', False):
                    try:
                        # 提取代码
                        code_str = response_str.split("</think>")[-1].replace("<|im_end|>", "").replace("", "").replace("<_end>", "").replace("<|endoftext|>", "").strip()
                        cleaned_code = extract_python_code(code_str)
                        # print("\ncode_str code:{}".format(code_str))
                        # print("\nCleaned code:{}".format(cleaned_code))
                        correct_codes_by_prompt[prompt_str].append(cleaned_code)
                        print(f"  ✅ Sample {i} is CORRECT, saving code for future comparison")
                    except Exception as e:
                        print(f"  ❌ Error extracting code for correct answer: {e}")
                else:
                    print(f"  ❌ Sample {i} is INCORRECT")
        
        # 第三步：修正错误答案的codebleu分数
        print("\n=== Step 3: Adjusting CodeBLEU scores for incorrect answers ===")
        for score_info in all_scores:
            if not score_info["dict_scores"].get('acc', False):  # 只要不是完全正确 (acc=True)，都尝试修正
                prompt_str = score_info["prompt_str"]
                correct_codes = correct_codes_by_prompt.get(prompt_str, [])
                old_bleu = score_info["dict_scores"].get('codebleu', 0)
                
                print(f"\n  Adjusting sample {score_info['index']} (incorrect answer)")
                print(f"  Prompt has {len(correct_codes)} correct solutions")
                
                if correct_codes:
                    # 提取当前response的代码
                    try:
                        code_str = score_info["response_str"].split("</think>")[-1].replace("<|im_end|>", "").replace("", "").replace("<_end>", "").replace("<|endoftext|>", "").strip()
                        # _, _, cleaned_code = code_exec_result(code_str)
                        cleaned_code=extract_python_code(code_str)
                        # 计算与所有正确代码的平均codebleu
                        total_bleu = 0.0
                        print(f"  Calculating CodeBLEU against {len(correct_codes)} correct solutions")
                        
                        for idx, correct_code in enumerate(correct_codes):
                            try:
                                codebleu_result = calc_codebleu(
                                    [cleaned_code],
                                    [correct_code],
                                    lang="python",
                                    weights=(0.1, 0.1, 0.4, 0.4)
                                )
                                bleu_score = codebleu_result["codebleu"]
                                total_bleu += bleu_score
                                print(f"    Against correct solution {idx+1}: CodeBLEU = {bleu_score:.4f}")
                            except Exception as e:
                                print(f"    ❌ CodeBLEU calculation error: {e}")
                                total_bleu += 0.0
                        
                        avg_bleu = total_bleu / len(correct_codes)
                        
                    #     # 修正分数
                        score_info["dict_scores"]['codebleu'] = avg_bleu
                   
                        score_info['total_score'] =score_info["dict_scores"]['score']- old_bleu + avg_bleu
                        
                        print(f"score_info: {score_info}")
                        print(f"  CodeBLEU adjustment for sample {score_info['index']}:")
                        print(f"    Prompt: '{prompt_str[:50]}...'")
                        print(f"    Old CodeBLEU: {old_bleu:.4f} -> New: {avg_bleu:.4f}")
                        print(f"    Old total: {score_info['dict_scores']['score']:.2f} -> New: {score_info['total_score']:.2f}")
                    except Exception as e:
                        print(f"  ❌ Error processing incorrect answer: {e}")

                        # 修正分数 - 修复点在这里
                    #     old_total_score = score_info['total_score']
                    #     new_total_score = old_total_score - old_bleu + avg_bleu
                        
                    #     score_info["dict_scores"]['codebleu'] = avg_bleu
                    #     score_info['total_score'] = new_total_score
                        
                    #     print(f"  CodeBLEU adjustment for sample {score_info['index']}:")
                    #     print(f"    Old CodeBLEU: {old_bleu:.4f} -> New: {avg_bleu:.4f}")
                    #     print(f"    Old total: {old_total_score:.2f} -> New: {new_total_score:.2f}")
                    # except Exception as e:
                    #     print(f"  ❌ Error processing incorrect answer: {e}")
                else:
                    print(f"  ⚠️ No correct solutions for this prompt, keeping original CodeBLEU: {old_bleu:.4f}")
        
        # 第四步：填充结果张量
        print("\n=== Step 4: Filling result tensors ===")
        # 初始化分项张量
        reward_tensor_items = {
            'score':[],
            'format': [],
            'execute': [],
            'answer': [],
            'codebleu': [],
            'filepath': [],
            'acc': []
        }
        # print('----------------naive.py----reward_tensor_items-------')
        # print(all_scores)
        
        print("\nFinal scores:")
        for score_info in all_scores:
            i = score_info["index"]
            valid_response_length = score_info["valid_response_length"]
            total_score = score_info["total_score"]
            dict_scores = score_info["dict_scores"]
            
            # 设置总分
            reward_tensor[i, valid_response_length - 1] = total_score
            
            # 设置分项分数
            for key in ['format', 'execute', 'answer', 'codebleu', 'filepath', 'acc']:
                if key in dict_scores:
                    reward_tensor_items[key].append(dict_scores[key])
            reward_tensor_items['score'].append(total_score)
            # 设置总分到 'score' 键
            # reward_tensor_items['score'][i, valid_response_length - 1] = total_score
            # print(f"  Sample {i}:")
            # print(f"    Total reward: {total_score:.2f}")
            # print(f"    Format: {dict_scores['format']:.2f}")
            # print(f"    Execute: {dict_scores['execute']:.2f}")
            # print(f"    Answer: {dict_scores['answer']:.2f}")
            # print(f"    CodeBLEU: {dict_scores['codebleu']:.4f}")
        
        print("\n===== Reward Calculation Completed =====")
        # print('----------------naive.py----reward_tensor_items-------')
        # print(reward_tensor_items)
    
        if return_dict:
            return {
                "reward_tensor": reward_tensor, # Tensor[batch_size, response_length]
                "reward_extra_info": reward_tensor_items,  # Dict[List[batch_size]]   Dict[Tensor[batch_size response_length]] {'format': [-1, -1, -0.5, ...], 'execute': [...], 'answer': [...]}
            }
        else:
            return reward_tensor