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
"""
Preprocess the TABLE dataset to parquet format
"""


import argparse
import os
import json
import random

import datasets

from verl.utils.hdfs_io import copy, makedirs


def apply_prompt(table_desc, query):
    prompt_head = """你是一位精通 Python 的数据分析师。你的任务是编写可执行的 Python 代码来解析表格，然后回答问题。

要求：
1. 根据问题，写出你的分析思路和方法，再根据这种方法写 Python 代码。
2. 请严格按照给你的文件路径和表格描述来生成代码。
3. 只生成一个代码块，并且严格以 ```python 开始，并以 ``` 结束。
4. 你的分析必须完全基于表格数据。如果用户的问题与数据分析无关，请礼貌拒绝。
5. 你需要生成可执行代码。如果有结果需要展示，请将结果存入answer函数中， 并用print展示。
6. 确保使用python库中pd.read_csv函数，读取给你的表格文件路径来进行数据处理，如需要读取多个表格，注意使用文件路径进行变量名区分。
7. 生成代码的过程中，请不要将数据转成DataFrame格式，请务必用pd.read_csv函数进行表格内容读取。

以下是提供的表格信息：
"""

    prompt_tail = """
确保最终答案是 Python 代码的最后一行，并且只能以 print(f'{answer}') 的形式出现，不能有其他形式。

让我们逐步思考，然后生成 Python 代码来分析表格并展示问题的最终答案。
输入问题：
"""
    # prefix = "table/"
    # table_desc = table_desc.replace("文件路径: ", "文件路径: " + prefix)
    prompt = prompt_head + "\n" + table_desc + "\n" + prompt_tail + query
    return prompt


def load_local_single_jsonl(data_dir):
    data = []
    with open(data_dir, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            line_new = {}
            if len(line['table_desc']) < 10:
                print([line['table_desc']])
                continue

            table_desc, query = line['table_desc'], line['question']
            prompt = apply_prompt(table_desc, query)

            line_new['prompt'] = prompt
            line_new['question'] = str(line['question'])
            line_new['gold_truth'] = str(line['gold_truth'])
            data.append(line_new)
    # shuffle
    random.shuffle(data)
    return data


def load_local_dataset(train_data_dir, test_data_dir):

    random.seed(2025)
    train_data = load_local_single_jsonl(train_data_dir)
    test_data = load_local_single_jsonl(test_data_dir)

    print('train_dataset_size:\n', len(train_data))
    print('an example of train_dataset:\n', train_data[0])
    print('test_dataset size:\n', len(test_data))
    print('an example of test_dataset:\n', test_data[0])

    return datasets.DatasetDict({
        "train": datasets.Dataset.from_list(train_data), 
        "test": datasets.Dataset.from_list(test_data)
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/table/qwen3_v6_20250618")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "table"
    train_data_dir = ''
    test_data_dir = ''

    dataset = load_local_dataset(train_data_dir, test_data_dir)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            prompt = example.pop("prompt")
            messages = [{'content': 'You are a helpful assistant.', 'role': 'system'}, 
                        {'content': prompt, 'role': 'user'}]
            print(messages)
            ground_truth = str(example.pop("gold_truth"))

            data = {
                "data_source": data_source,
                "prompt": messages,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": {"split": split, "index": idx, "question": example.pop("question")},
            }
            print(data)
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
