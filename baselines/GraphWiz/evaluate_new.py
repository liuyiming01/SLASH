import json
import re
from pathlib import Path
from typing import Callable
import random
import torch
from tqdm import tqdm
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Sequence, List
import argparse
import shutil 


def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text)  # 处理形如 123,456
    res = re.findall(r"(\d+(\.\d+)?)", text)  # 匹配 123456.789
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return 0.0

# def check(key, truth, predict):
#     if key in ['cycle', 'connectivity', 'hamilton', 'substructure', 'bipartite']:
#         # 优先用 ### 后面的答案区；否则用全文
#         text = predict.split('###')[-1] if '###' in predict else predict

#         # 取“第一个” yes/no（适配：答案一开始就给出）
#         m = re.search(r'\b(yes|no)\b', text, flags=re.IGNORECASE)
#         if not m:
#             return False
#         first = m.group(1).lower()

#         t = truth.lower()
#         if first == 'yes' and 'yes' in t:
#             return True
#         if first == 'no' and 'no' in t:
#             return True
#         return False

#     elif key in ['flow', 'shortest', 'triangle']:
#         t_num = extract_last_num(truth)
#         # 优先用 ### 后面的答案区；否则用全文
#         text = predict.split('###')[-1] if '###' in predict else predict

#         # 取“第一个”数字（并兼容 123,456）
#         text = re.sub(r"(\d),(\d)", r"\1\2", text)
#         m = re.search(r"(\d+(\.\d+)?)", text)
#         if not m:
#             return False
#         p_num = float(m.group(1))

#         return abs(t_num - p_num) < 1e-2

#     elif key == 'topology':
#         truth_tail = truth.split('###')[-1]
#         pred_nums = [int(x) for x in re.findall(r"-?\d+", predict)]
#         pred_s = "|" + "|".join(map(str, pred_nums)) + "|"

#         inner_lists = re.findall(r"\[([0-9,\s-]+)\]", truth_tail)
#         for inner in inner_lists:
#             seq = [int(x) for x in re.findall(r"-?\d+", inner)]
#             if not seq:
#                 continue
#             seq_s = "|" + "|".join(map(str, seq)) + "|"
#             if seq_s in pred_s:   # 连续匹配
#                 return True
#         return False

def check(key, truth, predict):
    if key in ['cycle', 'connectivity', 'hamilton', 'substructure', 'bipartite']:
        # 优先用 ### 后面的答案区；否则用全文
        text = predict.split('###')[-1] if '###' in predict else predict

        # 取“第一个” yes/no（适配：答案一开始就给出）
        m = re.search(r'\b(yes|no)\b', text, flags=re.IGNORECASE)
        if not m:
            return False
        first = m.group(1).lower()

        t = truth.lower()
        if first == 'yes' and 'yes' in t:
            return True
        if first == 'no' and 'no' in t:
            return True
        return False

    elif key in ['flow', 'shortest', 'triangle']:
        t_num = extract_last_num(truth)
        # 优先用 ### 后面的答案区；否则用全文
        text = predict.split('###')[-1] if '###' in predict else predict

        # 取第一个 'is' 后面的数字
        m = re.search(r'is\s*[:：]?\s*(-?\d+(\.\d+)?)', text, flags=re.IGNORECASE)
        if m:
            p_num = float(m.group(1))
        else:
            # fallback: 取第一个数字
            text = re.sub(r"(\d),(\d)", r"\1\2", text)
            m = re.search(r"(\d+(\.\d+)?)", text)
            if not m:
                return False
            p_num = float(m.group(1))

        return abs(t_num - p_num) < 1e-2

    elif key == 'topology':
        truth_tail = truth.split('###')[-1]
        # 取第一个 'is' 后面的数字列表
        m = re.search(r'is\s*[:：]?\s*([0-9,\s\-\>\[\]]+)', predict, flags=re.IGNORECASE)
        if m:
            nums = [int(x) for x in re.findall(r"-?\d+", m.group(1))]
            pred_s = "|" + "|".join(map(str, nums)) + "|"
        else:
            pred_nums = [int(x) for x in re.findall(r"-?\d+", predict)]
            pred_s = "|" + "|".join(map(str, pred_nums)) + "|"

        inner_lists = re.findall(r"\[([0-9,\s-]+)\]", truth_tail)
        for inner in inner_lists:
            seq = [int(x) for x in re.findall(r"-?\d+", inner)]
            if not seq:
                continue
            seq_s = "|" + "|".join(map(str, seq)) + "|"
            if seq_s in pred_s:   # 连续匹配
                return True
        return False

# def MODIFICATION(model, layers_heads_to_modify, delta_ratio, first_token_idx=0):
#     import sys
#     import types
#     import transformers.models.llama.modeling_llama as llama_modeling
#     sys.path.insert(0, "/home/lym/LLM-Research/Attention/Graph_Attention/GraphLens")
#     from modeling import modeling_llama_attn_shift
#     LlamaModel_forward, LlamaDecoderLayer_forward, LlamaAttention_forward = modeling_llama_attn_shift.get_modified_forward_llama(
#         layers_heads_to_modify = layers_heads_to_modify,
#         delta_ratio=delta_ratio,
#         first_token_idx=first_token_idx
#     )
#     model.model.forward = types.MethodType(LlamaModel_forward, model.model)
#     for idx, layer in enumerate(model.model.layers):
#         layer.forward = types.MethodType(LlamaDecoderLayer_forward, layer)
#         layer.self_attn.forward = types.MethodType(LlamaAttention_forward, layer.self_attn)

def MODIFICATION(model, layers_heads_to_modify, delta_ratio, first_token_idx=0):
    import sys
    import types

    sys.path.insert(0, "/home/lym/LLM-Research/Attention/Graph_Attention/GraphLens")

    model_type = getattr(getattr(model, "config", None), "model_type", None)
    model_name = model.__class__.__name__.lower()

    # Prefer config.model_type; fallback to class name heuristic
    is_llama = (model_type in {"llama"}) or ("llama" in model_name)
    is_qwen3 = (model_type in {"qwen3"}) or ("qwen3" in model_name)

    if is_llama:
        from modeling import modeling_llama_attn_shift
        LlamaModel_forward, LlamaDecoderLayer_forward, LlamaAttention_forward = (
            modeling_llama_attn_shift.get_modified_forward_llama(
                layers_heads_to_modify=layers_heads_to_modify,
                delta_ratio=delta_ratio,
                first_token_idx=first_token_idx,
            )
        )
        model.model.forward = types.MethodType(LlamaModel_forward, model.model)
        for layer in model.model.layers:
            layer.forward = types.MethodType(LlamaDecoderLayer_forward, layer)
            layer.self_attn.forward = types.MethodType(LlamaAttention_forward, layer.self_attn)
        return

    if is_qwen3:
        from modeling import modeling_qwen3_attn_shift
        Qwen3Model_forward, Qwen3DecoderLayer_forward, Qwen3Attention_forward = (
            modeling_qwen3_attn_shift.get_modified_forward_qwen3(
                layers_heads_to_modify=layers_heads_to_modify,
                delta_ratio=delta_ratio,
                first_token_idx=first_token_idx,
            )
        )
        model.model.forward = types.MethodType(Qwen3Model_forward, model.model)
        for layer in model.model.layers:
            layer.forward = types.MethodType(Qwen3DecoderLayer_forward, layer)
            layer.self_attn.forward = types.MethodType(Qwen3Attention_forward, layer.self_attn)
        return

    raise ValueError(
        f"Unsupported model for MODIFICATION(): model_type={model_type}, class={model.__class__.__name__}. "
        "Currently supports: llama, qwen3."
    )

def main(args):
    batch_size = args.batch_size
    print(f"main start, batch_size:{batch_size}")

    model, tokenizer = get_model(args.model_path)
    print("model loaded")

    # 记录修改模式：none / config / list
    mod_mode = "none"
    layers_heads_to_modify = None

    if args.layer_head_config_path is not None:
        # 情况 2: 使用外部 json 配置
        print(f"Loading layer-head config from {args.layer_head_config_path}")
        with open(args.layer_head_config_path, "r") as f:
            layers_heads_to_modify = json.load(f)
        mod_mode = "config"
    else:
        # 情况 3: 使用命令行列表指定要修改的 layer
        if args.layers_to_modify is not None:
            print(f"Using layers_to_modify from args: {args.layers_to_modify}")
            layers_to_modify = args.layers_to_modify
            layers_heads_to_modify = {
                str(l): list(range(model.config.num_attention_heads))
                for l in layers_to_modify
            }
            mod_mode = "list"
        else:
            # 情况 1: 不做任何修改
            print("No layer_head_config_path and no layers_to_modify; no modification will be applied.")
            layers_heads_to_modify = None
            mod_mode = "none"

    if layers_heads_to_modify:
        print(f"Applying modifications for layers: {list(layers_heads_to_modify.keys())}")
        MODIFICATION(model, layers_heads_to_modify, delta_ratio=args.delta_ratio)

    batch_llama = get_batch_llama(model, tokenizer, args)

    pure_model = args.model_path.split('/')[-1]
    prompt_tag = 'only'
    if args.cot:
        prompt_tag += '_cot'
    if args.sys_prompt:
        prompt_tag += '_sys'

    base_dir = f"{args.output_dir}/{pure_model}/{prompt_tag}"
    if mod_mode == "none":
        save_dir = f"{base_dir}/test"
    elif mod_mode == "config":
        config_tag = args.layer_head_config_path.split('_')[-1].replace('.json', '')
        save_dir = f"{base_dir}/modified_{config_tag}_delta{args.delta_ratio}"
    elif mod_mode == "list":
        layer_tag = "_".join(str(l) for l in args.layers_to_modify)
        save_dir = f"{base_dir}/delta{args.delta_ratio}_{layer_tag}"

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 若是 config 模式，把配置文件拷贝一份到 save_dir
    if mod_mode == "config" and args.layer_head_config_path is not None:
        cfg_src = Path(args.layer_head_config_path)
        cfg_dst = Path(save_dir) / cfg_src.name
        try:
            shutil.copy(cfg_src, cfg_dst)
            print(f"Copied layer config to {cfg_dst}")
        except Exception as e:
            print(f"Warning: failed to copy config file: {e}")

    default_tasks = ['cycle', 'connectivity', 'hamilton', 'substructure', 'bipartite', 'flow', 'shortest', 'triangle', 'topology']
    tasks = args.tasks if args.tasks else default_tasks

    results = {}

    for lang in tasks:
        print(f'===========we are testing in {lang}====================')

        with open(f'/home/lym/data1/Datasets/GraphWiz/GraphInstruct-Test/{lang}_test.json') as f:
            datas = f.readlines()
            
        gen_datas_jsonl = Path(save_dir) / f"_gen_{lang}_datas.jsonl"
        start_index = (
            len(open(gen_datas_jsonl).readlines()) if gen_datas_jsonl.exists() else 0
        )
        print(f"start_index: {start_index}")
        
        task_datas = [json.loads(item) for item in datas]

        for i in tqdm(range(start_index, len(task_datas), batch_size)):
            cur_batch = task_datas[i : i + batch_size]
            input_str_list, output_str_list = batch_gen(args, cur_batch, batch_llama, tokenizer)

            for j, (task_data, input_str, output_str) in enumerate(
                zip(cur_batch, input_str_list, output_str_list)
            ):
                with open(gen_datas_jsonl, "a") as f:
                    json.dump(
                        dict(
                            index=i + j,
                            label=task_data["answer"],
                            output_str=output_str,
                            source_data=task_data,
                            input_str=input_str,
                            task=lang
                        ),
                        f,
                    )
                    f.write("\n")

        with open(gen_datas_jsonl) as f:
            gen_datas = [json.loads(line) for line in f]

        correct_results = []
        wrong_results = []
        for gen in gen_datas:
            result = dict(
                is_correct=None,
                extract_true=gen["source_data"]["answer"],
                extract_pred=gen["output_str"].lstrip(),
                **gen,
            )
            
            if check(lang, result["extract_true"].lower(), result["extract_pred"].lower()):
                result["is_correct"] = True
                correct_results.append(result)
            else:
                result["is_correct"] = False
                wrong_results.append(result)

        result = f"Accuracy={len(correct_results)}/({len(correct_results)}+{len(wrong_results)})={len(correct_results)/(len(correct_results) + len(wrong_results))}"
        print(result)
        with open(Path(save_dir) / f"{lang}_correct.json", "w", encoding='utf-8') as f:
            json.dump(correct_results, f, ensure_ascii=False, indent=4)
        with open(Path(save_dir) / f"{lang}_wrong.json", "w", encoding='utf-8') as f:
            json.dump(wrong_results, f, ensure_ascii=False, indent=4)
    
        num_result = float(result.split('=')[-1])
        results[lang] = num_result

    average = sum(results.values()) / len(results)
    print(average)
    import csv

    csv_path = Path(save_dir) / f"NLG_evaluate_results_bs{batch_size}.csv"
    need_header = (not csv_path.exists()) or (csv_path.stat().st_size == 0)
    with open(csv_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if need_header:
            writer.writerow(["Task", "Accuracy"])

        for key, value in results.items():
            writer.writerow([key, value])

def get_prompt(args, cur_batch, tokenizer):
    if args.cot:
        prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request step by step, "
            "and then gives the final answer after '### ' on the last line.\n\n"
            "### Instruction:\n{query}\n\n### Response:"
        )
    else:
        # prompt = (
        #     "Below is an instruction that describes a task. "
        #     "Only give the final answer after '### '\n\n"
        #     "{query}\n"
        # )
        prompt = (
            "{query}"
        )

    input_strs = [v['input_prompt'] for v in cur_batch]
    input_str_list = [prompt.format(query=s) for s in input_strs]

    if args.sys_prompt:
        if 'qwen' in args.model_path.lower():
            input_str_list = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": t}
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for t in input_str_list
            ]
        else:
            input_str_list = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": t}
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for t in input_str_list
            ]
    
    return input_str_list

def batch_gen(args, cur_batch, batch_llm, tokenizer):

    input_str_list = get_prompt(args, cur_batch, tokenizer)
    output_str_list = batch_llm(input_str_list, args)

    return input_str_list, output_str_list


def get_batch_llama(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, args):
    @torch.inference_mode()
    def batch_llama(input_strs, args):
        input_ids_w_attnmask = tokenizer(
            input_strs,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False if args.sys_prompt and 'llama' in args.model_path.lower() else True
        ).to(model.device)
        
        output_ids = model.generate(
            input_ids=input_ids_w_attnmask.input_ids,
            attention_mask=input_ids_w_attnmask.attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=args.max_tokens,
            do_sample=False,
        ).tolist()
        
        
        real_output_ids = [
            output_id[len(input_ids_w_attnmask.input_ids[i]) :] for i, output_id in enumerate(output_ids)
        ]
        output_strs = tokenizer.batch_decode(real_output_ids, skip_special_tokens=True)
        return output_strs

    return batch_llama


def get_model(model_path: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    return model, tokenizer


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Eval the LLM on GraphWiz")
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Which GraphWiz task(s) to evaluate. Default: all GraphWiz tasks."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to baseline model",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save evaluation results",
        default="./Results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="batchsize",
        required=True
    )
    parser.add_argument(
        "--sys_prompt",
        action="store_true",
        help="whether to add system prompt",
        default=False
    )
    parser.add_argument(
        "--cot",
        action="store_true",
        help="whether to add system prompt",
        default=False
    )
    parser.add_argument(
        "--shot",
        type=int,
        help="how many examples in your prompts",
        default=4
    )
    parser.add_argument(
        "--shuffle",
        type= bool,
        help="whether to shuffle your choices",
        default = True
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        help="maximum output tokens",
        default = 1024
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="seed",
        default = 0
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="specific language to test",
        default = '/cpfs/user/chennuo/CN/XBenchMARK/Cross-Lingual-Consistency/1_easyrun/BMLAMA53'
    )
    parser.add_argument(
        '--layer_head_config_path',
        type=str,
        default=None,
        help="Path to a JSON file specifying which layer-heads to modify."
    )
    parser.add_argument(
        '--layers_to_modify',
        type=int,
        nargs='+',
        default=None,
        help="List of layer indices to modify when no layer_head_config_path is provided; "
             "set to None (do not pass this argument) to disable modification."
    )
    parser.add_argument(
        '--delta_ratio',
        type=float,
        default=0.4,
        help="The delta ratio for attention redistribution."
    )
    args = parser.parse_args()

    main(args=args)