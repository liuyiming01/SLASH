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

REPO_ROOT = "SLASH"

CHOICES = ['A', 'B', 'C', 'D', 'E', 'F','G', 'H', 'I', 'J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']


def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text)
    res = re.findall(r"(\d+(\.\d+)?)", text)
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return 0.0
    
    
def check(key, truth, predict):
    
    if key in ['cycle', 'connectivity', 'hamilton', 'substructure', 'bipartite']:
        if '###' in predict:
            if 'yes' in truth.lower() and 'yes' in predict.split('###')[-1].lower():
                # correct_samples[key].append(v)
                return True
            elif 'no' in truth.lower() and 'no' in predict.split('###')[-1].lower():
                return True
            return False
        else:
            matches = re.findall(r'\b(yes|no)\b', predict, flags=re.IGNORECASE)
            if matches:
                last_match = matches[-1].lower()
                if last_match == 'yes' and 'yes' in truth.lower():
                    return True
                elif last_match == 'no' and 'no' in truth.lower():
                    return True
            else:
                return False
    elif key in ['flow', 'shortest', 'triangle']:
      
        t_num = extract_last_num(truth)
        p_num = extract_last_num(predict.split('###')[-1])
        if abs(t_num - p_num) < 1e-2:
            return True
        return False
                
    elif key == 'topology':
        
        if '###' in predict:
            pre = predict.split('###')[-1].strip(' ')
            truth = truth.split('###')[-1].strip(' ')
            if truth in pre or pre in truth:
                return True
            return False
        else:
            truth = truth.split('###')[-1].split(',')
            for t in truth:
                if t in predict or t.strip(' ') in predict:
                    return True
            return False


def MODIFICATION(model, layers_heads_to_modify, gamma, first_token_idx=0):
    import sys
    import types

    sys.path.insert(0, REPO_ROOT)

    model_type = getattr(getattr(model, "config", None), "model_type", None)
    model_name = model.__class__.__name__.lower()

    # Prefer config.model_type; fallback to class name heuristic
    is_llama = (model_type in {"llama"}) or ("llama" in model_name)
    is_qwen3 = (model_type in {"qwen3"}) or ("qwen3" in model_name)
    is_mistral = (model_type in {"mistral"}) or ("mistral" in model_name)

    if is_llama:
        from modeling import modeling_llama_attn_shift
        LlamaModel_forward, LlamaDecoderLayer_forward, LlamaAttention_forward = (
            modeling_llama_attn_shift.get_modified_forward_llama(
                layers_heads_to_modify=layers_heads_to_modify,
                gamma=gamma,
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
                gamma=gamma,
                first_token_idx=first_token_idx,
            )
        )
        model.model.forward = types.MethodType(Qwen3Model_forward, model.model)
        for layer in model.model.layers:
            layer.forward = types.MethodType(Qwen3DecoderLayer_forward, layer)
            layer.self_attn.forward = types.MethodType(Qwen3Attention_forward, layer.self_attn)
        return

    if is_mistral:
        from modeling import modeling_mistral_attn_shift
        MistralModel_forward, MistralDecoderLayer_forward, MistralAttention_forward = (
            modeling_mistral_attn_shift.get_modified_forward_mistral(
                layers_heads_to_modify=layers_heads_to_modify,
                gamma=gamma,
                first_token_idx=first_token_idx,
            )
        )
        model.model.forward = types.MethodType(MistralModel_forward, model.model)
        for layer in model.model.layers:
            layer.forward = types.MethodType(MistralDecoderLayer_forward, layer)
            layer.self_attn.forward = types.MethodType(MistralAttention_forward, layer.self_attn)
        return

    raise ValueError(
        f"Unsupported model for MODIFICATION(): model_type={model_type}, class={model.__class__.__name__}. "
        "Currently supports: llama, qwen3, mistral."
    )

def main(
    args,
    is_bf16: bool = True,
    save_dir: str  = None,
):
    batch_size = args.batch_size
    print(f"Evaluating task={args.tasks}, model={args.model_path.split('/')[-1]}, gamma={args.gamma}, batch_size={args.batch_size}")
    
    model_path = args.model_path
    model, tokenizer = get_model(model_path, is_bf16=is_bf16)

    mod_mode = "none"
    layers_heads_to_modify = None

    if args.layer_head_config_path is not None:
        print(f"Loading layer-head config from {args.layer_head_config_path}")
        with open(args.layer_head_config_path, "r") as f:
            layers_heads_to_modify = json.load(f)
        mod_mode = "config"
    else:
        if args.layers_to_modify is not None:
            print(f"Using layers_to_modify from args: {args.layers_to_modify}")
            layers_to_modify = args.layers_to_modify
            layers_heads_to_modify = {
                str(l): list(range(model.config.num_attention_heads))
                for l in layers_to_modify
            }
            mod_mode = "list"
        else:
            print("No layer_head_config_path and no layers_to_modify; no modification will be applied.")
            layers_heads_to_modify = None
            mod_mode = "none"

    if layers_heads_to_modify:
        print(f"Applying modifications for layers: {list(layers_heads_to_modify.keys())}")
        MODIFICATION(model, layers_heads_to_modify, gamma=args.gamma)

    batch_llama = get_batch_llama(model, tokenizer, args)

    pure_model = model_path.split('/')[-1]
    if save_dir is None:
        base_dir = f"{args.output_dir}/{pure_model}/{model.dtype}"
        if mod_mode == "none":
            save_dir = f"{base_dir}/test"
        elif mod_mode == "config":
            # config_tag = args.layer_head_config_path.split('_')[-1].replace('.json', '')
            # save_dir = f"{base_dir}/modified_{config_tag}gamma{args.gamma}"
            save_dir = f"{base_dir}/modified_gamma{args.gamma}"
        elif mod_mode == "list":
            layer_tag = "_".join(str(l) for l in args.layers_to_modify)
            save_dir = f"{base_dir}/gamma{args.gamma}_{layer_tag}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if mod_mode == "config" and args.layer_head_config_path is not None:
        cfg_src = Path(args.layer_head_config_path)
        cfg_dst = Path(save_dir) / cfg_src.name
        try:
            shutil.copy(cfg_src, cfg_dst)
            print(f"Copied layer config to {cfg_dst}")
        except Exception as e:
            print(f"Warning: failed to copy config file: {e}")

    default_tasks = ['cycle', 'connectivity', 'hamilton', 'substructure', 'bipartite',
                     'flow', 'shortest', 'triangle', 'topology']
    if args.tasks is not None and len(args.tasks) > 0:
        tasks = args.tasks
    elif args.lang_only:
        tasks = [args.lang_only]
    else:
        tasks = default_tasks

    results = {}

    for lang in tasks:
        print(f'==========={args.run_mode} in {lang}====================')

        with open(f'/home/lym/data1/Datasets/GraphWiz/GraphInstruct-Test/{lang}_test.json') as f:
            total_datas = f.readlines()
        if args.run_mode == "calibration":
            random.seed(42)
            datas = random.sample(total_datas, 100)[:50]
        elif args.run_mode == "test":
            random.seed(42)
            calibration_data = random.sample(total_datas, 100)[:50]
            remaining_data = [data for data in total_datas if data not in calibration_data]
            datas = remaining_data
        else:
            raise ValueError(f"Unknown run_mode: {args.run_mode}")
            
        gen_datas_jsonl = Path(save_dir) / f"_gen_{lang}_datas.jsonl"
        start_index = (
            len(open(gen_datas_jsonl).readlines()) if gen_datas_jsonl.exists() else 0
        )
        print(f"start_index: {start_index}")
        
        gsm8k_datas = [json.loads(item) for item in datas]

        for i in tqdm(range(start_index, len(gsm8k_datas), batch_size)):
            cur_gsm8k_batch = gsm8k_datas[i : i + batch_size]
            input_str_list, output_str_list = gsm8k_batch_gen(lang, 
                cur_gsm8k_batch, batch_llama, args
            )
            for j, (gsm8k_data, input_str, output_str) in enumerate(
                zip(cur_gsm8k_batch, input_str_list, output_str_list)
            ):
                with open(gen_datas_jsonl, "a") as f:
                    json.dump(
                        dict(
                            index=i + j,
                            output_str=output_str,
                            source_data=gsm8k_data,
                            input_str=input_str,
                            task=lang
                        ),
                        f,
                    )
                    f.write("\n")

    # calculate acc
        with open(gen_datas_jsonl) as f:
            gen_datas = [json.loads(line) for line in f]

        correct_results = []
        wrong_results = []
        for gen in gen_datas:
            result = dict(
                **gen,
                extract_true=gen["source_data"]["answer"],
                extract_pred=gen["output_str"].lstrip(),
                is_correct=None,
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

def gsm8k_batch_gen(
    lang_, cur_gsm8k_batch, batch_llm, args
):
    lang = lang_ if lang_ != 'En_gsm8k' else 'English'
    prompt_no_input = (
      "Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request step by step.\n\n"
        "### Instruction:\n{query}\n\n### Response:"
    )
    try:
        curs_gsm8k_questions = [v['question'] for v in cur_gsm8k_batch]
    except:
        curs_gsm8k_questions = [v['input_prompt'] for v in cur_gsm8k_batch]
    # prompt_no_input = PROMPT_DICTS['normal_prompt']
    input_str_list = [prompt_no_input.format(query=q) for q in curs_gsm8k_questions]
    output_str_list = batch_llm(input_str_list)
    return input_str_list, output_str_list


def get_batch_llama(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, args):
    @torch.inference_mode()
    def batch_llama(input_strs):
        input_ids_w_attnmask = tokenizer(
            input_strs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        
        output_ids = model.generate(
            input_ids=input_ids_w_attnmask.input_ids,
            attention_mask=input_ids_w_attnmask.attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            generation_config=GenerationConfig(
                max_new_tokens=args.max_tokens,
                do_sample=False,
                temperature=0.0,  # t=0.0 raise error if do_sample=True
            ),
        ).tolist()
        
        
        real_output_ids = [
            output_id[len(input_ids_w_attnmask.input_ids[i]) :] for i, output_id in enumerate(output_ids)
        ]
        output_strs = tokenizer.batch_decode(real_output_ids, skip_special_tokens=True)
        return output_strs

    return batch_llama


def get_model(model_path: str, is_bf16: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print('new pad ', tokenizer.pad_token)

    if is_bf16:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            attn_implementation="eager",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype='auto',
            device_map='auto',
            attn_implementation="eager",
        ).cuda()
    model.eval()

    return model, tokenizer


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
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
    # parser.add_argument(
    #     "--streategy",
    #     type=str,
    #     help="which streategy to evaluate the model",
    #     required=True,
    #     choices=['Parallel','Cross']
    # )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="batchsize",
        required=True
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Which GraphWiz task(s) to evaluate. Default: all tasks."
    )
    parser.add_argument(
        "--lang_only",
        type=str,
        help="specific language to test",
        default = ''
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
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--run_mode", type=str, default="test")
    args = parser.parse_args()

    main(args=args)