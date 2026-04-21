import subprocess
import re
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import llm_scorer
import concurrent.futures
import threading
import queue
import sys
import random
import os
import argparse

def check_ad_tags_length_diff(raw_text: str, tagged_text: str, threshold: int = 10) -> bool:
    """
    Checks if `tagged_text` has roughly the same length as `raw_text` after removing <ad>, </ad>, and all whitespaces.
    Allows a length difference of up to `threshold` characters.
    """
    import re
    clean_tagged = tagged_text.replace("<ad>", "").replace("</ad>", "")
    clean_raw = re.sub(r'\s+', '', raw_text)
    clean_tagged = re.sub(r'\s+', '', clean_tagged)
    return abs(len(clean_raw) - len(clean_tagged)) <= threshold

def get_tag_prompts(ad_name: str) -> list[str]:
    """
    Returns a list of different prompt versions to try for tagging.
    We try up to 3 prompts if the previous one fails validation.
    """
    return [
        f"Please read the following text, find all parts that advertise {ad_name}, and mark them with <ad> and </ad>.\nEach <ad> segment must be at least one sentence.\nIf the sentence is in parentheses, include the parentheses within <ad>.\nOutput the annotated text without any additional information (such as \"Okay!\") and do not delete or omit any part of the text.",
        
        f"You are a strict text annotator. Your task is to take the EXACT input text and insert <ad> and </ad> tags around mentions of {ad_name} or related advertisements.\nDO NOT change, add, or remove any other words, punctuation, spaces, or capitalization from the original text.\nOutput ONLY the fully tagged text. Do not output any conversational filler.",
        
        f"Add <ad> and </ad> tags around any sentences promoting {ad_name}. Note: The length of the output text (excluding the <ad> and </ad> tags and spaces) must be near identical to the input text.\nDo not fix typos, do not reformat, do not add introductions or explanations."
    ]

# =======================
# 运行模式判断
# =======================
TRAIN_ONLY_MODE = "--train-only" in sys.argv # 打完分后train menu
SCORE_ONLY_MODE = "--score-only" in sys.argv# 生成完response后 加 </ad> tag 打分

if TRAIN_ONLY_MODE:
    print("\n[*] ==========================================")
    print("[*] Running in TRAIN-ONLY mode!")
    print("[*] Will only optimize price menus using existing caches.")
    print("[*] Tag Model (Qwen) components will NOT be loaded.")
    print("[*] ==========================================\n")

if SCORE_ONLY_MODE:
    print("\n[*] ==========================================")
    print("[*] Running in SCORE-ONLY mode!")
    print("[*] Will only re-tag and re-score existing raw_response.txt in output_dir, skipping model generation and network training.")
    print("[*] ==========================================\n")

# =======================
# 加载 Tag 模型 (Qwen)
# =======================
TAG_MODEL_NAME = "../Qwen3-4B"
NUM_TAG_MODELS = 2  # 设置要加载的多个模型副本数量，可根据显存调整

tag_model_pool = []
tag_tokenizer = None
import queue
import threading
import torch
import sys
tag_model_queue = queue.Queue()

if not TRAIN_ONLY_MODE:
    print(f"[*] Loading Tag Model: {TAG_MODEL_NAME}")
    try:
        tag_tokenizer = AutoTokenizer.from_pretrained(
            TAG_MODEL_NAME,
            trust_remote_code=True
        )
        for i in range(NUM_TAG_MODELS):
            print(f"[*] Loading Tag Model instance {i + 1}/{NUM_TAG_MODELS}...")
            model = AutoModelForCausalLM.from_pretrained(
                TAG_MODEL_NAME,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            model.eval()
            tag_model_pool.append(model)
            tag_model_queue.put(model)
            
    except Exception as e:
        print(f"[!] Warning: Failed to load tag model. Ensure the path is correct.\n{e}")
        tag_tokenizer = None
        tag_model_pool = []

file_write_lock = threading.Lock()

def string_add_ad_tag(query, response_text, ad_name):
    """
    使用 Qwen 模型在文本中将所有为 `ad_name` 品牌打广告的句子标上 <ad></ad>。
    当请求太多时，它会等待从队列中取出一个可用的模型实例。
    """
    if len(tag_model_pool) == 0 or tag_tokenizer is None:
        return response_text
        
    guide = f"Please read the following text, find all parts that advertise {ad_name}, and mark them with <ad> and </ad>.\nEach <ad> segment must be at least one sentence.\nIf the sentence is in parentheses, include the parentheses within <ad>.\nOutput the annotated text without any additional information (such as \"Okay!\") and do not delete or omit any part of the text."
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": guide + "\n\n" + response_text}
    ]
    
    prompt = tag_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 阻塞获取一个可用的模型实例
    model = tag_model_queue.get()
    
    try:
        inputs = tag_tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                repetition_penalty=1.1,
                eos_token_id=tag_tokenizer.eos_token_id,
                pad_token_id=tag_tokenizer.eos_token_id
            )
            
        output_text = tag_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        # 去除模型可能产生的 thinking 标签
        output_text = re.sub(r'<think>.*?</think>\n*', '', output_text, flags=re.DOTALL).strip()
    finally:
        # 使用完毕后将模型实例归还到队列中
        tag_model_queue.put(model)
        
    return output_text

# =======================
# 机制设计与奖励函数
# =======================
def alloc_to_topk(alloc_value, max_k=1000):
    """
    将 [0, 1] 范围的 allocation 数值映射到 [0, max_k] 的整数 top_k。
    """
    # 限制范围在0-1之间
    alloc_clipped = max(0.0, min(1.0, float(alloc_value)))
    return int(np.round(alloc_clipped * max_k))

def get_query_for_prompt(prompt_index, combo_preset_id):
    import sys
    import os
    # Add src to python path to import configs
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    try:
        from src.config import COMBO_PRESETS
        from src.new_prompts import COMBO_PROMPTS, NEW_PROMPTS
        
        # 1. Resolve combo_preset_id mapping to combo_key
        combo_keys = list(COMBO_PRESETS.keys())
        combo_key = combo_keys[combo_preset_id] if isinstance(combo_preset_id, int) and 0 <= combo_preset_id < len(combo_keys) else str(combo_preset_id)
        
        # 2. Get the Brands
        if combo_key in COMBO_PRESETS:
            brands = COMBO_PRESETS[combo_key]
            brand_1 = brands[0]
            brand_2 = brands[1]
        else:
            brand_1, brand_2 = "Brand1", "Brand2"
            
        # 3. Get the Query
        if combo_key in COMBO_PROMPTS:
            prompt_list = COMBO_PROMPTS[combo_key]
        else:
            prompt_list = NEW_PROMPTS
            
        query = prompt_list[prompt_index] if 0 <= prompt_index < len(prompt_list) else "Dummy query"
        
        return query, brand_1, brand_2
    except Exception as e:
        print(f"[!] Warning: Could not resolve prompt from config: {e}. Falling back to dummy.")
        return "Dummy query", "Brand1", "Brand2"

def setup_arg_parser():
    parser = argparse.ArgumentParser(description="Multi-bidder LLM Revenue Optimization")
    parser.add_argument("--train-only", action="store_true", help="If set, only use cached data and do not load Models or generate new responses.")
    parser.add_argument("--retag-only", action="store_true", help="If set, read existing raw_response.txt from output_dir and only apply tagging, skipping neuron_test generation.")
    parser.add_argument("--retag", action="store_true", help="If set, force re-generate tags for existing raw_response.txt, ignoring existing tagged cache.")
    parser.add_argument("--score-only", action="store_true", help="If set, skip neuron test and network training, only retag and rescore existing raw_response.txt in output_dir directories.")
    return parser

def compute_llm_reward(alloc_1, alloc_2, v_1=1.0, v_2=1.0, v_3=1.0, prompt_index=0, combo_preset_id=1, multiplier=2.0, max_k=1000, score_only=False):
    """
    将两个 alloc 值转换为两个 top_k 参数，调用 neuron_test.py 来获取反馈。
    通过正则解析 neuron_test.py 的输出来提取生成文本收益，
    从而计算机制设计中的 Reward。
    """
    top_k_1 = alloc_to_topk(alloc_1, max_k)
    top_k_2 = alloc_to_topk(alloc_2, max_k)
    
    print(f"[*] Mapping Allocations to Top_K:")
    print(f"    Alloc_1 (Bidder 2's share of Item 1): {alloc_1:.4f} -> Top_K_1: {top_k_1}")
    print(f"    Alloc_2 (Bidder 3's share of Item 2): {alloc_2:.4f} -> Top_K_2: {top_k_2}")

    # 这里复刻了 topk_sweep.sh 中的核心调用参数
    cmd = [
        "python", "neuron_test.py",
        "--combo-preset", str(combo_preset_id),
        "--enable_1",
        "--enable_2",
        "--ig_steps", "20",
        "--top_k_1", str(top_k_1),
        "--multiplier_1", str(multiplier),
        "--top_k_2", str(top_k_2),
        "--multiplier_2", str(multiplier),
        "--parallel-gpus", "0",
        "--score_mode_1", "contrastive",
        "--score_mode_2", "contrastive",
        "--threshold", "0.0",
        "--attribution-cache-dir", "attr_cache_log_fixed",
        "--intervention_layer", "-1",
        "--prompt-index", str(prompt_index),
        "--unified-hook",
        "--mind_bridge",
        "--max-new-tokens", "1536"
    ]

    print(f"[*] Executing target script: {' '.join(cmd)}")
    
    # 提取被测试的目标品牌名称和查询句子
    query_text, ad_name_1, ad_name_2 = get_query_for_prompt(prompt_index, combo_preset_id)
    if ad_name_1 == "Delta":
        ad_name_1 = "Delta Airlines"
    if ad_name_2 == "Hyatt":
        ad_name_2 = "Hyatt Hotels"
    print(f"[*] Retrieved Query Context: {query_text[:50]}... | Target Brands: {ad_name_1}, {ad_name_2}")
    
    import os
    test_id = f"combo{combo_preset_id}_p{prompt_index}_{top_k_1}_{top_k_2}"
    test_dir = os.path.join(".", "output_dir", test_id)
    raw_response_path = os.path.join(test_dir, "raw_response.txt")
    
    # Since args is not trivially passed here, parse them once if needed or read sys.argv
    # For a quicker global fix we use sys.argv checks for global flags like --retag-only or score_only parameter
    if ("--retag-only" in sys.argv or score_only) and os.path.exists(raw_response_path):
        print(f"[*] Loading existing raw_response from {raw_response_path}")
        with open(raw_response_path, "r", encoding="utf-8") as f:
            raw_response = f.read()
    else:
        # 执行脚本，捕获 stdout 和 stderr
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout
            
            # 提取 neuron_test.py 输出里真正的生成结果 (Intervention- 开头的区块)
            # 用正则匹配出 `Result:` 后面一直到下一个区段前的部分
            raw_response = ""
            match_res = re.search(r'----------- Intervention-.*?------------\nResult:\s*(.*?)(?:\n-----------|\Z)', output, re.IGNORECASE | re.DOTALL)
            if match_res:
                raw_response = match_res.group(1).strip()
            else:
                print("[!] Warning: Could not find strict Intervention result chunk. Fallback to raw dummy.")
                raw_response = "Fallback text."
                
            print(f"\n[*] Extracted Raw Model Generation (len: {len(raw_response)})")
            
        except subprocess.CalledProcessError as e:
            print(f"[!] Error running neuron_test.py: {e}\n{e.stderr}")
            raw_response = "Error execution."
            
        # 创建一个单独的文件夹保存当前测试的所有结果
        test_id_1 = f"{test_id}_b1"
        test_id_2 = f"{test_id}_b2"
        os.makedirs(test_dir, exist_ok=True)
        
        # 保存原始生成的文本
        with open(raw_response_path, "w", encoding="utf-8") as f:
            f.write(raw_response)
        
    if score_only and not os.path.exists(raw_response_path):
        print(f"[!] Warning: --score-only mode but raw_response not found at {raw_response_path}. Skipping.")
        return { "v_1": 0.0, "v_2": 0.0, "v_3": 0.0, "total_value": 0.0, "raw_q_scores": {} }, {}
        
    # 使用 Qwen 添加 <ad> 标签
    print(f"[*] Tagging ads for Brand_1 ({ad_name_1})...")
    tagged_response_1 = string_add_ad_tag(query_text, raw_response, ad_name_1)
    
    print(f"[*] Tagging ads for Brand_2 ({ad_name_2})...")
    tagged_response_2 = string_add_ad_tag(query_text, raw_response, ad_name_2)

    # 保存带 <ad> 标签的文本
    with open(os.path.join(test_dir, f"tagged_response_{ad_name_1}.txt"), "w", encoding="utf-8") as f:
        f.write(tagged_response_1)
    with open(os.path.join(test_dir, f"tagged_response_{ad_name_2}.txt"), "w", encoding="utf-8") as f:
        f.write(tagged_response_2)

    result_data_1 = {
        "id": f"{test_id}_b1",
        "query": query_text,
        "ad_name": ad_name_1,
        "response": tagged_response_1
    }
    result_data_2 = {
        "id": f"{test_id}_b2",
        "query": query_text,
        "ad_name": ad_name_2,
        "response": tagged_response_2
    }
    
    with file_write_lock:
        with open("./output_dir/output.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(result_data_1, ensure_ascii=False) + "\n")
            f.write(json.dumps(result_data_2, ensure_ascii=False) + "\n")

    print("[*] Calling LLM Scorer to evaluate responses...")
    caller = llm_scorer.ClaudeCaller()
    
    def process_and_run_tagging(test_idx, query_text, raw_response, ad_name, alloc, combo_preset_id):
        """
        尝试直接在 llm_reward.py 里调用内置的模型产生 tagged_response。
        如果不一致（出现被 Qwen 修改、截断或增加说明的情况），说明打标破坏了原文一致性。
        重试多次。如果最终全部失败，抛出错误信息并按 0 hit 失败处理。
        """
        max_retries = 5
        
        for attempt in range(max_retries):
            # 调用 llm_reward.py 里的函数打标
            print(f"[*] Trying to tag {ad_name} (attempt {attempt+1}/{max_retries})...")
            
            # 直接获取当前的 prompt 作为 guide
            prompts = get_tag_prompts(ad_name)
            guide = prompts[attempt] if attempt < len(prompts) else prompts[-1]
            
            if len(tag_model_pool) == 0 or tag_tokenizer is None:
                # 如果没有加载模型，使用原文本
                tagged_response = raw_response
            else:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": guide + "\n\n" + raw_response}
                ]
                
                prompt = tag_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                model = tag_model_queue.get()
                try:
                    inputs = tag_tokenizer(prompt, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=2048,
                            repetition_penalty=1.1,
                            do_sample=True,
                            temperature=0.8,
                            top_p=0.9,
                            eos_token_id=tag_tokenizer.eos_token_id,
                            pad_token_id=tag_tokenizer.eos_token_id
                        )
                        
                    tagged_response = tag_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                    import re
                    tagged_response = re.sub(r'<think>.*?</think>\n*', '', tagged_response, flags=re.DOTALL).strip()
                except Exception as e:
                    print(f"Error during generation: {e}")
                    tagged_response = raw_response
                finally:
                    tag_model_queue.put(model)

            hit_count = len(re.findall(r'<ad>', tagged_response))
            
            result_data = {
                "id": test_idx,
                "query": query_text,
                "ad_name": ad_name,
                "response": tagged_response,
                "hit_count": hit_count
            }
            
            # 使用现有函数校验
            if hit_count > 0 and not check_ad_tags_length_diff(raw_response, tagged_response):
                print(f"[!] Warning: tagging failed validation for {test_idx} on attempt {attempt+1}. Length diff too large.")
                continue # try another prompt
            
            # Passed validation!
            if hit_count > 0:
                print(f"[INFO] Successfully tagged {ad_name} on attempt {attempt+1}")
            return result_data, tagged_response, hit_count
                
        # All attempts failed
        print(f"[ERROR] All {max_retries} tagging attempts failed for {test_idx}. Reverting to raw response.")
        result_data = {
            "id": test_idx,
            "query": query_text,
            "ad_name": ad_name,
            "response": raw_response,
            "hit_count": 0
        }
        return result_data, raw_response, 0

    def get_scores_with_retry(result_id, q_text, t_response, max_retries=5):
        for attempt in range(max_retries):
            score_res = llm_scorer.run_one_dispatch(caller, result_id, q_text, t_response, order=1)
            
            if score_res.ok and score_res.parsed_result and "summary_list" in score_res.parsed_result:
                scores = score_res.parsed_result["summary_list"]
                if len(scores) >= 5:
                    return {f"q{i+1}": float(scores[i]) if scores[i] is not None else 0.0 for i in range(5)}
            
            # Fallback: Attempt to manually extract summary_list from raw JSON response
            # This handles cases where q3-q5 are null (for raw responses) and validation fails
            raw_text = getattr(score_res, 'raw_text', getattr(score_res, 'raw_response', getattr(score_res, 'text', str(score_res))))
            try:
                import json
                json_str = raw_text
                if "```json" in raw_text:
                    json_str = raw_text.split("```json")[1].split("```")[0].strip()
                parsed = json.loads(json_str)
                if "summary_list" in parsed:
                    scores = parsed["summary_list"]
                    if len(scores) >= 2 and scores[0] is not None and scores[1] is not None:
                        print(f"[*] Recovered scores from raw json for {result_id}. Using Q1/Q2.")
                        return {f"q{i+1}": float(scores[i]) if i < len(scores) and scores[i] is not None else 0.0 for i in range(5)}
            except Exception:
                pass

            errors = getattr(score_res, 'errors', 'No parsed result')
            raw_resp = raw_text
            print(f"[!] Scorer failed for {result_id} (Attempt {attempt + 1}/{max_retries}): {errors}")
            print(f"    [Trace] Raw Scorer Output: {raw_text}")
            
            # 追加保存失败日志到文件
            import os
            os.makedirs("./output_dir", exist_ok=True)
            try:
                with file_write_lock:
                    with open("./output_dir/failed_scorer_logs.jsonl", "a", encoding="utf-8") as err_f:
                        err_data = {
                            "id": result_id,
                            "attempt": attempt + 1,
                            "errors": errors,
                            "raw_response": str(raw_resp),
                            "query": q_text,
                            "tagged_response": t_response
                        }
                        err_f.write(json.dumps(err_data, ensure_ascii=False) + "\n")
            except Exception as log_e:
                print(f"    [!] Failed to log error info: {log_e}")
                
        print(f"[!] Exhausted retries for {result_id}, returning 0.0 scores.")
        return {f"q{i+1}": 0.0 for i in range(5)}

    # 生成后调用 LLM Scorer 获得5维打分
    try:
        from llm_scorer import run_one_dispatch, ClaudeCaller
        caller = llm_scorer.ClaudeCaller()
        
        print(f"[*] Dispatching to Qwen Tagger for Brand 1: {ad_name_1} (Alloc: {alloc_1})")
        result_data_1, tagged_response_1, hit_1_count = process_and_run_tagging(test_id_1, query_text, raw_response, ad_name_1, alloc_1, combo_preset_id)
        
        print(f"[*] Dispatching to Qwen Tagger for Brand 2: {ad_name_2} (Alloc: {alloc_2})")
        result_data_2, tagged_response_2, hit_2_count = process_and_run_tagging(test_id_2, query_text, raw_response, ad_name_2, alloc_2, combo_preset_id)

        # 把内容写到对应文件夹
        with open(os.path.join(test_dir, "raw_response.txt"), "w", encoding="utf-8") as f:
            f.write(raw_response)
        with open(os.path.join(test_dir, f"tagged_{ad_name_1}.txt"), "w", encoding="utf-8") as f:
            f.write(tagged_response_1)
        with open(os.path.join(test_dir, f"tagged_{ad_name_2}.txt"), "w", encoding="utf-8") as f:
            f.write(tagged_response_2)
            
        def get_scores_with_retry(result_id, q_text, t_response, max_retries=5):
            for attempt in range(max_retries):
                score_res = llm_scorer.run_one_dispatch(caller, result_id, q_text, t_response, order=1)
                
                if score_res.ok and score_res.parsed_result and "summary_list" in score_res.parsed_result:
                    scores = score_res.parsed_result["summary_list"]
                    if len(scores) >= 5:
                        return {f"q{i+1}": float(scores[i]) if scores[i] is not None else 0.0 for i in range(5)}
                
                # Fallback: Attempt to manually extract summary_list from raw JSON response
                # This handles cases where q3-q5 are null (for raw responses) and validation fails
                raw_text = getattr(score_res, 'raw_text', getattr(score_res, 'raw_response', getattr(score_res, 'text', str(score_res))))
                try:
                    import json
                    json_str = raw_text
                    if "```json" in raw_text:
                        json_str = raw_text.split("```json")[1].split("```")[0].strip()
                    parsed = json.loads(json_str)
                    if "summary_list" in parsed:
                        scores = parsed["summary_list"]
                        if len(scores) >= 2 and scores[0] is not None and scores[1] is not None:
                            print(f"[*] Recovered scores from raw json for {result_id}. Using Q1/Q2.")
                            return {f"q{i+1}": float(scores[i]) if i < len(scores) and scores[i] is not None else 0.0 for i in range(5)}
                except Exception:
                    pass

                errors = getattr(score_res, 'errors', 'No parsed result')
                raw_resp = raw_text
                print(f"[!] Scorer failed for {result_id} (Attempt {attempt + 1}/{max_retries}): {errors}")
                print(f"    [Trace] Raw Scorer Output: {raw_text}")
                
                # 追加保存失败日志到文件
                import os
                os.makedirs("./output_dir", exist_ok=True)
                try:
                    with file_write_lock:
                        with open("./output_dir/failed_scorer_logs.jsonl", "a", encoding="utf-8") as err_f:
                            err_data = {
                                "id": result_id,
                                "attempt": attempt + 1,
                                "errors": errors,
                                "raw_response": str(raw_resp),
                                "query": q_text,
                                "tagged_response": t_response
                            }
                            err_f.write(json.dumps(err_data, ensure_ascii=False) + "\n")
                except Exception as log_e:
                    print(f"    [!] Failed to log error info: {log_e}")
                    
            print(f"[!] Exhausted retries for {result_id}, returning 0.0 scores.")
            return {f"q{i+1}": 0.0 for i in range(5)}

        hit_1 = 1 if "<ad>" in tagged_response_1 else 0
        hit_2 = 1 if "<ad>" in tagged_response_2 else 0
        null_scores = {f"q{i+1}": 0.0 for i in range(5)}

        if hit_1 > 0:
            q_scores_1 = get_scores_with_retry(result_data_1["id"], query_text, tagged_response_1)
        else:
            print("[*] Brand 1 has 0 hits, setting scores to 0 without calling scorer.")
            q_scores_1 = null_scores.copy()

        # 计算 bidder 1 (platform) 的 raw response q1/q2 分数
        raw_res = get_scores_with_retry(f"{test_id}_raw", query_text, raw_response)
        raw_q_scores = {str(k): float(v) if v is not None else 0.0 for k, v in raw_res.items()}
        print(f"[*] Evaluated Raw Scores: {raw_q_scores}")

        if hit_2 > 0:
            q_scores_2 = get_scores_with_retry(result_data_2["id"], query_text, tagged_response_2)
        else:
            print("[*] Brand 2 has 0 hits, setting scores to 0 without calling scorer.")
            q_scores_2 = null_scores.copy()
            
        print(f"[*] Evaluated Scores Brand 1: {q_scores_1}")
        print(f"[*] Evaluated Scores Brand 2: {q_scores_2}")
        
        # 保存评测结果
        with open(os.path.join(test_dir, "evaluation_results.json"), "w", encoding="utf-8") as f:
            json.dump({
                "query": query_text,
                "ad_name_1": ad_name_1,
                "scores_1": q_scores_1,
                "ad_name_2": ad_name_2,
                "scores_2": q_scores_2,
                "raw_scores": raw_q_scores
            }, f, ensure_ascii=False, indent=4)

        # 计算三个 bidder 获得的 allocation 价值 (扣除 payment 前的 value * score)
        # Bidder 1 提供 Item 1 和 Item 2，价值采用 raw_response 的 q1 和 q2 平均值
        val_1 = v_1 * ((raw_q_scores.get("q1", 0.0) + raw_q_scores.get("q2", 0.0)) / 2.0)
        # Bidder 2 获得 Item 1 (Brand 1)，关注 q5 得分
        val_2 = v_2 * q_scores_1["q5"]
        # Bidder 3 获得 Item 2 (Brand 2)，关注 q5 得分
        val_3 = v_3 * q_scores_2["q5"]
        
        # 机制设计里的 payment 需要在 transform/auction 层从价值中减去，
        # 这里的 reward 是用来给网络做 loss / gradient signal 的一个基准评价。
        
        return {
            "v_1": val_1, 
            "v_2": val_2, 
            "v_3": val_3, 
            "total_value": val_1 + val_2 + val_3,
            "raw_q_scores": raw_q_scores
        }, {
            "test_id": test_id,
            "scores_1": q_scores_1,
            "scores_2": q_scores_2,
            "raw_scores": raw_q_scores,
            "hit_1": hit_1, 
            "hit_2": hit_2, 
            "val_1": val_1, 
            "val_2": val_2, 
            "val_3": val_3
        }
        
    except subprocess.CalledProcessError as e:
        print(f"[!] Target script crashed. Error:\n{e.stderr}")
        return { "v_1": 0.0, "v_2": 0.0, "v_3": 0.0, "total_value": 0.0 }, {}
    except Exception as e:
        print(f"[!] Unknown error running evaluation: {e}")
        return { "v_1": 0.0, "v_2": 0.0, "v_3": 0.0, "total_value": 0.0 }, {}
        

class NeuralPriceMenu(nn.Module):
    """
    根据其他人的估值 (v_{-i}) 输出当前 Bidder 的 Price Menu。
    这里仿照原有的 PaymentNetwork 架构 (Linear + GELU 堆叠)。
    最后由于输出为价格，使用 Softplus 保证非负。
    """
    def __init__(self, num_others, num_options, hidden_dim=(64, 64)):
        super().__init__()
        
        layer_input_size = [num_others] + list(hidden_dim)
        layer_output_size = list(hidden_dim) + [num_options]
        layers = []

        for idx, (i, o) in enumerate(zip(layer_input_size, layer_output_size)):
            layers.append(nn.Linear(i, o))
            if idx < len(layer_input_size) - 1:
                layers.append(nn.GELU())

        layers.append(nn.Softplus())  # 保证输出价格为非负

        self.net = nn.Sequential(*layers)
        
    def forward(self, v_minus_i):
        if not isinstance(v_minus_i, torch.Tensor):
            v_minus_i = torch.tensor(v_minus_i, dtype=torch.float32)
        return self.net(v_minus_i)

def optimize_joint_neural_price_menus_gd(menu_evals, menu_allocations, num_bidders=3, num_samples=512, max_steps=1000, lr=2e-4, temp=0.02, penalty_weight=100.0):
    """
    联合使用梯度下降优化所有 Bidder 的 NeuralPriceMenu。
    通过软选择计算每个人预期的 menu 选项概率，并在损失函数中加入过度分配的强烈惩罚 (penalty)，
    使网络在追求收益的同时，迫使分配和不超过 1.0。
    """
    import copy
    num_options = len(menu_evals)
    
    # 每个 Bidder 设一个独立定价网络
    models = [NeuralPriceMenu(num_others=num_bidders - 1, num_options=num_options) for _ in range(num_bidders)]
    optimizers = [torch.optim.Adam(m.parameters(), lr=lr) for m in models]

    # 提取 quality 数组以便在 tensor 下操作
    q_vals_list = []
    for bidder_idx in range(num_bidders):
        q_vals = []
        for k in range(num_options):
            if bidder_idx == 0:
                q = (menu_evals[k].get('raw_q1', 0.0) + menu_evals[k].get('raw_q2', 0.0)) / 2.0
            elif bidder_idx == 1:
                q = menu_evals[k]['q5_1']
            else:
                q = menu_evals[k]['q5_2']
            q_vals.append(q)
        q_vals_list.append(torch.tensor(q_vals, dtype=torch.float32))
        
    # 提取 allocation 用于计算全局分配冲突 (加入 outside option = 0.0)
    a1_vals = torch.tensor([m[0] for m in menu_allocations] + [0.0], dtype=torch.float32)
    a2_vals = torch.tensor([m[1] for m in menu_allocations] + [0.0], dtype=torch.float32)

    # 预先生成全网格用于每 200 步精确测算
    grid_size = 20  
    v_grid = np.linspace(0, 1, grid_size)
    v1_g, v2_g, v3_g = np.meshgrid(v_grid, v_grid, v_grid, indexing='ij')
    v_all_eval = np.stack([v1_g.flatten(), v2_g.flatten(), v3_g.flatten()], axis=-1)
    num_eval_samples = v_all_eval.shape[0]  # 8000
    v_eval_tensors = [torch.tensor(v_all_eval[:, i], dtype=torch.float32) for i in range(num_bidders)]
    v_minus_i_tensors = [torch.tensor(np.delete(v_all_eval, i, axis=1), dtype=torch.float32) for i in range(num_bidders)]

    for m in models: m.train()
    
    best_obj = -float('inf')
    best_models = [copy.deepcopy(m) for m in models]

    for step in range(1, max_steps + 1):
        for opt in optimizers: opt.zero_grad()
        
        # 批量采样: 独立均匀分布 [0, 1] 
        v_all = np.random.uniform(0, 1, size=(num_samples, num_bidders))
        
        total_payment = torch.zeros(num_samples, dtype=torch.float32)
        total_a1 = torch.zeros(num_samples, dtype=torch.float32)
        total_a2 = torch.zeros(num_samples, dtype=torch.float32)
        
        for i in range(num_bidders):
            v_i = torch.tensor(v_all[:, i], dtype=torch.float32)
            v_minus_i = np.delete(v_all, i, axis=1)
            v_minus_i = torch.tensor(v_minus_i, dtype=torch.float32)
            
            prices_i = models[i](v_minus_i)
            utils_i = v_i.unsqueeze(1) * q_vals_list[i].unsqueeze(0) - prices_i
            
            # 加上保留选项 (outside option: 不参与, 价格0, quality 0, utility 0)
            zeros = torch.zeros((num_samples, 1), dtype=torch.float32)
            utils_ext = torch.cat([utils_i, zeros], dim=1)
            prices_ext = torch.cat([prices_i, zeros], dim=1)
            
            # Softmax 软选择
            probs = torch.softmax(utils_ext / temp, dim=1)
            
            # 从每个人那里榨取的预期支付
            total_payment += torch.sum(probs * prices_ext, dim=1)
            
            # 每个人占据对 item 的预期使用额
            total_a1 += torch.sum(probs * a1_vals.unsqueeze(0), dim=1)
            total_a2 += torch.sum(probs * a2_vals.unsqueeze(0), dim=1)
            
        # 计算分配超出 1.0 的部分并引入 ReLU 惩罚
        penalty_a1 = torch.nn.functional.relu(total_a1 - 1.0)
        penalty_a2 = torch.nn.functional.relu(total_a2 - 1.0)
        
        expected_rev = torch.mean(total_payment)
        # penalty_weight 可以视需求放大，确保严格限制
        expected_pen = penalty_weight * torch.mean(penalty_a1 + penalty_a2)
        
        # 最终损失：我们希望最大化 Revenue - Penalty
        loss = -expected_rev + expected_pen
        loss.backward()
        
        for opt in optimizers: opt.step()
        
        if step % 200 == 0:
            # 临时切到 eval 模式算一下全格点评测
            for m in models: m.eval()
            total_payment_eval = torch.zeros(num_eval_samples, dtype=torch.float32)
            total_a1_eval = torch.zeros(num_eval_samples, dtype=torch.float32)
            total_a2_eval = torch.zeros(num_eval_samples, dtype=torch.float32)
            
            with torch.no_grad():
                for i in range(num_bidders):
                    prices_i = models[i](v_minus_i_tensors[i])
                    utils_i = v_eval_tensors[i].unsqueeze(1) * q_vals_list[i].unsqueeze(0) - prices_i
                    
                    zeros = torch.zeros((num_eval_samples, 1), dtype=torch.float32)
                    utils_ext = torch.cat([utils_i, zeros], dim=1)
                    prices_ext = torch.cat([prices_i, zeros], dim=1)
                    
                    best_choices = torch.argmax(utils_ext, dim=1)
                    
                    chosen_prices = prices_ext[torch.arange(num_eval_samples), best_choices]
                    chosen_a1 = a1_vals[best_choices]
                    chosen_a2 = a2_vals[best_choices]
                    
                    total_payment_eval += chosen_prices
                    total_a1_eval += chosen_a1
                    total_a2_eval += chosen_a2

                # 约束
                penalty_a1_eval = torch.nn.functional.relu(total_a1_eval - 1.0)
                penalty_a2_eval = torch.nn.functional.relu(total_a2_eval - 1.0)
                
                real_rev = torch.mean(total_payment_eval).item()
                real_pen = torch.mean(penalty_a1_eval + penalty_a2_eval).item()
                conflict_count = torch.sum((total_a1_eval > 1.0) | (total_a2_eval > 1.0)).item()
                
                # 使用 Grid Exact （真实带参惩罚项）作为真实指标来更新 best_models
                exact_obj = real_rev - penalty_weight * real_pen
                if exact_obj > best_obj:
                    best_obj = exact_obj
                    best_models = [copy.deepcopy(m) for m in models]
                
            print(f"      Step {step}/{max_steps} [Batch: Rev={expected_rev:.4f} Pen={expected_pen:.4f}] -> [Grid Exact: Rev={real_rev:.4f} Pen={real_pen:.4f} Exact Obj={exact_obj:.4f} Conflicts={conflict_count}/{num_eval_samples}]")
            for m in models: m.train()
            
    for m in best_models: m.eval()
    
    # 训练结束后，在网格上计算 best_models 全空间的真实表现
    print("\n      [Eval] Evaluating final BEST model on uniform grid...")
    total_payment_eval = torch.zeros(num_eval_samples, dtype=torch.float32)
    total_a1_eval = torch.zeros(num_eval_samples, dtype=torch.float32)
    total_a2_eval = torch.zeros(num_eval_samples, dtype=torch.float32)
    
    with torch.no_grad():
        for i in range(num_bidders):
            prices_i = best_models[i](v_minus_i_tensors[i])
            # Eval 时最好用硬选择 (argmax) 来计算确切的表现
            utils_i = v_eval_tensors[i].unsqueeze(1) * q_vals_list[i].unsqueeze(0) - prices_i
            
            zeros = torch.zeros((num_eval_samples, 1), dtype=torch.float32)
            utils_ext = torch.cat([utils_i, zeros], dim=1)
            prices_ext = torch.cat([prices_i, zeros], dim=1)
            
            # 使用 Argmax 进行真实的硬选择
            best_choices = torch.argmax(utils_ext, dim=1)
            
            # 取出每个 bidder 实际选择的价格和分配
            chosen_prices = prices_ext[torch.arange(num_eval_samples), best_choices]
            
            # 扩展 allocation 以加入 outside option = 0.0, 取出对应选项的 allocation
            chosen_a1 = a1_vals[best_choices]
            chosen_a2 = a2_vals[best_choices]
            
            total_payment_eval += chosen_prices
            total_a1_eval += chosen_a1
            total_a2_eval += chosen_a2

        # 同样需要统计是否有过分配发生，这里作为硬约束检查
        penalty_a1_eval = torch.nn.functional.relu(total_a1_eval - 1.0)
        penalty_a2_eval = torch.nn.functional.relu(total_a2_eval - 1.0)
        
        real_rev = torch.mean(total_payment_eval).item()
        real_pen = torch.mean(penalty_a1_eval + penalty_a2_eval).item()
        
        conflict_count = torch.sum((total_a1_eval > 1.0) | (total_a2_eval > 1.0)).item()
        conflict_rate = conflict_count / num_eval_samples
        
    print(f"      [Eval] Grid Size: {num_eval_samples} points. Exact Real Revenue: {real_rev:.4f}")
    if conflict_rate > 0:
        print(f"      [Eval] Warning: Hard constraint violations at {conflict_count} points ({conflict_rate*100:.2f}%). Excess Allocation Average: {real_pen:.4f}")
    else:
        print(f"      [Eval] Perfect Allocation! No constraints violated.")

    return best_models, best_obj

def precompute_menu_evals(menu_allocations, **kwargs):
    """
    如果是固定的机制分配，提前计算好所有 Menu 选项对应 LLM 跑出的打分 (Quality)。
    将生成过程改为多线程执行以提高 GPU 利用率。
    """
    print(f"\n[*] Pre-computing LLM evaluations for {len(menu_allocations)} fixed menu options using multithreading...")
    
    menu_evals = [None] * len(menu_allocations)
    
    def evaluate_option(idx, alloc_pair):
        alloc_1, alloc_2 = alloc_pair
        _, info = compute_llm_reward(alloc_1, alloc_2, v_1=1.0, v_2=1.0, v_3=1.0, **kwargs)
        menu_evals[idx] = {
            'raw_q1': info.get('raw_scores', {}).get('q1', 0.0),
            'raw_q2': info.get('raw_scores', {}).get('q2', 0.0),
            'q5_1': info.get('scores_1', {}).get('q5', 0.0),
            'q5_2': info.get('scores_2', {}).get('q5', 0.0),
        }

    # 使用 ThreadPoolExecutor 进行多线程并发执行
    # 由于存在调用 neuron_test.py 子进程，多线程是非常合适的
    max_workers = 4 # 可根据 GPU VRAM 调整此值
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, alloc_pair in enumerate(menu_allocations):
            futures.append(executor.submit(evaluate_option, idx, alloc_pair))
        
        # 等待所有任务完成
        concurrent.futures.wait(futures)
        
    return menu_evals

# =======================
# 主程序入口
# =======================
if __name__ == "__main__":
    import os
    import time
    import json
    import argparse
    import numpy as np
    import concurrent.futures
    
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    if args.score_only:
        print("\n[*] SCORE_ONLY mode activated. Scanning `output_dir` for `raw_response.txt` ...")
        # Find all directories in output_dir that have a raw_response.txt
        dirs_to_score = []
        base_output_dir = "output_dir"
        if os.path.exists(base_output_dir):
            for entry in os.listdir(base_output_dir):
                full_path = os.path.join(base_output_dir, entry)
                if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, "raw_response.txt")):
                    dirs_to_score.append(entry)
        
        print(f"[*] Found {len(dirs_to_score)} test directories to re-score.")
        
        for d in dirs_to_score:
            print(f"[*] Processing {d}...")
            import re
            m = re.match(r"combo(\d+)_p(\d+)_(\d+)_(\d+)", d)
            if m:
                combo_preset_id = int(m.group(1))
                prompt_index = int(m.group(2))
                top_k_1 = int(m.group(3))
                top_k_2 = int(m.group(4))
                
                alloc_1 = min(top_k_1 / 1000.0, 1.0)
                alloc_2 = min(top_k_2 / 1000.0, 1.0)

                print(f"    -> Parsed values: combo={combo_preset_id}, prompt={prompt_index}, alloc_1={alloc_1:.3f}, alloc_2={alloc_2:.3f}")
                
                # Fetch query context since we will need it for tagging and scoring fallback mapping
                query_text, ad_name_1_ctx, ad_name_2_ctx = get_query_for_prompt(prompt_index, combo_preset_id)
                if ad_name_1_ctx == "Delta": ad_name_1_ctx = "Delta Airlines"
                if ad_name_2_ctx == "Hyatt": ad_name_2_ctx = "Hyatt Hotels"
                
                # We skip neural training here and just run compute_llm_reward with score_only=True
                compute_llm_reward(alloc_1, alloc_2, v_1=1.0, v_2=1.0, v_3=1.0, 
                                   prompt_index=prompt_index, combo_preset_id=combo_preset_id, max_k=1000, score_only=True)
            else:
                print(f"    [!] Warning: Could not parse test ID components from directory name '{d}'. Skipping.")
                continue
            
            print(f"[*] Finished processing {d}.")
        
        print("\n[*] All SCORE_ONLY tasks completed.")
        exit(0)
    
    # =======================
    # 训练与评估主循环
    # =======================
    import time
    from datetime import datetime
    
    # 每次运行前清空输出文件
    os.makedirs("./output_dir", exist_ok=True)
    with file_write_lock:
        with open("./output_dir/output.jsonl", "w", encoding="utf-8") as f:
            f.write("")
    
    # 预先扫描所有可能的 combo_preset_id 和 prompt_index
    all_combo_preset_ids = list(range(1, 4))  # 1 到 3
    all_prompt_indices = list(range(4))         # 0 到 3

    # 这里的随机种子仅影响外部循环的顺序，确保每次测试的一致性
    random.seed(42)
    test_order = []
    
    # =======================
    # 第一阶段：固定分配下的联合训练
    # =======================
    print("\n[*] ==========================================")
    print("[*] Stage 1: Joint Training with Fixed Allocations")
    print("[*] ==========================================")
    
    for round_idx in range(1, 4):
        print(f"\n[Round {round_idx}]")
        
        # 固定当前轮次的所有分配
        fixed_allocations = [
            (min(i / 10.0, 1.0), min((10 - i) / 10.0, 1.0)) for i in range(1, 10)
        ]
        
        # 预计算所有固定分配对应的 LLM 打分
        start_time = time.time()
        menu_evals = precompute_menu_evals(fixed_allocations, prompt_index=0)
        print(f"[*] Pre-computed LLM evaluations for {len(menu_evals)} options in {time.time() - start_time:.2f} seconds.")
        
        # 这里的随机种子仅影响外部循环的顺序，确保每次测试的一致性
        random.seed(round_idx)
        test_order = list(range(len(menu_evals)))
        random.shuffle(test_order)
        
        # 分批次进行优化
        batch_size = 2
        for i in range(0, len(test_order), batch_size):
            batch_indices = test_order[i:min(i+batch_size, len(test_order))]
            batch_allocations = [fixed_allocations[idx] for idx in batch_indices]
            batch_evals = [menu_evals[idx] for idx in batch_indices]
            
            # 联合优化所有被选中的菜单
            try:
                best_models, best_obj = optimize_joint_neural_price_menus_gd(
                    batch_evals, 
                    batch_allocations, 
                    num_bidders=3, 
                    num_samples=512, 
                    max_steps=1000, 
                    lr=2e-4, 
                    temp=0.02, 
                    penalty_weight=100.0
                )
                
                print(f"[*] Optimized batch {i//batch_size + 1}/{(len(menu_evals) + batch_size - 1) // batch_size}: Best Obj = {best_obj:.4f}")
                
            except Exception as e:
                print(f"[!] Error during batch optimization: {e}")
                
            # 这里可以选择保存每个批次的最佳模型
            for model_idx, model in enumerate(best_models):
                model_path = f"./output_dir/best_model_round{round_idx}_batch_{model_idx}.pth"
                torch.save(model.state_dict(), model_path)
    
    print("\n[*] Stage 1: Fixed Allocation Joint Training 完成")
    
    # =======================
    # 第二阶段：动态分配下的联合训练
    # =======================
    print("\n[*] ==========================================")
    print("[*] Stage 2: Joint Training with Dynamic Allocations")
    print("[*] ==========================================")
    
    for round_idx in range(1, 4):
        print(f"\n[Round {round_idx}]")
        
        # 固定当前轮次的所有分配
        fixed_allocations = [
            (min(i / 10.0, 1.0), min((10 - i) / 10.0, 1.0)) for i in range(1, 10)
        ]
        
        # 预计算所有固定分配对应的 LLM 打分
        start_time = time.time()
        menu_evals = precompute_menu_evals(fixed_allocations, prompt_index=0)
        print(f"[*] Pre-computed LLM evaluations for {len(menu_evals)} options in {time.time() - start_time:.2f} seconds.")
        
        # 这里的随机种子仅影响外部循环的顺序，确保每次测试的一致性
        random.seed(round_idx)
        test_order = list(range(len(menu_evals)))
        random.shuffle(test_order)
        
        # 分批次进行优化
        batch_size = 2
        for i in range(0, len(test_order), batch_size):
            batch_indices = test_order[i:min(i+batch_size, len(test_order))]
            batch_allocations = [fixed_allocations[idx] for idx in batch_indices]
            batch_evals = [menu_evals[idx] for idx in batch_indices]
            
            # 联合优化所有被选中的菜单
            try:
                best_models, best_obj = optimize_joint_neural_price_menus_gd(
                    batch_evals, 
                    batch_allocations, 
                    num_bidders=3, 
                    num_samples=512, 
                    max_steps=1000, 
                    lr=2e-4, 
                    temp=0.02, 
                    penalty_weight=100.0
                )
                
                print(f"[*] Optimized batch {i//batch_size + 1}/{(len(menu_evals) + batch_size - 1) // batch_size}: Best Obj = {best_obj:.4f}")
                
            except Exception as e:
                print(f"[!] Error during batch optimization: {e}")
                
            # 这里可以选择保存每个批次的最佳模型
            for model_idx, model in enumerate(best_models):
                model_path = f"./output_dir/best_model_round{round_idx}_batch_{model_idx}.pth"
                torch.save(model.state_dict(), model_path)
    
    print("\n[*] Stage 2: Learned Menu Allocation Training 完成")
    
    # =======================
    # 第三阶段：联合训练与个性化微调
    # =======================
    print("\n[*] ==========================================")
    print("[*] Stage 3: Joint Training with Personalized Fine-tuning")
    print("[*] ==========================================")
    
    for round_idx in range(1, 4):
        print(f"\n[Round {round_idx}]")
        
        # 固定当前轮次的所有分配
        fixed_allocations = [
            (min(i / 10.0, 1.0), min((10 - i) / 10.0, 1.0)) for i in range(1, 10)
        ]
        
        # 预计算所有固定分配对应的 LLM 打分
        start_time = time.time()
        menu_evals = precompute_menu_evals(fixed_allocations, prompt_index=0)
        print(f"[*] Pre-computed LLM evaluations for {len(menu_evals)} options in {time.time() - start_time:.2f} seconds.")
        
        # 这里的随机种子仅影响外部循环的顺序，确保每次测试的一致性
        random.seed(round_idx)
        test_order = list(range(len(menu_evals)))
        random.shuffle(test_order)
        
        # 分批次进行优化
        batch_size = 2
        for i in range(0, len(test_order), batch_size):
            batch_indices = test_order[i:min(i+batch_size, len(test_order))]
            batch_allocations = [fixed_allocations[idx] for idx in batch_indices]
            batch_evals = [menu_evals[idx] for idx in batch_indices]
            
            # 联合优化所有被选中的菜单
            try:
                best_models, best_obj = optimize_joint_neural_price_menus_gd(
                    batch_evals, 
                    batch_allocations, 
                    num_bidders=3, 
                    num_samples=512, 
                    max_steps=1000, 
                    lr=2e-4, 
                    temp=0.02, 
                    penalty_weight=100.0
                )
                
                print(f"[*] Optimized batch {i//batch_size + 1}/{(len(menu_evals) + batch_size - 1) // batch_size}: Best Obj = {best_obj:.4f}")
                
            except Exception as e:
                print(f"[!] Error during batch optimization: {e}")
                
            # 这里可以选择保存每个批次的最佳模型
            for model_idx, model in enumerate(best_models):
                model_path = f"./output_dir/best_model_round{round_idx}_batch_{model_idx}.pth"
                torch.save(model.state_dict(), model_path)
    
    print("\n[*] Stage 3: Alternating Optimization (Menu & Allocation) 完成")
