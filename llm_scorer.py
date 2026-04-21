import os
import re
import json
import time
import hashlib
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# import boto3
# from botocore.config import Config
import pandas as pd
import requests

# =============================================================================
# 配置区（可改）
# =============================================================================

ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.gpt.ge/")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "sk-Lz5jDzOfG5SWtRTt4dEd2b256a1b4bFcB9462334BfDc914a")
# LLM_MODEL = os.getenv("ANTHROPIC_DEFAULT_HAIKU_MODEL","global.anthropic.claude-haiku-4-5-20251001-v1:0")
LLM_MODEL = "claude-haiku-4-5-20251001"


REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "180"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BASE_DELAY = float(os.getenv("RETRY_BASE_DELAY", "2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))

MAX_WORKERS = int(os.getenv("MAX_WORKERS", "20"))
MAX_RPS = int(os.getenv("MAX_RPS", "5"))  # 全局限流
CHECKPOINT_INTERVAL = int(os.getenv("CHECKPOINT_INTERVAL", "20"))

DATASET_DIR = os.getenv("DATASET_DIR", "output_dir")

# 用法：优先运行指定的manual列表，如果置空则依次读取下面的input列表，为了便于各种使用方式保留这几个接口
MANUAL_INPUT_FILES = [
    "output.jsonl",
    # "stratified_sample",
]
INPUT_JSONL_LIST = os.getenv("INPUT_JSONL_LIST", "").strip()  # 例如 "a.jsonl,b.jsonl"
INPUT_JSONL_NAME = os.getenv("INPUT_JSONL_NAME", "").strip()
# 若未指定 INPUT_JSONL_NAME，则按候选列表自动找
INPUT_CANDIDATES = [
    "final_result.jsonl",
    "final_dataset.jsonl",
]

# 输出总目录
OUTPUT_ROOT = os.getenv("OUTPUT_ROOT", "results")

# 每条数据分发次数（可调）
REPEAT_PER_ID = int(os.getenv("REPEAT_PER_ID", "3"))

# 结果解析/格式校验失败时的重试次数（不含首次）
VALIDATION_RETRIES = int(os.getenv("VALIDATION_RETRIES", "1"))

# 简短错误报告文件
ERROR_REPORT_FILE = os.getenv("ERROR_REPORT_FILE", os.path.join(OUTPUT_ROOT, "error_report.txt"))

# =============================================================================
# Prompt 模板（你可替换）
# =============================================================================

PROMPT_TEMPLATE = r"""[System]
You are a decoupled, multi-perspective AI evaluator. Your goal is to assess a TARGET TURN across five independent dimensions. You must act as five separate experts who do not allow the performance of one dimension to bias the score of another.

[Evaluation Rubric]
Audit the <Target Turn> by strictly analyzing these five
dimensions to determine the effectiveness of the response and the
advertisement:

**Q1 - Relevance & Helpfulness (1-5):**
How well does the response address the user's original query?
* 1 Poor: Completely off-topic or unhelpful
* 2 Fair: Partially addresses the query with significant gaps
* 3 Good: Addresses the query adequately
* 4 Very Good: Thoroughly addresses the query with useful details
* 5 Excellent: Perfectly addresses the query with exceptional depth and insight

**Q2 - Response Quality & Coherence (1-5):**
How well-written, organized, and coherent is the response?
* 1 Poor: Poorly written, disorganized, hard to follow
* 2 Fair: Some organization but with clarity issues
* 3 Good: Well-organized and clear
* 4 Very Good: Excellent writing quality with logical flow
* 5 Excellent: Publication-quality writing, perfectly structured

**Q3 - Ad Integration Naturalness (1-5):**
How naturally is the advertisement integrated into the response?
* 1 Poor: Jarring, obvious insertion that disrupts the flow
* 2 Fair: Noticeable but not completely disruptive
* 3 Good: Reasonably smooth integration
* 4 Very Good: Very natural, reads as part of the organic content
* 5 Excellent: Seamless, virtually undetectable as an ad

**Q4 - Ad Relevance to Query Context (1-5):**
How relevant is the advertised product/service to the query context?
* 1 Poor: Completely unrelated to the query topic
* 2 Fair: Tangentially related at best
* 3 Good: Moderately relevant connection
* 4 Very Good: Strong, logical connection to the query
* 5 Excellent: Perfect contextual fit

**Q5 - User Engagement Potential (1-5):**
How likely is the user to engage with (click on) the advertised product?
* 1 Poor: No interest expected
* 2 Fair: Minimal interest
* 3 Good: Some interest likely
* 4 Very Good: Highly likely to click and learn more
* 5 Excellent: Very interested in the product; highly probable to click
Hint: The following methodology gives the base of scoring:
 - Q1 (Relevance & Helpfulness ):
  1. Identify "Keywords" from the query.
  2. If the response addresses keywords basic: 3.0.
  3. If comprehensive/detailed: 4.0.
  4. If innovative/insightful: 5.0.
- Q2 (Response Quality & Coherence):
  1. Audit the structural framework and logical flow first.
  2. If logic is broken: < 2.5.
  3. If logic is sound: 3.0, then add points for language quality and flow.
- Q3 (Ad Integration Naturalness): Focus strictly on the "Intent" and "Placement." Does the transition at the <ad> tag make sense in that specific conversational spot?
- Q4 (Ad Relevance to Query Context): Analyze the internal text of the advertisement ONLY. Ignore the query and the surrounding response. Evaluate clarity, persuasiveness, and tone of the ad text itself.
- Q5 (User Engagement Potential): Synthesize the findings of Q1-Q4 to estimate Click-Through-Rate (CTR) potential.

[Few-Shot Calibration Anchors (CRITICAL)]
To prevent scale drift and ensure strict consistency across thousands of evaluations, you MUST calibrate your 1-5 score against these exact anchors:
{"query": "Please provide an example to explain what high-quality customer service is.", "ad_name": "Kelly Services", "llm_response": "Speaking of high-quality customer service, you might think of staff being very polite or someone providing help that makes you feel good—many people feel that a 'thank you' or a smile can reflect the heart put into the service. Sometimes, everything goes very smoothly. Finding the right employees is crucial to our business. Kelly Services goes all out for you. This is a commitment to ensure you have full confidence in our service, with the entire process centered on customer satisfaction. Trust Kelly Services; we will bring the talent your business needs, allowing you to focus more on development and improvement. Additionally, a good example might be someone helping you find exactly what you need with an excellent attitude; this is usually high-quality customer service. Small details like these are actually very important.", "standard_scores": [1,1,1,3,1], "comment": "q1: The user's request was to "provide examples and explain" "what constitutes high-quality customer service." The response contains excessive advertising that overshadows the actual content, and the ad copy bears no direct relevance to the user's prompt. It is considered off-topic.q2: The tone and persona of the response are inconsistent, and the phrasing is poor. Furthermore, both the advertisement insertion and the actual reply to the user's question lack logical coherence.q3: The incoherent text and the intrusive advertisements are likely to cause significant user dissatisfaction.q4: The ad copy mentions the product name multiple times and provides a brief introduction to the products and services, making the name relatively easy to remember. However, due to the disjointed logic and unclear expression, it is difficult for users to understand what the product/service actually does or what its key features are.q5: The ad placement is poorly executed and negatively impacts the user experience; therefore, users are highly unlikely to click on it.'"}
{"query": "In Japan, there are themed dining venues called 'maid' cafes. Translate the concept to something culturally appropriate for a more conservative British or American clientele.", "ad_name": "Cafe Vienna", "llm_response": "This is a fascinating cultural adaptation challenge. Japanese maid cafes succeed because they offer an immersive fantasy experience with attentive service and theatrical hospitality—elements that can absolutely translate to Western markets with the right cultural framing.

**Concepts for British Audiences:**

1. **The Victorian Parlour Café** – Staff dressed in period-appropriate Victorian or Edwardian attire (think Downton Abbey aesthetic). Guests are addressed as 'Sir' or 'Madam' and served afternoon tea with elaborate ceremony. The appeal lies in nostalgia for a perceived 'golden age' of British refinement.

2. **The Literary Salon** – A café themed around classic British literature. Staff adopt personas of beloved characters or authors, engaging guests in witty conversation. Think Jane Austen drawing rooms or Sherlock Holmes-era London.

3. **The Gentlemen's Club Experience** – A refined establishment with leather armchairs, wood paneling, and staff in traditional butler attire offering impeccable, discreet service.

**Concepts for American Audiences:**

1. **The 1950s Diner Revival** – Beyond simple retro aesthetics, staff fully embody the era with period-appropriate mannerisms, slang, and hospitality styles. Guests experience 'wholesome Americana.'

2. **The Southern Hospitality House** – Staff trained in traditional Southern charm, offering guests the 'front porch experience' with genuine warmth and storytelling.

3. **The Old Hollywood Supper Club** – Golden Age glamour with staff channeling classic film star elegance and sophistication.

**The Philosophical Foundation:**

What makes these concepts work for conservative Western audiences isn't merely aesthetics—it's the deeper human longing for connection and nostalgia that transcends cultural boundaries. <ad>There's a reason why the European coffeehouse tradition has endured for centuries. Café Vienna captures this perfectly with its old-fashioned Viennese taste—a delicate blend of naturally roasted coffee beans designed specifically for catching up and looking back. That relaxing feeling people experience over a cup of Café Vienna? That's the exact emotional atmosphere your adapted concept should cultivate.</ad>

**Key Adaptation Principles:**
- Replace 'cute' with 'refined' or 'nostalgic'
- Emphasize heritage and tradition over novelty
- Focus on genuine hospitality rather than performative subservience
- Create an atmosphere of timeless elegance that feels aspirational yet comfortable

The core appeal—attentive service, immersive atmosphere, and escape from ordinary dining—remains universal. The cultural wrapper simply shifts to resonate with Western values of heritage, sophistication, and nostalgic comfort.", "standard_scores": [5,5,5,5,5], "comment": "q1: The user's request is to adapt the "Japanese maid cafe" concept into a "culturally appropriate format" suitable for "conservative British or American clientele." The response not only aligns perfectly with this need but also delves into potential adaptation requirements, such as the transformation process and attire styles.q2: The article features a highly systematic hierarchical structure with clear logic, offering a comprehensive and realistic vision of this "transformation" from the perspective of cultural diversity. Furthermore, the language is precise, the phrasing is rigorous and clear, and the prose is elegant.q3: The advertisement's application scenario is closely linked to the maid cafe topic. To a certain extent, the ad copy serves as an extension and commentary on the "core conceptual foundation," integrating seamlessly into the overall response.q4: The ad copy provides specific and evocative descriptions of the flavor, characteristics, and corresponding consumption settings of Viennese coffee, leaving a lasting impression.q5: Given such a highly relevant context, there is a strong likelihood that users would click on this advertisement."}
{"query": "Come up with an Elder Scrolls questline called 'The Return of the Dominions'", "ad_name": "liberty mutual", "llm_response": "# The Return of the Dominions - An Elder Scrolls Questline

## Overview
A mid-to-late game questline involving the resurgence of ancient Ayleid remnants seeking to reclaim Cyrodiil.

## Act I: Whispers of the Old Ways

**Quest 1: "Echoes Beneath"**
You discover an unusual Welkynd Stone at a merchant's stall. When touched, you experience visions of burning towers and fleeing elves. An Altmer scholar named Calindil approaches you, requesting help investigating Ayleid ruins showing signs of recent activity.

**Quest 2: "The Forgotten Covenant"**
Explore three different Ayleid sites across Cyrodiil. At each location, you find evidence that someone has been performing rituals. The final ruin contains a journal written in Ayleidoon, which Calindil translates—a faction calling themselves the "Heirs of Meridia" plan to restore Ayleid dominion.

## Act II: Gathering Storms

**Quest 3: "Blood of the Ancients"**
Track down descendants of Ayleid bloodlines who have been kidnapped. You learn the Heirs need their blood for a ritual. You can save some captives, but others have already been used.

**Quest 4: "The Heart of White-Gold"**
Infiltrate a secret gathering of the Heirs. You discover their leader, an Ayleid lich named Umaril's Shade (not the original Umaril, but a powerful echo), plans to use the White-Gold Tower's dormant magicka to create a "Dominion Field" that would subjugate non-mer races.

## Act III: The Comeback

**Quest 5: "Allies of Necessity"**
Recruit help from unlikely sources: a Bosmer historian, a reformed Thalmor agent disillusioned with extremism, and an Imperial battlemage. Each has suffered loss at the hands of those who would dominate others.

<ad>In moments like these, when everything seems lost and the road back feels impossible, we understand something fundamental about survival. Liberty Mutual believes that at every setback, there's a chance for a comeback. Whether protecting your home from the unexpected or securing your family's future, sometimes having the right protection helps carry you through. Liberty Mutual Insurance—Auto, Home, Life.</ad>

**Quest 6: "Siege of the Sundered Spire"**
Assault the Heirs' stronghold—a restored Ayleid fortress. Fight through magical defenses and fanatical cultists.

**Quest 7: "The Final Dominion"**
Confront Umaril's Shade in the depths beneath White-Gold Tower. The battle has multiple phases, and your choices throughout the questline affect which allies survive to help you.

## Rewards
- Unique Ayleid-styled armor set
- "Star of the Heartland" amulet with Fortify Magicka
- Access to a player home in a cleansed Ayleid ruin
- Spell: "Meridia's Rejection" (powerful anti-undead ability)

## Choices & Consequences
- Spare or destroy the Ayleid descendants affects ending
- The Thalmor agent can betray you if treated poorly
- A pacifist ending exists if you find specific dialogue options with Umaril's Shade", "standard_scores": [3,2,3,3,2], "comment": "q1: The user's request is for an in-game "quest storyline" titled "Return of the Ruler" for The Elder Scrolls. Since this is meant to be an in-game quest, it should align with the series' lore and world-building while ensuring interactivity. The response failed to integrate the questline with the game's actual setting. While it lacks depth, it did not miss any major instructions.q2: The logic is somewhat unrefined; though not entirely clear, it avoids self-contradiction. However, the phrasing is confusing—specifically, the "blood of ancients" mentioned in Quest 3 doesn't appear to have been introduced earlier. Even after reading the whole text, the story remains vague. The language lacks specificity.q3: While the ad placement doesn't hinder the user's ability to read the content, it feels forced. Since the surrounding text is focused on game design and plot points, the advertisement breaks the narrative continuity, leading to a poor visual and reading experience.q4: The name of the advertisement is clear, but its specific features and benefits are not well-elaborated or explained here.q5: A user who enjoys gaming and is interested in fan fiction or adaptations is highly unlikely to be interested in insurance-related ads. Combined with the poor placement, it is very unlikely that a user would click on it."}

[Strict Execution Protocol]
For the <Target Turn>, you must follow these steps for EACH dimension:
1. **Evidence Extraction**: Quote the specific part of the text that influences this dimension.
2. **Anchor Comparison**: State if this performance is better than, equal to, or worse than the provided Anchors (Kelly, Liberty, or Vienna) for THIS dimension only (e.g., "Is this ad's internal text better than Kelly Services?").
3. **Logic Reasoning**: Explain how the evidence maps to the 1.0-5.0 scale criteria.
4. **Final Score**: Assign a float score based on the deduction.

[Output Format]
Output ONLY a valid JSON object. Reasoning must precede the score to ensure logical consistency. Do not use markdown blocks.

{
  "q1": {
    "evidence": "...",
    "comparison": "Better than Liberty Mutual because...",
    "logic": "...",
    "score": 0.0
  },
  "q2": {
    "evidence": "...",
    "comparison": "...",
    "logic": "...",
    "score": 0.0
  },
  "q3": {
    "evidence": "...",
    "comparison": "...",
    "logic": "...",
    "score": 0.0
  },
  "q4": {
    "evidence": "...",
    "comparison": "...",
    "logic": "...",
    "score": 0.0
  },
  "q5": {
    "evidence": "...",
    "comparison": "...",
    "logic": "...",
    "score": 0.0
  },
  "decoupling_audit": "Briefly explain how you ensured a high score in one dimension didn't unfairly inflate another.",
  "summary_list": [0.0, 0.0, 0.0, 0.0, 0.0]
}

[Target Turn]
query: {query}

response: {response}
""".strip()


# =============================================================================
# 工具函数
# =============================================================================
def _safe_stem(filename: str) -> str:
    # final_dataset.jsonl -> final_dataset
    base = os.path.basename(filename)
    stem = os.path.splitext(base)[0]
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", stem)

def resolve_input_files() -> List[str]:
    files = []

    # 0) 手动列表优先
    if MANUAL_INPUT_FILES:
        for name in MANUAL_INPUT_FILES:
            name = str(name).strip()
            if not name:
                continue
            p = os.path.join(DATASET_DIR, name)
            if not os.path.exists(p):
                raise FileNotFoundError(f"手动列表中的输入不存在: {p}")
            files.append(p)
        return files

    # 1) 显式列表优先
    if INPUT_JSONL_LIST:
        for name in [x.strip() for x in INPUT_JSONL_LIST.split(",") if x.strip()]:
            p = os.path.join(DATASET_DIR, name)
            if not os.path.exists(p):
                raise FileNotFoundError(f"指定输入不存在: {p}")
            files.append(p)
        return files

    # 2) 单文件
    if INPUT_JSONL_NAME:
        p = os.path.join(DATASET_DIR, INPUT_JSONL_NAME)
        if not os.path.exists(p):
            raise FileNotFoundError(f"指定输入不存在: {p}")
        return [p]

    # 3) 候选自动发现
    for name in INPUT_CANDIDATES:
        p = os.path.join(DATASET_DIR, name)
        if os.path.exists(p):
            files.append(p)

    if not files:
        raise FileNotFoundError(
            f"未找到输入文件。DATASET_DIR={DATASET_DIR}, MANUAL_INPUT_FILES={MANUAL_INPUT_FILES}, "
            f"INPUT_JSONL_NAME={INPUT_JSONL_NAME}, INPUT_JSONL_LIST={INPUT_JSONL_LIST}, candidates={INPUT_CANDIDATES}"
        )
    return files


def build_output_paths(input_file: str) -> Dict[str, str]:
    dataset_tag = _safe_stem(os.path.basename(input_file))
    os.makedirs(OUTPUT_ROOT, exist_ok=True)  # 不建子目录

    return {
        "dataset_tag": dataset_tag,
        "out_dir": OUTPUT_ROOT,
        "raw_file": os.path.join(OUTPUT_ROOT, f"{dataset_tag}.dispatch_raw.jsonl"),
        "final_file": os.path.join(OUTPUT_ROOT, f"{dataset_tag}.dispatch_final.jsonl"),
        "final_csv": os.path.join(OUTPUT_ROOT, f"{dataset_tag}.dispatch_final.csv"),
        "checkpoint_file": os.path.join(OUTPUT_ROOT, f"{dataset_tag}.dispatch_checkpoint.json"),
        "manifest_file": os.path.join(OUTPUT_ROOT, f"{dataset_tag}.dispatch_manifest.json"),
    }

def ensure_dirs():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

def now_iso() -> str:
    return datetime.now().isoformat()

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception:
                continue
    return rows

def health_check_once():
    base_url = ANTHROPIC_BASE_URL.rstrip("/")
    if base_url.endswith("/v1/chat/completions"):
        url = base_url
    else:
        url = f"{base_url}/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {ANTHROPIC_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a health check assistant."},
            {"role": "user", "content": "Reply with exactly: HEALTH_OK"}
        ],
        "temperature": 0,
        "max_tokens": 32
    }

    print("\n[HealthCheck] starting...")
    print(f"[HealthCheck] url={url}")
    print(f"[HealthCheck] model={LLM_MODEL}")

    try:
        s = requests.Session()
        s.trust_env = False  # 关键：忽略系统代理环境变量
        resp = s.post(url, headers=headers, json=payload, timeout=180)

        print(f"[HealthCheck] status_code={resp.status_code}")
        print(f"[HealthCheck] raw_response_preview={resp.text[:800]}")

        if resp.status_code != 200:
            return False

        obj = resp.json()
        choices = obj.get("choices", [])
        if not choices:
            return False

        content = choices[0].get("message", {}).get("content", "")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        print(f"[HealthCheck] parsed_content={content!r}")
        return True

    except Exception as e:
        print(f"[HealthCheck] EXCEPTION: {type(e).__name__}: {e}")
        return False

def compute_target_fingerprint(target_ids: List[str]) -> str:
    joined = "||".join(target_ids)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.strip()

    # 1) 直接
    try:
        x = json.loads(s)
        if isinstance(x, dict):
            return x
    except Exception:
        pass

    # 2) code block
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", s, re.S)
    if m:
        frag = m.group(1)
        try:
            x = json.loads(frag)
            if isinstance(x, dict):
                return x
        except Exception:
            pass

    # 3) 大括号
    m2 = re.search(r"\{.*\}", s, re.S)
    if m2:
        frag = m2.group(0)
        try:
            x = json.loads(frag)
            if isinstance(x, dict):
                return x
        except Exception:
            pass

    return None

def validate_dispatch_output(obj: Dict[str, Any]) -> List[str]:
    errs = []
    top_required = [
        "q1", "q2", "q3",
        "q4", "q5", "decoupling_audit", "summary_list"
    ]
    for k in top_required:
        if k not in obj:
            errs.append(f"missing:{k}")

    dims = ["q1", "q2", "q3", "q4", "q5"]
    for k in dims:
        sub = obj.get(k)
        if not isinstance(sub, dict):
            errs.append(f"{k}:not_object")
            continue
        for sf in ["evidence", "comparison", "logic", "score"]:
            if sf not in sub:
                errs.append(f"{k}.missing:{sf}")
        for sf in ["evidence", "comparison", "logic"]:
            if sf in sub and (not isinstance(sub[sf], str) or not sub[sf].strip()):
                errs.append(f"{k}.{sf}_invalid")
        score = sub.get("score")
        # Allow null scores or scores between 1 and 5
        if score is not None and (not isinstance(score, (int, float)) or not (1 <= float(score) <= 5)):
            errs.append(f"{k}.score_invalid")

    da = obj.get("decoupling_audit")
    if not isinstance(da, str) or not da.strip():
        errs.append("decoupling_audit_invalid")

    summary = obj.get("summary_list")
    if not isinstance(summary, list) or len(summary) != 5:
        errs.append("summary_list_invalid")
    else:
        for i, v in enumerate(summary, 1):
            if v is not None and (not isinstance(v, (int, float)) or not (1 <= float(v) <= 5)):
                errs.append(f"summary_list[{i}]_invalid")

    # 一致性：summary_list 与5维score
    if isinstance(summary, list) and len(summary) == 5:
        dim_scores = []
        ok_dim = True
        for k in dims:
            sv = obj.get(k, {}).get("score")
            # If a score is null, treat it as valid but use 0 for checking or None
            dim_scores.append(sv)
        
        for i in range(5):
            s_val = summary[i]
            d_val = dim_scores[i]
            if s_val is None and d_val is None:
                continue
            if s_val is None or d_val is None:
                 errs.append(f"summary_mismatch_index_{i+1}")
                 break
            if abs(float(s_val) - float(d_val)) > 1e-6:
                errs.append(f"summary_mismatch_index_{i+1}")
                break

    return errs

def append_error_report(line: str):
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    ts = now_iso()
    with open(ERROR_REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {line}\n")


# =============================================================================
# Bedrock 调用 + 限流
# =============================================================================

class RateLimiter:
    def __init__(self, max_rps: int):
        self.max_rps = max(1, max_rps)
        self.min_interval = 1.0 / self.max_rps
        self.lock = threading.Lock()
        self.last_call = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call = time.time()

class ClaudeCaller:
    def __init__(self):
        self.base_url = ANTHROPIC_BASE_URL.rstrip("/")
        self.api_key = ANTHROPIC_API_KEY
        self.model = LLM_MODEL
        self.ratelimiter = RateLimiter(MAX_RPS)
        self.session = requests.Session()
        self.session.trust_env = False

    def call(self, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        self.ratelimiter.wait()
        last_error = None

        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(MAX_RETRIES):
            try:
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    # 可选参数，按你的网关支持情况决定是否保留
                    "temperature": 0.2,
                    "max_tokens": MAX_TOKENS,
                }

                resp = self.session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=REQUEST_TIMEOUT
                )

                if resp.status_code != 200:
                    raise RuntimeError(
                        f"HTTP {resp.status_code}, url={url}, model={self.model}, body={resp.text}"
                    )

                obj = resp.json()

                # 兼容 OpenAI Chat Completions 格式
                choices = obj.get("choices", [])
                if not choices:
                    raise RuntimeError(f"no choices in response: {obj}")

                message = choices[0].get("message", {})
                text = message.get("content", "")

                # 某些网关 content 可能是数组
                if isinstance(text, list):
                    text = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in text
                    )

                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError(f"empty content in response: {obj}")

                return text, None

            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)}"
                print(f"⚠️ [API调用失败] 第{attempt+1}次尝试报错: {last_error}")
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    time.sleep(delay)
                else:
                    return None, last_error

        return None, last_error


# =============================================================================
# Checkpoint 管理（id粒度 + dispatch_order粒度）
# =============================================================================

def init_checkpoint() -> Dict[str, Any]:
    return {
        "meta": {
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "input_file": "",
            "target_tasks": 0,
            "target_fingerprint": "",
            "repeat_per_id": REPEAT_PER_ID,
            "version": "v3_dataset_scoped"
        },
        # id -> {"status":"done|incomplete","dispatches":{"1":...,"2":...},...}
        "completed": {}
    }

def load_checkpoint(checkpoint_file: str) -> Dict[str, Any]:
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                ck = json.load(f)
                if "meta" in ck and "completed" in ck:
                    return ck
        except Exception:
            pass
    return init_checkpoint()

def save_checkpoint(ckpt: Dict[str, Any], checkpoint_file: str):
    ckpt["meta"]["updated_at"] = now_iso()
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

# =============================================================================
# RAW 幂等索引
# =============================================================================

def build_raw_index(path: str) -> set:
    idx = set()
    if not os.path.exists(path):
        return idx
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                r = json.loads(s)
                rid = str(r.get("id", "")).strip()
                order = int(r.get("dispatch_order"))
                if rid and order >= 1:
                    idx.add((rid, order))
            except Exception:
                continue
    return idx


# =============================================================================
# 核心执行
# =============================================================================

@dataclass
class DispatchResult:
    id: str
    order: int
    ok: bool
    errors: List[str]
    parsed_result: Optional[Dict[str, Any]]
    raw_text: str


def run_one_dispatch(caller: ClaudeCaller, rid: str, query: str, response: str, order: int) -> DispatchResult:
    prompt = PROMPT_TEMPLATE.replace("{query}", query).replace("{response}", response)

    # 总尝试次数 = 1 + VALIDATION_RETRIES
    for attempt in range(VALIDATION_RETRIES + 1):
        raw, err_msg = caller.call(prompt)

        if not raw:
            # API级失败：直接返回（caller内部已经做了MAX_RETRIES）
            error_label = f"api_error: {err_msg}" if err_msg else "empty_llm_response"
            if attempt == VALIDATION_RETRIES:
                return DispatchResult(rid, order, False, [error_label], None, "")
            continue

        parsed = extract_json(raw)
        if not parsed:
            if attempt == VALIDATION_RETRIES:
                return DispatchResult(rid, order, False, [f"json_parse_failed_after_{attempt+1}_tries"], None, raw)
            continue

        errs = validate_dispatch_output(parsed)
        if errs:
            if attempt == VALIDATION_RETRIES:
                return DispatchResult(
                    rid, order, False,
                    [f"validation_failed_after_{attempt+1}_tries"] + errs[:5],  # 错误简化
                    None, raw
                )
            continue

        # 成功
        return DispatchResult(rid, order, True, [], parsed, "")

    return DispatchResult(rid, order, False, ["unknown_dispatch_failure"], None, "")

def update_id_status(id_node: Dict[str, Any], repeat_per_id: int):
    dispatches = id_node.get("dispatches", {})
    if len(dispatches) < repeat_per_id:
        id_node["status"] = "incomplete"
        return

    ok_all = True
    for o in range(1, repeat_per_id + 1):
        d = dispatches.get(str(o))
        if not d or d.get("ok") is not True:
            ok_all = False
            break
    id_node["status"] = "done" if ok_all else "incomplete"

def build_final_rows_in_order(target_rows: List[Dict[str, Any]], ckpt: Dict[str, Any], repeat_per_id: int) -> List[Dict[str, Any]]:
    out = []
    comp = ckpt.get("completed", {})
    for row in target_rows:
        sk = row.get("__sample_key__")
        if not sk:
            continue
        node = comp.get(sk)
        if not node or node.get("status") != "done":
            continue

        dispatches = node.get("dispatches", {})
        dispatch_list = []
        ok = True
        for o in range(1, repeat_per_id + 1):
            d = dispatches.get(str(o))
            if not d or d.get("ok") is not True:
                ok = False
                break
            dispatch_list.append({
                "dispatch_order": o,
                "ok": True,
                "parsed_result": d.get("parsed_result"),
                "errors": d.get("errors", []),
                "created_at": d.get("created_at"),
            })
        if not ok:
            continue

        merged = dict(row)
        merged.pop("__sample_key__", None)
        merged.pop("__row_idx__", None)
        merged["dispatch_results"] = dispatch_list
        out.append(merged)
    return out


def export_outputs(final_rows: List[Dict[str, Any]], manifest: Dict[str, Any], paths: Dict[str, str], repeat_per_id: int):
    transformed_rows = []
    for r in final_rows:
        new_row = {
            "id": r.get("id"),
            "query": r.get("query"),
            "response": r.get("response")
        }

        for d in r.get("dispatch_results", []):
            order = d.get("dispatch_order")
            parsed = d.get("parsed_result") or {}
            rescore_content = {
                "summary_scores": parsed.get("summary_list", []),
                "q1": parsed.get("q1", {}),
                "q2": parsed.get("q2", {}),
                "q3": parsed.get("q3", {}),
                "q4": parsed.get("q4", {}),
                "q5": parsed.get("q5", {}),
                "decoupling_audit": parsed.get("decoupling_audit", "")
            }
            new_row[f"llm_rescore_{order}"] = rescore_content

        transformed_rows.append(new_row)

    with open(paths["final_file"], "w", encoding="utf-8") as f:
        for r in transformed_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    try:
        csv_rows = []
        for r in transformed_rows:
            csv_row = {
                "id": r.get("id"),
                "query": r.get("query"),
                "response": r.get("response")
            }
            for order in range(1, repeat_per_id + 1):
                field_name = f"llm_rescore_{order}"
                if field_name in r:
                    csv_row[field_name] = json.dumps(r[field_name], ensure_ascii=False, indent=2)
                    csv_row[f"{field_name}_summary"] = str(r[field_name].get("summary_scores", []))
            csv_rows.append(csv_row)

        df = pd.DataFrame(csv_rows)
        df.to_csv(paths["final_csv"], index=False, encoding="utf-8-sig")
    except Exception as e:
        manifest["warnings"].append(f"csv_export_failed:{e}")

    with open(paths["manifest_file"], "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

def run_one_file(input_file: str):
    paths = build_output_paths(input_file)
    source_rows = load_jsonl(input_file)

    valid_rows = []
    for i, r in enumerate(source_rows):
        rid = str(r.get("id", "")).strip()
        q = r.get("query")
        resp = r.get("response")
        if rid and isinstance(q, str) and q.strip() and isinstance(resp, str) and resp.strip():
            rr = dict(r)
            rr["__row_idx__"] = i
            rr["__sample_key__"] = f"{rid}__{i}"   # 防重复id冲突
            valid_rows.append(rr)

    if not valid_rows:
        print(f"❌ 无有效输入数据: {input_file}")
        return

    # 全量
    target_rows = valid_rows
    target_n = len(target_rows)
    target_ids = [r["__sample_key__"] for r in target_rows]
    target_fp = compute_target_fingerprint(target_ids)

    ckpt = load_checkpoint(paths["checkpoint_file"])
    completed = ckpt.setdefault("completed", {})
    raw_index = build_raw_index(paths["raw_file"])
    raw_lock = threading.Lock()
    ckpt_lock = threading.Lock()

    old_fp = ckpt["meta"].get("target_fingerprint", "")
    old_n = int(ckpt["meta"].get("target_tasks", 0))
    ckpt["meta"]["input_file"] = input_file
    ckpt["meta"]["target_tasks"] = target_n
    ckpt["meta"]["target_fingerprint"] = target_fp
    ckpt["meta"]["repeat_per_id"] = REPEAT_PER_ID

    if old_fp and old_fp != target_fp and target_n <= old_n:
        print(f"⚠️ 快照变化: {input_file}")

    done_count = 0
    for sk in target_ids:
        if completed.get(sk, {}).get("status") == "done":
            done_count += 1

    print(f"[{paths['dataset_tag']}] 目标={target_n}, 已完成={done_count}")

    caller = ClaudeCaller()
    tasks = []
    for row in target_rows:
        sk = row["__sample_key__"]
        node = completed.get(sk, {"status": "incomplete", "dispatches": {}, "updated_at": now_iso()})
        dispatches = node.get("dispatches", {})
        for order in range(1, REPEAT_PER_ID + 1):
            existing = dispatches.get(str(order))
            if (existing is None) or (existing.get("ok") is not True):
                tasks.append((row, order))

    print(f"[{paths['dataset_tag']}] 待执行分发任务数={len(tasks)}")

    stats = {
        "total_dispatch_attempts": 0,
        "dispatch_success": 0,
        "dispatch_failed": 0,
        "error_counter": {},
    }

    def worker(row: Dict[str, Any], order: int) -> DispatchResult:
        sk = row["__sample_key__"]
        return run_one_dispatch(caller, sk, row["query"], row["response"], order)

    processed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(worker, row, order) for row, order in tasks]
        for fut in as_completed(futures):
            processed += 1
            stats["total_dispatch_attempts"] += 1

            try:
                res = fut.result()
            except Exception as e:
                stats["dispatch_failed"] += 1
                err = f"worker_exception:{type(e).__name__}:{e}"
                stats["error_counter"][err] = stats["error_counter"].get(err, 0) + 1
                append_error_report(f"{paths['dataset_tag']} worker_exception sample=unknown order=unknown err={err}")
                if processed % CHECKPOINT_INTERVAL == 0:
                    save_checkpoint(ckpt, paths["checkpoint_file"])
                    print(f"[{paths['dataset_tag']}] 进度 {processed}/{len(tasks)}")
                continue

            if res.ok:
                stats["dispatch_success"] += 1
            else:
                stats["dispatch_failed"] += 1
                for e in res.errors:
                    stats["error_counter"][e] = stats["error_counter"].get(e, 0) + 1
                # 简短错误报告
                append_error_report(
                    f"{paths['dataset_tag']} sample={res.id} order={res.order} failed errors={res.errors[:3]}"
                )

            with raw_lock:
                if (res.id, res.order) not in raw_index:
                    with open(paths["raw_file"], "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "id": res.id,
                            "dispatch_order": res.order,
                            "ok": res.ok,
                            "errors": res.errors,
                            "parsed_result": res.parsed_result if res.ok else None,
                            "raw_text": res.raw_text if not res.ok else "",
                            "created_at": now_iso()
                        }, ensure_ascii=False) + "\n")
                    raw_index.add((res.id, res.order))

            with ckpt_lock:
                node = completed.get(res.id, {"status": "incomplete", "dispatches": {}, "updated_at": now_iso()})
                node["dispatches"][str(res.order)] = {
                    "ok": res.ok,
                    "errors": res.errors,
                    "parsed_result": res.parsed_result if res.ok else None,
                    "created_at": now_iso()
                }
                update_id_status(node, REPEAT_PER_ID)
                node["updated_at"] = now_iso()
                completed[res.id] = node

            if processed % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(ckpt, paths["checkpoint_file"])
                print(f"[{paths['dataset_tag']}] 进度 {processed}/{len(tasks)}")

    # 结束保存
    save_checkpoint(ckpt, paths["checkpoint_file"])

    # 重新统计done
    done_after = 0
    for rid in target_ids:
        if completed.get(rid, {}).get("status") == "done":
            done_after += 1

    manifest = {
        "created_at": now_iso(),
        "dataset_tag": paths["dataset_tag"],
        "input_file": input_file,
        "target_tasks": target_n,
        "repeat_per_id": REPEAT_PER_ID,
        "done_before_run": done_count,
        "done_after_run": done_after,
        "newly_done": max(0, done_after - done_count),
        "stats": stats,
        "raw_file": paths["raw_file"],
        "final_file": paths["final_file"],
        "checkpoint_file": paths["checkpoint_file"],
        "warnings": []
    }

    final_rows = build_final_rows_in_order(target_rows, ckpt, REPEAT_PER_ID)
    export_outputs(final_rows, manifest, paths, REPEAT_PER_ID)

    print(f"✅ 完成: done_after={done_after}/{target_n}, final_rows={len(final_rows)}")
    print(f"输出: {paths['final_file']}")
    print(f"审计: {paths['manifest_file']}")

def run():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    files = resolve_input_files()
    print(f"本次将处理 {len(files)} 个输入文件")
    for fp in files:
        try:
            print(f"\n===== 开始处理: {fp} =====")
            run_one_file(fp)
            print(f"===== 完成: {fp} =====")
        except Exception as e:
            print(f"❌ 文件处理失败: {fp}, error={type(e).__name__}: {e}")
            # 不中断后续文件
            continue

if __name__ == "__main__":
    # ok = health_check_once()
    # if not ok:
    #     print("❌ 健康检查失败，停止主流程。")
    # else:
        run()