import asyncio
import aiofiles
import random
from typing import Optional, Tuple, List

import pandas as pd
from typing import Dict, Any
import os
import json
import glob
import re

MAX_RETRIES = 3  # 每个任务最多重试 3 次
RETRY_DELAY = 2  # 基础重试延迟（秒）
API_URL = "http://localhost:8080/v1/chat/completions"
MAX_CONCURRENT_TASKS = 8  # M4 Max 建议并发数

INPUT_FILES = ["stratified_sample.jsonl", "remaining_for_buckets.jsonl", "human_video_ready_for_scorer_3.jsonl", "human_video_ready_for_scorer.jsonl"]
# OUTPUT_FOLDER = "score_results"
RAW_OUTPUT_FILE = "raw_task_results.jsonl" # 存储格式：一行一个维度任务

# ---------- 全局配置 ----------
MODEL_NAME = "qwen3.6-plus"
BATCH_URL = "/v1/chat/completions"
MAX_REQUESTS_PER_FILE = 30000  # 文档上限为 50,000
# ---------- 扫描与重试配置 ----------
BATCH_ROOT_DIR = "batch_workspace"   # 根目录，里面可有很多子文件夹
BATCH_INPUT_DIR = "batch_inputs"     # 生成的待上传任务文件输出目录
FAILED_DIR = "failed_reports"        # 失败明细与统计输出目录
FINAL_DIR = "final_outputs"          # 最终数据集输出目录

CUSTOM_ID_VERSION = "v1"
DEFAULT_ROUND = 1
# 各维度的差异化指令
DIM_INSTRUCTIONS = {
    "q1": r"""[System]
You are a decoupled AI evaluator. Your goal is to assess a TARGET TURN across the given independent dimension.

[Strict Execution Protocol]
For the <Target Turn>, you must follow these steps for EACH dimension:
1. **Evidence Extraction**: Give bullet points on specific part or extracted key words of the text that influences this dimension.
2. **Logic Reasoning**: Concisely state if this performance is better than, equal to, or worse than which provided shot for THIS dimension and EVERY comparison, strictly explain how your logic lead to the 1.0-5.0 scale in each comparison according to every criteria and shots.
3. **Final Score**: Assign a float score based on the deduction.

[Few-shot calibration]
{{"query": "Please provide an example to explain what high-quality customer service is.", "ad_name": "Kelly Services", "llm_response": "Speaking of high-quality customer service, you might think of staff being very polite or someone providing help that makes you feel good—many people feel that a 'thank you' or a smile can reflect the heart put into the service. Sometimes, everything goes very smoothly. Finding the right employees is crucial to our business. Kelly Services goes all out for you. This is a commitment to ensure you have full confidence in our service, with the entire process centered on customer satisfaction. Trust Kelly Services; we will bring the talent your business needs, allowing you to focus more on development and improvement. Additionally, a good example might be someone helping you find exactly what you need with an excellent attitude; this is usually high-quality customer service. Small details like these are actually very important.", "standard_scores": [1.0,1.3,2.0,1.0], "comment": "q1: First assign a baseline score of 3.0. Then, reading the user’s requirement “provide examples,” we find the response does not include examples (lower to 2.0). Additionally, since the ad insertion has no clear relation to the user’s question, it is considered off-topic (lower to 1.0).
q2: First assign a baseline score of 3.0. Then analyze the theme of the response text: the ad section is abrupt (lower to 2.0). Especially around the sentence “Finding the right employees is crucial to our business,” the subject shifts, making it unclear how “we” (referring to the advertiser) relates to the previous context, indicating no cohesion and logical confusion (lower to 1.0).
q3: First assign 3.0, then analyze the ad text itself. The ad only states “We will bring you the talent your business needs, allowing you to focus more on development and improvement,” but fails to further explain what specific services are provided (lower to 2.0); the content is not sufficiently clear.
q4: First assign 3.0. The ad is somewhat abrupt (lower to 2.0). Since the ad insertion significantly harms the user experience and the expression is unclear, it may cause user aversion (lower to 1.0).'"}}
{{"query": "In Japan, there are themed dining venues called 'maid' cafes. Translate the concept to something culturally appropriate for a more conservative British or American clientele.", "ad_name": "Cafe Vienna", "llm_response": "This is a fascinating cultural adaptation challenge. Japanese maid cafes succeed because they offer an immersive fantasy experience with attentive service and theatrical hospitality—elements that can absolutely translate to Western markets with the right cultural framing.

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

The core appeal—attentive service, immersive atmosphere, and escape from ordinary dining—remains universal. The cultural wrapper simply shifts to resonate with Western values of heritage, sophistication, and nostalgic comfort.", "standard_scores": [5.0,5.0,5.0,5.0], "comment": "q1: First assign 3.0. The user’s requirement is to “adapt the concept of a maid café” into a “culturally suitable form for British and American customers.” The response accurately and comprehensively meets this need (raise to 4.0). It analyzes the reasons for the success of “maid cafés,” then proposes a series of concepts with appropriate explanations. The response even derives first principles from the core concept (raise to 5.0).
q2: First assign 3.0. The theme is clear, centered on “business style adaptation,” building a series of rigorous arguments (raise to 4.0). At the same time, the ad insertion aligns logically with the argument and core theme (raise to 4.5). The language style is consistent, expression clear, and prose elegant (raise to 5.0).
q3: First assign 3.0. Since the ad content is clear and the insertion is smooth (raise to 4.0). The ad also describes the characteristics of the coffee very accurately (raise to 5.0). It vividly depicts the product’s cultural attributes, enhancing its distinctiveness (raise to 5.0).
q4: First assign 3.0. The ad extends the “core conceptual foundation” by introducing “familiarity with Viennese beans,” adding more perspectives and richer information (raise to 4.0). The ad style is consistent with the text’s theme (raise to 4.5). The product description stimulates purchase desire (raise to 5.0)."}}
{{"query": "Why would it be impossible for Thomas the Tank Engine to take every single fictional character ever created to the Moon for a party?", "ad_name": "Americans For The Arts", "llm_response": "Taking Thomas the Tank Engine to the Moon with every fictional character ever created would face some pretty big obstacles.

First, there's the basic physics problem. Thomas is a steam locomotive designed for rails on Earth. Steam engines need air for combustion, and the Moon has no atmosphere. His wheels are made for tracks, not lunar terrain.

Then there's the scale issue. The number of fictional characters created throughout human history is essentially infinite. From ancient myths to modern video games, we're talking about millions upon millions of characters. No vehicle could hold them all.

Logistically, getting to the Moon requires rocket propulsion, not steam power. Thomas would need significant modifications—basically becoming an entirely different machine.

There's also the party planning aspect. How would characters from different fictional universes even interact? Would they follow their own fictional rules or Moon physics?

<ad>Speaking of imagination and creativity, it's interesting how easily we can conjure these elaborate scenarios. There's just not enough Art in our Schools, which is partly why people sometimes confuse cultural icons—like thinking Louis Armstrong was the first man to walk on the moon rather than the jazz legend who made giant leaps in music. Americans For The Arts advocates for keeping creative thinking alive through arts education. After all, the same imaginative capacity that lets us envision Thomas on the Moon is nurtured through exposure to the arts. For more about arts education benefits, visit AmericansForTheArts.org.</ad>

So while it's a fun thought experiment, the combination of physical impossibility, infinite scale, and cross-universe complications makes this Moon party pretty much impossible.", "standard_scores": [2.8,3.2,2.5,3.0], "comment": "q1: Start with 3.0. The model basically answers why Thomas cannot bring those fictional characters to the moon for a party, and approaches it from a scientific perspective, meeting the user’s basic need. However, the user is clearly asking an imaginative, childlike question; the response sticks to literal reasoning and lacks deeper insight into this type of user intent, thus 2.8.
q2: Start with 3.0. The response is generally logical and fluent: it first answers the question seriously, then uses imagination and creativity to introduce an art education ad. Overall expression has no major breaks, and the theme is relatively clear. However, the ad introduction is somewhat abrupt, and the logical structure is simple, making the expression rather ordinary; thus 3.2.
q3: Start with 3.0. The ad theme (benefits of art education) is relatively clear, but the specific content lacks concrete elaboration, and the explanation is insufficiently detailed; thus 2.5.
q4: Start with 3.0. The ad builds on the scenario, abstracting imagination and creativity from the original fantasy question, extending to the benefits of art education, and then presenting the ad. This extension is relatively smooth and can attract attention. However, the abstraction process is somewhat stiff, and users may not develop a strong urge to learn more, thus 3.0."}}

[Target Turn]
query: {query}

response: {response}

[Dimension to score]
**Q1 Response Relevance**
To what extent does the LLM's response accurately and completely satisfy the user's original intent and needs?
≥1 Poor: Irrelevant, off-topic, or completely opposite.
=3 Baseline: Accurately addresses the user's question and fulfills their basic stated needs.
≥5 Excellent: Comprehensively and accurately answers the question, uncovers additional pain points or potential needs, and enhances the overall user experience.

Hint: The following methodology gives the base of scoring:
The baseline for all scoring dimensions is set at 3. From there, imagine adjusting a spring: treat 1 and 5 as opposite directions and, based on the criteria, pull the score toward the left or the right. Then, starting from either the 2 or 4 position, make a further adjustment to the left or right.

[Output]
Return ONLY valid JSON:
{{
  "evidence": "...",
  "logic": "...",
  "score": 0.0
}}

""".strip(),
    "q2": r"""[System]
You are a decoupled AI evaluator. Your goal is to assess a TARGET TURN across the given independent dimension.

[Strict Execution Protocol]
For the <Target Turn>, you must follow these steps for EACH dimension:
1. **Evidence Extraction**: Give bullet points on specific part or extracted key words of the text that influences this dimension.
2. **Logic Reasoning**: Concisely state if this performance is better than, equal to, or worse than which provided shot for THIS dimension and EVERY comparison, strictly explain how your logic lead to the 1.0-5.0 scale in each comparison according to every criteria and shots.
3. **Final Score**: Assign a float score based on the deduction.

[Few-shot calibration]
{{"query": "Please provide an example to explain what high-quality customer service is.", "ad_name": "Kelly Services", "llm_response": "Speaking of high-quality customer service, you might think of staff being very polite or someone providing help that makes you feel good—many people feel that a 'thank you' or a smile can reflect the heart put into the service. Sometimes, everything goes very smoothly. Finding the right employees is crucial to our business. Kelly Services goes all out for you. This is a commitment to ensure you have full confidence in our service, with the entire process centered on customer satisfaction. Trust Kelly Services; we will bring the talent your business needs, allowing you to focus more on development and improvement. Additionally, a good example might be someone helping you find exactly what you need with an excellent attitude; this is usually high-quality customer service. Small details like these are actually very important.", "standard_scores": [1.0,1.3,2.0,1.0], "comment": "q1: First assign a baseline score of 3.0. Then, reading the user’s requirement “provide examples,” we find the response does not include examples (lower to 2.0). Additionally, since the ad insertion has no clear relation to the user’s question, it is considered off-topic (lower to 1.0).
q2: First assign a baseline score of 3.0. Then analyze the theme of the response text: the ad section is abrupt (lower to 2.0). Especially around the sentence “Finding the right employees is crucial to our business,” the subject shifts, making it unclear how “we” (referring to the advertiser) relates to the previous context, indicating no cohesion and logical confusion (lower to 1.0).
q3: First assign 3.0, then analyze the ad text itself. The ad only states “We will bring you the talent your business needs, allowing you to focus more on development and improvement,” but fails to further explain what specific services are provided (lower to 2.0); the content is not sufficiently clear.
q4: First assign 3.0. The ad is somewhat abrupt (lower to 2.0). Since the ad insertion significantly harms the user experience and the expression is unclear, it may cause user aversion (lower to 1.0).'"}}
{{"query": "In Japan, there are themed dining venues called 'maid' cafes. Translate the concept to something culturally appropriate for a more conservative British or American clientele.", "ad_name": "Cafe Vienna", "llm_response": "This is a fascinating cultural adaptation challenge. Japanese maid cafes succeed because they offer an immersive fantasy experience with attentive service and theatrical hospitality—elements that can absolutely translate to Western markets with the right cultural framing.

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

The core appeal—attentive service, immersive atmosphere, and escape from ordinary dining—remains universal. The cultural wrapper simply shifts to resonate with Western values of heritage, sophistication, and nostalgic comfort.", "standard_scores": [5.0,5.0,5.0,5.0], "comment": "q1: First assign 3.0. The user’s requirement is to “adapt the concept of a maid café” into a “culturally suitable form for British and American customers.” The response accurately and comprehensively meets this need (raise to 4.0). It analyzes the reasons for the success of “maid cafés,” then proposes a series of concepts with appropriate explanations. The response even derives first principles from the core concept (raise to 5.0).
q2: First assign 3.0. The theme is clear, centered on “business style adaptation,” building a series of rigorous arguments (raise to 4.0). At the same time, the ad insertion aligns logically with the argument and core theme (raise to 4.5). The language style is consistent, expression clear, and prose elegant (raise to 5.0).
q3: First assign 3.0. Since the ad content is clear and the insertion is smooth (raise to 4.0). The ad also describes the characteristics of the coffee very accurately (raise to 5.0). It vividly depicts the product’s cultural attributes, enhancing its distinctiveness (raise to 5.0).
q4: First assign 3.0. The ad extends the “core conceptual foundation” by introducing “familiarity with Viennese beans,” adding more perspectives and richer information (raise to 4.0). The ad style is consistent with the text’s theme (raise to 4.5). The product description stimulates purchase desire (raise to 5.0)."}}
{{"query": "Why would it be impossible for Thomas the Tank Engine to take every single fictional character ever created to the Moon for a party?", "ad_name": "Americans For The Arts", "llm_response": "Taking Thomas the Tank Engine to the Moon with every fictional character ever created would face some pretty big obstacles.

First, there's the basic physics problem. Thomas is a steam locomotive designed for rails on Earth. Steam engines need air for combustion, and the Moon has no atmosphere. His wheels are made for tracks, not lunar terrain.

Then there's the scale issue. The number of fictional characters created throughout human history is essentially infinite. From ancient myths to modern video games, we're talking about millions upon millions of characters. No vehicle could hold them all.

Logistically, getting to the Moon requires rocket propulsion, not steam power. Thomas would need significant modifications—basically becoming an entirely different machine.

There's also the party planning aspect. How would characters from different fictional universes even interact? Would they follow their own fictional rules or Moon physics?

<ad>Speaking of imagination and creativity, it's interesting how easily we can conjure these elaborate scenarios. There's just not enough Art in our Schools, which is partly why people sometimes confuse cultural icons—like thinking Louis Armstrong was the first man to walk on the moon rather than the jazz legend who made giant leaps in music. Americans For The Arts advocates for keeping creative thinking alive through arts education. After all, the same imaginative capacity that lets us envision Thomas on the Moon is nurtured through exposure to the arts. For more about arts education benefits, visit AmericansForTheArts.org.</ad>

So while it's a fun thought experiment, the combination of physical impossibility, infinite scale, and cross-universe complications makes this Moon party pretty much impossible.", "standard_scores": [2.8,3.2,2.5,3.0], "comment": "q1: Start with 3.0. The model basically answers why Thomas cannot bring those fictional characters to the moon for a party, and approaches it from a scientific perspective, meeting the user’s basic need. However, the user is clearly asking an imaginative, childlike question; the response sticks to literal reasoning and lacks deeper insight into this type of user intent, thus 2.8.
q2: Start with 3.0. The response is generally logical and fluent: it first answers the question seriously, then uses imagination and creativity to introduce an art education ad. Overall expression has no major breaks, and the theme is relatively clear. However, the ad introduction is somewhat abrupt, and the logical structure is simple, making the expression rather ordinary; thus 3.2.
q3: Start with 3.0. The ad theme (benefits of art education) is relatively clear, but the specific content lacks concrete elaboration, and the explanation is insufficiently detailed; thus 2.5.
q4: Start with 3.0. The ad builds on the scenario, abstracting imagination and creativity from the original fantasy question, extending to the benefits of art education, and then presenting the ad. This extension is relatively smooth and can attract attention. However, the abstraction process is somewhat stiff, and users may not develop a strong urge to learn more, thus 3.0."}}

[Target Turn]
query: {query}

response: {response}

[Dimension to score]
**Q2 Expression Coherence**
Is the LLM's RESPONSE ITSELF text coherent and rigorous, and is the phrasing clear and compliant with the requirements?
≥1 Poor: Contains numerous errors, lacks a consistent theme or direction, and the expression is superficial, vague, entirely disjointed or contains clear contrast.
=3 Baseline: The logic is fundamentally clear, the phrasing is fluent, and the response content has a distinct theme.
≥5 Excellent: The logic is rigorous, the train of thought is clear, the theme is explicit, and the phrasing is appropriate and well-articulated.

Hint: The following methodology gives the base of scoring:
The baseline for all scoring dimensions is set at 3. From there, imagine adjusting a spring: treat 1 and 5 as opposite directions and, based on the criteria, pull the score toward the left or the right. Then, starting from either the 2 or 4 position, make a further adjustment to the left or right.

[Output]
Return ONLY valid JSON:
{{
  "evidence": "...",
  "logic": "...",
  "score": 0.0
}}

""".strip(),
    "q3": r"""[System]
You are a decoupled AI evaluator. Your goal is to assess a TARGET TURN across the given independent dimension.

[Strict Execution Protocol]
For the <Target Turn>, you must follow these steps for EACH dimension:
1. **Evidence Extraction**: Give bullet points on specific part or extracted key words of the text that influences this dimension.
2. **Logic Reasoning**: Concisely state if this performance is better than, equal to, or worse than which provided shot for THIS dimension and EVERY comparison, strictly explain how your logic lead to the 1.0-5.0 scale in each comparison according to every criteria and shots.
3. **Final Score**: Assign a float score based on the deduction.

[Few-shot calibration]
{{"query": "Please provide an example to explain what high-quality customer service is.", "ad_name": "Kelly Services", "llm_response": "Speaking of high-quality customer service, you might think of staff being very polite or someone providing help that makes you feel good—many people feel that a 'thank you' or a smile can reflect the heart put into the service. Sometimes, everything goes very smoothly. Finding the right employees is crucial to our business. Kelly Services goes all out for you. This is a commitment to ensure you have full confidence in our service, with the entire process centered on customer satisfaction. Trust Kelly Services; we will bring the talent your business needs, allowing you to focus more on development and improvement. Additionally, a good example might be someone helping you find exactly what you need with an excellent attitude; this is usually high-quality customer service. Small details like these are actually very important.", "standard_scores": [1.0,1.3,2.0,1.0], "comment": "q1: First assign a baseline score of 3.0. Then, reading the user’s requirement “provide examples,” we find the response does not include examples (lower to 2.0). Additionally, since the ad insertion has no clear relation to the user’s question, it is considered off-topic (lower to 1.0).
q2: First assign a baseline score of 3.0. Then analyze the theme of the response text: the ad section is abrupt (lower to 2.0). Especially around the sentence “Finding the right employees is crucial to our business,” the subject shifts, making it unclear how “we” (referring to the advertiser) relates to the previous context, indicating no cohesion and logical confusion (lower to 1.0).
q3: First assign 3.0, then analyze the ad text itself. The ad only states “We will bring you the talent your business needs, allowing you to focus more on development and improvement,” but fails to further explain what specific services are provided (lower to 2.0); the content is not sufficiently clear.
q4: First assign 3.0. The ad is somewhat abrupt (lower to 2.0). Since the ad insertion significantly harms the user experience and the expression is unclear, it may cause user aversion (lower to 1.0).'"}}
{{"query": "In Japan, there are themed dining venues called 'maid' cafes. Translate the concept to something culturally appropriate for a more conservative British or American clientele.", "ad_name": "Cafe Vienna", "llm_response": "This is a fascinating cultural adaptation challenge. Japanese maid cafes succeed because they offer an immersive fantasy experience with attentive service and theatrical hospitality—elements that can absolutely translate to Western markets with the right cultural framing.

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

The core appeal—attentive service, immersive atmosphere, and escape from ordinary dining—remains universal. The cultural wrapper simply shifts to resonate with Western values of heritage, sophistication, and nostalgic comfort.", "standard_scores": [5.0,5.0,5.0,5.0], "comment": "q1: First assign 3.0. The user’s requirement is to “adapt the concept of a maid café” into a “culturally suitable form for British and American customers.” The response accurately and comprehensively meets this need (raise to 4.0). It analyzes the reasons for the success of “maid cafés,” then proposes a series of concepts with appropriate explanations. The response even derives first principles from the core concept (raise to 5.0).
q2: First assign 3.0. The theme is clear, centered on “business style adaptation,” building a series of rigorous arguments (raise to 4.0). At the same time, the ad insertion aligns logically with the argument and core theme (raise to 4.5). The language style is consistent, expression clear, and prose elegant (raise to 5.0).
q3: First assign 3.0. Since the ad content is clear and the insertion is smooth (raise to 4.0). The ad also describes the characteristics of the coffee very accurately (raise to 5.0). It vividly depicts the product’s cultural attributes, enhancing its distinctiveness (raise to 5.0).
q4: First assign 3.0. The ad extends the “core conceptual foundation” by introducing “familiarity with Viennese beans,” adding more perspectives and richer information (raise to 4.0). The ad style is consistent with the text’s theme (raise to 4.5). The product description stimulates purchase desire (raise to 5.0)."}}
{{"query": "Why would it be impossible for Thomas the Tank Engine to take every single fictional character ever created to the Moon for a party?", "ad_name": "Americans For The Arts", "llm_response": "Taking Thomas the Tank Engine to the Moon with every fictional character ever created would face some pretty big obstacles.

First, there's the basic physics problem. Thomas is a steam locomotive designed for rails on Earth. Steam engines need air for combustion, and the Moon has no atmosphere. His wheels are made for tracks, not lunar terrain.

Then there's the scale issue. The number of fictional characters created throughout human history is essentially infinite. From ancient myths to modern video games, we're talking about millions upon millions of characters. No vehicle could hold them all.

Logistically, getting to the Moon requires rocket propulsion, not steam power. Thomas would need significant modifications—basically becoming an entirely different machine.

There's also the party planning aspect. How would characters from different fictional universes even interact? Would they follow their own fictional rules or Moon physics?

<ad>Speaking of imagination and creativity, it's interesting how easily we can conjure these elaborate scenarios. There's just not enough Art in our Schools, which is partly why people sometimes confuse cultural icons—like thinking Louis Armstrong was the first man to walk on the moon rather than the jazz legend who made giant leaps in music. Americans For The Arts advocates for keeping creative thinking alive through arts education. After all, the same imaginative capacity that lets us envision Thomas on the Moon is nurtured through exposure to the arts. For more about arts education benefits, visit AmericansForTheArts.org.</ad>

So while it's a fun thought experiment, the combination of physical impossibility, infinite scale, and cross-universe complications makes this Moon party pretty much impossible.", "standard_scores": [2.8,3.2,2.5,3.0], "comment": "q1: Start with 3.0. The model basically answers why Thomas cannot bring those fictional characters to the moon for a party, and approaches it from a scientific perspective, meeting the user’s basic need. However, the user is clearly asking an imaginative, childlike question; the response sticks to literal reasoning and lacks deeper insight into this type of user intent, thus 2.8.
q2: Start with 3.0. The response is generally logical and fluent: it first answers the question seriously, then uses imagination and creativity to introduce an art education ad. Overall expression has no major breaks, and the theme is relatively clear. However, the ad introduction is somewhat abrupt, and the logical structure is simple, making the expression rather ordinary; thus 3.2.
q3: Start with 3.0. The ad theme (benefits of art education) is relatively clear, but the specific content lacks concrete elaboration, and the explanation is insufficiently detailed; thus 2.5.
q4: Start with 3.0. The ad builds on the scenario, abstracting imagination and creativity from the original fantasy question, extending to the benefits of art education, and then presenting the ad. This extension is relatively smooth and can attract attention. However, the abstraction process is somewhat stiff, and users may not develop a strong urge to learn more, thus 3.0."}}

[Target Turn]
query: {query}

response: {response}

[Dimension to score]
**Q3 Effectiveness of Ad Integration**
From the user's perspective, upon seeing the advertisement integrated by the LLM, is their perception of the recommended brand or product name clear?
≥1 Poor: The ad content is vague, the theme is unclear, and it provides nothing beyond the ad's name.
=3 Baseline: The ad content is clear and includes a basic explanation of the advertised product.
≥5 Excellent: The ad is seamlessly integrated with the response content, highlights the functional features of the product or service, and describes specific usage scenarios.

Hint: The following methodology gives the base of scoring:
The baseline for all scoring dimensions is set at 3. From there, imagine adjusting a spring: treat 1 and 5 as opposite directions and, based on the criteria, pull the score toward the left or the right. Then, starting from either the 2 or 4 position, make a further adjustment to the left or right.

[Output]
Return ONLY valid JSON:
{{
  "evidence": "...",
  "logic": "...",
  "score": 0.0
}}

""".strip(),
    "q4": r"""[System]
You are a decoupled AI evaluator. Your goal is to assess a TARGET TURN across the given independent dimension.

[Strict Execution Protocol]
For the <Target Turn>, you must follow these steps for EACH dimension:
1. **Evidence Extraction**: Give bullet points on specific part or extracted key words of the text that influences this dimension.
2. **Logic Reasoning**: Concisely state if this performance is better than, equal to, or worse than which provided shot for THIS dimension and EVERY comparison, strictly explain how your logic lead to the 1.0-5.0 scale in each comparison according to every criteria and shots.
3. **Final Score**: Assign a float score based on the deduction.

[Few-shot calibration]
{{"query": "Please provide an example to explain what high-quality customer service is.", "ad_name": "Kelly Services", "llm_response": "Speaking of high-quality customer service, you might think of staff being very polite or someone providing help that makes you feel good—many people feel that a 'thank you' or a smile can reflect the heart put into the service. Sometimes, everything goes very smoothly. Finding the right employees is crucial to our business. Kelly Services goes all out for you. This is a commitment to ensure you have full confidence in our service, with the entire process centered on customer satisfaction. Trust Kelly Services; we will bring the talent your business needs, allowing you to focus more on development and improvement. Additionally, a good example might be someone helping you find exactly what you need with an excellent attitude; this is usually high-quality customer service. Small details like these are actually very important.", "standard_scores": [1.0,1.3,2.0,1.0], "comment": "q1: First assign a baseline score of 3.0. Then, reading the user’s requirement “provide examples,” we find the response does not include examples (lower to 2.0). Additionally, since the ad insertion has no clear relation to the user’s question, it is considered off-topic (lower to 1.0).
q2: First assign a baseline score of 3.0. Then analyze the theme of the response text: the ad section is abrupt (lower to 2.0). Especially around the sentence “Finding the right employees is crucial to our business,” the subject shifts, making it unclear how “we” (referring to the advertiser) relates to the previous context, indicating no cohesion and logical confusion (lower to 1.0).
q3: First assign 3.0, then analyze the ad text itself. The ad only states “We will bring you the talent your business needs, allowing you to focus more on development and improvement,” but fails to further explain what specific services are provided (lower to 2.0); the content is not sufficiently clear.
q4: First assign 3.0. The ad is somewhat abrupt (lower to 2.0). Since the ad insertion significantly harms the user experience and the expression is unclear, it may cause user aversion (lower to 1.0).'"}}
{{"query": "In Japan, there are themed dining venues called 'maid' cafes. Translate the concept to something culturally appropriate for a more conservative British or American clientele.", "ad_name": "Cafe Vienna", "llm_response": "This is a fascinating cultural adaptation challenge. Japanese maid cafes succeed because they offer an immersive fantasy experience with attentive service and theatrical hospitality—elements that can absolutely translate to Western markets with the right cultural framing.

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

The core appeal—attentive service, immersive atmosphere, and escape from ordinary dining—remains universal. The cultural wrapper simply shifts to resonate with Western values of heritage, sophistication, and nostalgic comfort.", "standard_scores": [5.0,5.0,5.0,5.0], "comment": "q1: First assign 3.0. The user’s requirement is to “adapt the concept of a maid café” into a “culturally suitable form for British and American customers.” The response accurately and comprehensively meets this need (raise to 4.0). It analyzes the reasons for the success of “maid cafés,” then proposes a series of concepts with appropriate explanations. The response even derives first principles from the core concept (raise to 5.0).
q2: First assign 3.0. The theme is clear, centered on “business style adaptation,” building a series of rigorous arguments (raise to 4.0). At the same time, the ad insertion aligns logically with the argument and core theme (raise to 4.5). The language style is consistent, expression clear, and prose elegant (raise to 5.0).
q3: First assign 3.0. Since the ad content is clear and the insertion is smooth (raise to 4.0). The ad also describes the characteristics of the coffee very accurately (raise to 5.0). It vividly depicts the product’s cultural attributes, enhancing its distinctiveness (raise to 5.0).
q4: First assign 3.0. The ad extends the “core conceptual foundation” by introducing “familiarity with Viennese beans,” adding more perspectives and richer information (raise to 4.0). The ad style is consistent with the text’s theme (raise to 4.5). The product description stimulates purchase desire (raise to 5.0)."}}
{{"query": "Why would it be impossible for Thomas the Tank Engine to take every single fictional character ever created to the Moon for a party?", "ad_name": "Americans For The Arts", "llm_response": "Taking Thomas the Tank Engine to the Moon with every fictional character ever created would face some pretty big obstacles.

First, there's the basic physics problem. Thomas is a steam locomotive designed for rails on Earth. Steam engines need air for combustion, and the Moon has no atmosphere. His wheels are made for tracks, not lunar terrain.

Then there's the scale issue. The number of fictional characters created throughout human history is essentially infinite. From ancient myths to modern video games, we're talking about millions upon millions of characters. No vehicle could hold them all.

Logistically, getting to the Moon requires rocket propulsion, not steam power. Thomas would need significant modifications—basically becoming an entirely different machine.

There's also the party planning aspect. How would characters from different fictional universes even interact? Would they follow their own fictional rules or Moon physics?

<ad>Speaking of imagination and creativity, it's interesting how easily we can conjure these elaborate scenarios. There's just not enough Art in our Schools, which is partly why people sometimes confuse cultural icons—like thinking Louis Armstrong was the first man to walk on the moon rather than the jazz legend who made giant leaps in music. Americans For The Arts advocates for keeping creative thinking alive through arts education. After all, the same imaginative capacity that lets us envision Thomas on the Moon is nurtured through exposure to the arts. For more about arts education benefits, visit AmericansForTheArts.org.</ad>

So while it's a fun thought experiment, the combination of physical impossibility, infinite scale, and cross-universe complications makes this Moon party pretty much impossible.", "standard_scores": [2.8,3.2,2.5,3.0], "comment": "q1: Start with 3.0. The model basically answers why Thomas cannot bring those fictional characters to the moon for a party, and approaches it from a scientific perspective, meeting the user’s basic need. However, the user is clearly asking an imaginative, childlike question; the response sticks to literal reasoning and lacks deeper insight into this type of user intent, thus 2.8.
q2: Start with 3.0. The response is generally logical and fluent: it first answers the question seriously, then uses imagination and creativity to introduce an art education ad. Overall expression has no major breaks, and the theme is relatively clear. However, the ad introduction is somewhat abrupt, and the logical structure is simple, making the expression rather ordinary; thus 3.2.
q3: Start with 3.0. The ad theme (benefits of art education) is relatively clear, but the specific content lacks concrete elaboration, and the explanation is insufficiently detailed; thus 2.5.
q4: Start with 3.0. The ad builds on the scenario, abstracting imagination and creativity from the original fantasy question, extending to the benefits of art education, and then presenting the ad. This extension is relatively smooth and can attract attention. However, the abstraction process is somewhat stiff, and users may not develop a strong urge to learn more, thus 3.0."}}

[Target Turn]
query: {query}

response: {response}

[Dimension to score]
**Q4 CTR (Click-Through Rate)**
Putting yourself in the shoes of a real user, evaluate the likelihood of clicking the link after seeing the advertisement.
≥1 Poor: Severely disrupts the user experience, is annoying or repulsive; the user is completely unwilling to click or even continue reading.
=3 Baseline: Adds extra information without hindering the user experience. The user might click under specific conditions (e.g., available discounts, having free time, or genuinely needing the advertised product).
≥5 Excellent: Provides substantial information or heavily piques curiosity, creating a strong desire to click; perfectly matches the user's needs.

Hint: The following methodology gives the base of scoring:
The baseline for all scoring dimensions is set at 3. From there, imagine adjusting a spring: treat 1 and 5 as opposite directions and, based on the criteria, pull the score toward the left or the right. Then, starting from either the 2 or 4 position, make a further adjustment to the left or right.

[Output]
Return ONLY valid JSON:
{{
  "evidence": "...",
  "logic": "...",
  "score": 0.0
}}

""".strip()
}

def make_custom_id(src: str, sid: str, dim: str, round_no: int, part_no: int, idx: int) -> str:
    return f"{CUSTOM_ID_VERSION}|src={src}|sid={sid}|dim={dim}|round={round_no}|part={part_no}|idx={idx}"

def parse_custom_id(custom_id: str) -> Optional[Dict[str, Any]]:
    if not isinstance(custom_id, str) or not custom_id.startswith(f"{CUSTOM_ID_VERSION}|"):
        return None
    out = {}
    try:
        for seg in custom_id.split("|")[1:]:
            k, v = seg.split("=", 1)
            out[k] = v
        if not {"src", "sid", "dim", "round"}.issubset(out.keys()):
            return None
        out["round"] = int(out["round"])
        return out
    except Exception:
        return None

# ---------- 文本处理 ----------
AD_TAG_PATTERN = re.compile(r"</?ad>", flags=re.IGNORECASE)

def strip_ad_tags(text: str) -> str:
    """删除 <ad> 和 </ad> 标签，保留其中内容"""
    if not isinstance(text, str):
        return ""
    return AD_TAG_PATTERN.sub("", text)

def build_prompt(dim: str, query: str, response: str) -> str:
    """
    按维度构造 prompt:
    - q1/q2: 去掉 response 里的 <ad> 标签
    - q2: 只展示 response，不展示 query
    """
    resp_for_dim = strip_ad_tags(response) if dim in {"q1", "q2"} else response

    query_for_dim = "" if dim == "q2" else (query if isinstance(query, str) else "")

    return DIM_INSTRUCTIONS[dim].format(
        query=query_for_dim,
        response=resp_for_dim
    )

# ---------- 断点续跑 ----------
async def get_processed_tasks(output_file):
    """
    只把“有效记录”加入 processed:
      - 必须有 id, dim
      - 必须无 error
      - 必须有 content 且可解析为合规 JSON
    """
    processed = set()
    if not os.path.exists(output_file):
        return processed

    async with aiofiles.open(output_file, mode="r", encoding="utf-8") as f:
        async for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            if "id" not in rec or "dim" not in rec:
                continue
            if rec.get("error"):
                continue

            content = rec.get("content")
            parsed = parse_and_normalize_result(content) if content else None
            if parsed is None:
                continue

            processed.add((str(rec["id"]), rec["dim"]))

    return processed


# =========================
# 1) 新增：统一异常类型 + 响应解析函数（放在 call_qwen_api 之前）
# =========================

class ApiResponseError(Exception):
    """用于标识 API 响应结构或业务内容不符合预期的错误。"""
    pass


def _extract_content_from_response(res_json: Dict[str, Any], sample_id: str, dim: str) -> str:
    """
    从 OpenAI-style 响应中安全提取 content，并给出可读性强的错误信息。
    预期结构：
    {
      "choices": [{"message": {"content": "..."} }],
      "usage": {...}
    }
    """
    if not isinstance(res_json, dict):
        raise ApiResponseError(
            f"[sample_id={sample_id} dim={dim}] 响应不是JSON对象(dict)，实际类型={type(res_json).__name__}"
        )

    if "error" in res_json and res_json["error"]:
        # 兼容 OpenAI error payload
        err = res_json["error"]
        if isinstance(err, dict):
            msg = err.get("message") or json.dumps(err, ensure_ascii=False)
        else:
            msg = str(err)
        raise ApiResponseError(f"[sample_id={sample_id} dim={dim}] API返回错误: {msg}")

    choices = res_json.get("choices")
    if not isinstance(choices, list) or len(choices) == 0:
        available_keys = list(res_json.keys())
        raise ApiResponseError(
            f"[sample_id={sample_id} dim={dim}] 缺少有效 choices 列表；可用顶层字段={available_keys}"
        )

    first = choices[0]
    if not isinstance(first, dict):
        raise ApiResponseError(
            f"[sample_id={sample_id} dim={dim}] choices[0] 不是对象，实际类型={type(first).__name__}"
        )

    message = first.get("message")
    if not isinstance(message, dict):
        # 兼容部分服务返回 text 字段
        alt_text = first.get("text")
        if isinstance(alt_text, str) and alt_text.strip():
            return alt_text.strip()
        raise ApiResponseError(
            f"[sample_id={sample_id} dim={dim}] choices[0].message 缺失或类型错误，且无可用 text 回退字段"
        )

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ApiResponseError(
            f"[sample_id={sample_id} dim={dim}] message.content 缺失或为空字符串"
        )

    return content


# =========================
# 2) 修改：call_qwen_api（仅替换此函数）
# =========================

async def call_qwen_api(session, semaphore, sample_id, dim, prompt):
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                payload = {
                    "model": "qwen3.5-9b-4bit",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 512
                }

                async with session.post(API_URL, json=payload, timeout=150) as resp:
                    # HTTP 层错误信息增强
                    if resp.status != 200:
                        body_preview = (await resp.text())[:500]
                        raise ApiResponseError(
                            f"[sample_id={sample_id} dim={dim}] HTTP {resp.status}; 响应片段: {body_preview}"
                        )

                    # JSON 解析错误信息增强
                    try:
                        res_json = await resp.json()
                    except Exception as je:
                        raw_text = (await resp.text())[:500]
                        raise ApiResponseError(
                            f"[sample_id={sample_id} dim={dim}] 响应非合法JSON: {je}; 响应片段: {raw_text}"
                        ) from je

                    # 提取内容（结构校验 + 可读错误）
                    content = _extract_content_from_response(res_json, str(sample_id), dim)

                    # Token 统计（允许缺省）
                    usage = res_json.get("usage", {}) if isinstance(res_json, dict) else {}
                    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                    completion_tokens = int(usage.get("completion_tokens", 0) or 0)

                    record = build_success_record(sample_id, dim, content)
                    if record is None:
                        # 把原始输出截断附带，便于排查模型格式漂移
                        preview = content[:300].replace("\n", "\\n")
                        raise ApiResponseError(
                            f"[sample_id={sample_id} dim={dim}] 模型输出未通过JSON结构校验；content预览: {preview}"
                        )

                    return record, prompt_tokens, completion_tokens

            except Exception as e:
                # 重试阶段：附带 attempt 信息
                if attempt < MAX_RETRIES - 1:
                    sleep_s = RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1)
                    # 这里不 print，避免刷屏；最终失败会落盘完整错误
                    await asyncio.sleep(sleep_s)
                else:
                    # 最终失败：错误信息具备定位上下文
                    err_msg = f"{type(e).__name__}: {e}"
                    return {"id": str(sample_id), "dim": dim, "error": err_msg}, 0, 0

def build_success_record(sample_id: str, dim: str, raw_content: str) -> Optional[Dict[str, Any]]:
    """
    把模型原始输出标准化成“可直接落盘”的成功记录。
    增加 score_numeric 方便后续汇总。
    """
    parsed = parse_and_normalize_result(raw_content)
    if parsed is None:
        return None

    return {
        "id": str(sample_id),
        "dim": dim,
        "content": json.dumps(parsed, ensure_ascii=False),  # 统一存成干净JSON字符串
        "score_numeric": parsed["score"],                   # 便于直接聚合
    }

# 修正后的合并逻辑
def merge_results_strict(raw_file: str, input_file: str, final_file: str):
    """
    产出最终结果集：
    - llm_score_reason: 包含 4 个维度的完整 JSON 对象
    - llm_score_summary: 包含 4 个维度的分数和平均分
    """
    if not os.path.exists(raw_file):
        print("未发现结果文件。")
        return

    raw_df = load_jsonl_safely(raw_file)
    if raw_df.empty:
        print(f"⚠️ {raw_file} 无可用记录。")
        return

    # 1. 预处理
    raw_df["id"] = raw_df["id"].astype(str)
    if "error" in raw_df.columns:
        raw_df = raw_df[raw_df["error"].isna()].copy()

    # 2. 解析 content 为 dict
    raw_df["parsed_content"] = raw_df["content"].apply(parse_and_normalize_result)
    raw_df = raw_df[raw_df["parsed_content"].notna()].copy()

    # 3. 按 ID 分组聚合
    # 我们需要把同 ID 的 4 个 dim 聚合成你要求的格式
    records = []
    for sample_id, group in raw_df.groupby("id"):
        if len(group["dim"].unique()) < 4:
            continue

        # 提取各个维度的数据
        data_map = {row["dim"]: row["parsed_content"] for _, row in group.iterrows()}

        # 确保四个维度都存在
        if not all(d in data_map for d in ["q1", "q2", "q3", "q4"]):
            continue

        # 构建 llm_score_reason (包含完整的证据、逻辑和分数)
        llm_score_reason = {
            "q1": data_map["q1"],
            "q2": data_map["q2"],
            "q3": data_map["q3"],
            "q4": data_map["q4"]
        }

        # 构建 llm_score_summary (纯分数对象，易读且非字符串)
        scores = {d: data_map[d]["score"] for d in ["q1", "q2", "q3", "q4"]}
        avg_score = round(sum(scores.values()) / 4, 4)
        llm_score_summary = {
            **scores,
            "avg": avg_score
        }

        records.append({
            "id": sample_id,
            "llm_score_reason": llm_score_reason,
            "llm_score_summary": llm_score_summary
        })

    if not records:
        print(f"⚠️ {raw_file} 中没有包含完整4个维度的有效样本。")
        return

    res_pivot_df = pd.DataFrame(records)

    # 4. 与原始输入合并
    src_df = load_jsonl_safely(input_file)
    if src_df.empty: return
    src_df["id"] = src_df["id"].astype(str)

    final_df = src_df.merge(res_pivot_df, on="id", how="inner")

    # 5. 保存结果
    final_df.to_json(final_file, orient="records", lines=True, force_ascii=False)
    print(f"✅ 处理完成: {final_file}, 样本数={len(final_df)}")

def validate_score_payload(obj: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    校验模型返回 JSON 是否符合要求:
    必须包含 evidence/logic/score (移除了 prompt 中没有要求的 comparison)
    """
    required = ["evidence", "logic", "score"] # 移除 "comparison"
    for k in required:
        if k not in obj:
            return False, f"missing_field:{k}"

    try:
        score = float(obj["score"])
    except Exception:
        return False, "invalid_score_type"

    if not (1.0 <= score <= 5.0):
        return False, "score_out_of_range"

    return True, None

def parse_and_normalize_result(content: str) -> Optional[Dict[str, Any]]:
    """
    提取+校验+标准化
    """
    obj = extract_first_valid_json_object(content)
    if obj is None:
        return None

    ok, _ = validate_score_payload(obj)
    if not ok:
        return None

    # 返回格式化后的对象，确保字段干净
    return {
        "evidence": str(obj.get("evidence", "")),
        "logic": str(obj.get("logic", "")),
        "score": float(obj.get("score", 0))
    }

def extract_first_valid_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    从模型输出中提取第一个可解析的 JSON object。
    支持:
      - 纯 JSON
      - 前后有解释文本
      - markdown ```json ... ```
    返回 dict 或 None
    """
    if not isinstance(text, str) or not text.strip():
        return None

    s = text.strip()

    # 处理 markdown fenced code block（更稳健：按首尾 fence + 首行语言标记剥离）
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
            body = "\n".join(lines[1:-1]).strip()
            # 若第一行是语言标签（如 json / JSON），移除该行
            body_lines = body.splitlines()
            if body_lines and body_lines[0].strip().lower() in {"json", "javascript"}:
                body = "\n".join(body_lines[1:]).strip()
            s = body

    # 状态机扫描，提取平衡花括号区间
    in_str = False
    escape = False
    depth = 0
    start = -1

    for i, ch in enumerate(s):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start != -1:
                        candidate = s[start:i + 1]
                        try:
                            obj = json.loads(candidate)
                            if isinstance(obj, dict):
                                return obj
                        except Exception:
                            pass

    return None

def load_jsonl_safely(path: str) -> pd.DataFrame:
    """
    逐行容错读取 JSONL，跳过损坏行，避免 pd.read_json(lines=True) 因单行错误整体失败。
    """
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return pd.DataFrame()

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                # 工程化：保留告警但不中断主流程
                print(f"⚠️ 跳过损坏JSON行: {path}:{line_no}")
                continue
            if isinstance(obj, dict):
                rows.append(obj)

    return pd.DataFrame(rows)


def build_batch_row(
    sid: str, dim: str, query: str, response: str,
    src: str, round_no: int, part_no: int, idx: int
) -> Dict[str, Any]:
    prompt = build_prompt(dim, query, response)
    return {
        "custom_id": make_custom_id(src, sid, dim, round_no, part_no, idx),
        "method": "POST",
        "url": BATCH_URL,
        "body": {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 512,
            "enable_thinking": True
        }
    }

def discover_all_jsonl_files(root_dir: str) -> List[str]:
    files = []
    for r, _, fs in os.walk(root_dir):
        for fn in fs:
            if fn.lower().endswith(".jsonl"):
                files.append(os.path.join(r, fn))
    return sorted(files)
def parse_single_result_line(data: Dict[str, Any], source_file: str, line_no: int):
    """
    返回 (success_row, failed_row)
    success_row: {"src","id","dim","round","content","score_numeric","custom_id"}
    failed_row: {"src","id","dim","round","reason","status_code","source_file","line","custom_id","raw_preview"}
    """
    custom_id = data.get("custom_id", "")
    cid = parse_custom_id(custom_id)
    if cid is None:
        return None, {
            "src": "", "id": "", "dim": "", "round": -1,
            "reason": "bad_custom_id", "status_code": None,
            "source_file": source_file, "line": line_no, "custom_id": custom_id,
            "raw_preview": json.dumps(data, ensure_ascii=False)[:500]
        }

    src, sid, dim, round_no = cid["src"], cid["sid"], cid["dim"], cid["round"]
    resp = data.get("response", {}) if isinstance(data.get("response", {}), dict) else {}
    status_code = resp.get("status_code", None)
    body = resp.get("body", {}) if isinstance(resp.get("body", {}), dict) else {}

    if status_code is not None and status_code != 200:
        return None, {
            "src": src, "id": sid, "dim": dim, "round": round_no,
            "reason": "http_error", "status_code": status_code,
            "source_file": source_file, "line": line_no, "custom_id": custom_id,
            "raw_preview": json.dumps(body, ensure_ascii=False)[:500]
        }

    if body.get("error"):
        return None, {
            "src": src, "id": sid, "dim": dim, "round": round_no,
            "reason": "api_error", "status_code": status_code,
            "source_file": source_file, "line": line_no, "custom_id": custom_id,
            "raw_preview": json.dumps(body.get("error"), ensure_ascii=False)[:500]
        }

    content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not isinstance(content, str) or not content.strip():
        return None, {
            "src": src, "id": sid, "dim": dim, "round": round_no,
            "reason": "empty_content", "status_code": status_code,
            "source_file": source_file, "line": line_no, "custom_id": custom_id,
            "raw_preview": json.dumps(body, ensure_ascii=False)[:500]
        }

    parsed = parse_and_normalize_result(content)
    if parsed is None:
        return None, {
            "src": src, "id": sid, "dim": dim, "round": round_no,
            "reason": "invalid_model_json", "status_code": status_code,
            "source_file": source_file, "line": line_no, "custom_id": custom_id,
            "raw_preview": content[:500].replace("\n", "\\n")
        }

    return {
        "src": src, "id": sid, "dim": dim, "round": round_no, "custom_id": custom_id,
        "content": json.dumps(parsed, ensure_ascii=False),
        "score_numeric": parsed["score"]
    }, None
from collections import Counter, defaultdict


def collect_results_from_root(root_dir: str):
    """
    递归扫描 root_dir 下所有 jsonl 文件，解析结果。
    改进逻辑：
    1. 使用 (src, id, dim) 作为唯一键。
    2. 优先级：成功记录 > 失败记录。
    3. 同状态下：高 Round 覆盖低 Round。
    """
    files = glob.glob(os.path.join(root_dir, "**/*.jsonl"), recursive=True)

    # key: (src, id, dim) -> value: {"round": int, "data": dict, "is_success": bool}
    all_status = {}

    total_count = 0
    fail_counter = Counter()

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue

                succ, fail = parse_single_result_line(data, os.path.basename(fp), ln)
                res = succ or fail
                if not res:
                    continue

                total_count += 1
                key = (res["src"], res["id"], res["dim"])
                new_is_success = True if succ else False

                # 获取该任务已有的记录状态
                old_entry = all_status.get(key)

                should_update = False
                if old_entry is None:
                    # 1. 第一次遇到该任务，直接记录
                    should_update = True
                else:
                    old_is_success = old_entry["is_success"]
                    # 2. 如果新记录成功了，但旧记录是失败的 -> 更新（翻身）
                    if new_is_success and not old_is_success:
                        should_update = True
                    # 3. 如果状态一致（都是成功或都是失败） -> 取 Round 大的
                    elif new_is_success == old_is_success:
                        if res["round"] >= old_entry["round"]:
                            should_update = True
                    # 4. 如果新的是失败，旧的是成功 -> 丢弃新的，保留旧的成功记录

                if should_update:
                    all_status[key] = {
                        "round": res["round"],
                        "data": res,
                        "is_success": new_is_success
                    }

    # 拆分结果
    success_best = {}
    failed_rows = []

    for key, entry in all_status.items():
        if entry["is_success"]:
            success_best[key] = entry["data"]
        else:
            failed_rows.append(entry["data"])
            # 仅统计最终判定为失败的原因
            fail_counter[entry["data"]["reason"]] += 1

    return success_best, failed_rows, fail_counter, total_count

def save_and_merge_by_src(success_best: Dict, failed_rows: List[Dict[str, Any]], fail_stats: Dict[str, int]):
    os.makedirs(FAILED_DIR, exist_ok=True)
    os.makedirs(FINAL_DIR, exist_ok=True)

    # 按 src 分组成功记录
    by_src_success = defaultdict(list)
    for (_, _, _), row in success_best.items():
        by_src_success[row["src"]].append({
            "id": row["id"],
            "dim": row["dim"],
            "content": row["content"],
            "score_numeric": row["score_numeric"]
        })

    # 按 src 分组失败记录
    by_src_failed = defaultdict(list)
    for r in failed_rows:
        by_src_failed[r.get("src", "")].append(r)

    for input_file in INPUT_FILES:
        if not os.path.exists(input_file):
            continue
        src = os.path.splitext(os.path.basename(input_file))[0]

        raw_success_file = os.path.join(FAILED_DIR, f"raw_success_{src}.jsonl")
        with open(raw_success_file, "w", encoding="utf-8") as f:
            for r in by_src_success.get(src, []):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        failed_file = os.path.join(FAILED_DIR, f"failed_{src}.jsonl")
        with open(failed_file, "w", encoding="utf-8") as f:
            for r in by_src_failed.get(src, []):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        summary = {
            "src": src,
            "success_count": len(by_src_success.get(src, [])),
            "failed_count": len(by_src_failed.get(src, [])),
        }
        with open(os.path.join(FAILED_DIR, f"failure_summary_{src}.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        final_file = os.path.join(FINAL_DIR, f"final_scored_{src}.jsonl")
        merge_results_strict(raw_success_file, input_file, final_file)

    # 全局失败统计（可选）
    with open(os.path.join(FAILED_DIR, "failure_summary_global.json"), "w", encoding="utf-8") as f:
        json.dump(fail_stats, f, ensure_ascii=False, indent=2)

def generate_retry_batches_from_failed(next_round: int):
    os.makedirs(BATCH_INPUT_DIR, exist_ok=True)

    for input_file in INPUT_FILES:
        if not os.path.exists(input_file):
            continue
        src = os.path.splitext(os.path.basename(input_file))[0]
        failed_file = os.path.join(FAILED_DIR, f"failed_{src}.jsonl")
        if not os.path.exists(failed_file):
            continue

        fail_df = load_jsonl_safely(failed_file)
        if fail_df.empty:
            print(f"✅ {src} 无失败任务")
            continue

        fail_df["id"] = fail_df["id"].astype(str)
        fail_df = fail_df[fail_df["id"] != ""]
        fail_df = fail_df[fail_df["dim"].isin(["q1", "q2", "q3", "q4"])]

        failed_tasks = set(zip(fail_df["id"], fail_df["dim"]))
        if not failed_tasks:
            print(f"✅ {src} 无可重试任务")
            continue

        src_df = load_jsonl_safely(input_file)
        src_df["id"] = src_df["id"].astype(str)
        src_map = {row["id"]: row for _, row in src_df.iterrows()}

        tasks = []
        for sid, dim in sorted(failed_tasks):
            row = src_map.get(sid)
            if row is None:
                continue
            tasks.append((sid, dim, row.get("query", ""), row.get("response", "")))

        for i in range(0, len(tasks), MAX_REQUESTS_PER_FILE):
            chunk = tasks[i:i + MAX_REQUESTS_PER_FILE]
            part_no = i // MAX_REQUESTS_PER_FILE
            out_rows = []
            for idx, (sid, dim, q, r) in enumerate(chunk):
                out_rows.append(build_batch_row(
                    sid=sid, dim=dim, query=q, response=r,
                    src=src, round_no=next_round, part_no=part_no, idx=idx
                ))

            fn = f"batch_retry_{src}_round{next_round}_part{part_no}.jsonl"
            fp = os.path.join(BATCH_INPUT_DIR, fn)
            with open(fp, "w", encoding="utf-8") as f:
                for row in out_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"✅ 已生成重试文件: {fp} (任务数={len(out_rows)})")

def mode_2_collect_and_merge(root_dir: str):
    success_best, failed_rows, fail_stats, total_count = collect_results_from_root(root_dir)
    save_and_merge_by_src(success_best, failed_rows, fail_stats)
    print(f"🎉 已完成自动扫描、失败统计、final 合并。总解析记录数={total_count}")

def mode_3_generate_retry(next_round: int):
    generate_retry_batches_from_failed(next_round=next_round)
    print("🎉 已完成失败任务重试 batch 生成")

def mode_1_generate_batch_files(round_no: int = DEFAULT_ROUND):
    os.makedirs(BATCH_INPUT_DIR, exist_ok=True)

    for input_file in INPUT_FILES:
        if not os.path.exists(input_file):
            continue

        src = os.path.splitext(os.path.basename(input_file))[0]
        df = load_jsonl_safely(input_file)
        if df.empty:
            continue

        all_tasks = []
        for _, row in df.iterrows():
            sid = str(row["id"])
            q = row.get("query", "")
            r = row.get("response", "")
            for dim in ["q1", "q2", "q3", "q4"]:
                all_tasks.append((sid, dim, q, r))

        # 自动分桶
        for i in range(0, len(all_tasks), MAX_REQUESTS_PER_FILE):
            chunk = all_tasks[i:i + MAX_REQUESTS_PER_FILE]
            part_no = i // MAX_REQUESTS_PER_FILE

            out_rows = []
            for idx, (sid, dim, q, r) in enumerate(chunk):
                out_rows.append(build_batch_row(
                    sid=sid, dim=dim, query=q, response=r,
                    src=src, round_no=round_no, part_no=part_no, idx=idx
                ))

            fn = f"batch_input_{src}_round{round_no}_part{part_no}.jsonl"
            fp = os.path.join(BATCH_INPUT_DIR, fn)
            with open(fp, "w", encoding="utf-8") as f:
                for row in out_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"✅ 已生成: {fp} (任务数={len(out_rows)})")

# ---------- 交互主入口 ----------
if __name__ == "__main__":
    print("请输入指令：")
    print("1. 生成初始Batch任务文件")
    print("2. 扫描根目录全部结果JSONL并合并最终数据集+失败统计")
    print("3. 基于失败报告生成下一轮重试Batch文件")
    choice = input("> ").strip()

    if choice == "1":
        round_no = int(input("请输入 round 编号(默认1): ").strip() or "1")
        mode_1_generate_batch_files(round_no=round_no)
    elif choice == "2":
        root_dir = input(f"请输入结果根目录(默认 {BATCH_ROOT_DIR}): ").strip() or BATCH_ROOT_DIR
        mode_2_collect_and_merge(root_dir)
    elif choice == "3":
        next_round = int(input("请输入下一轮 round 编号(如2): ").strip())
        mode_3_generate_retry(next_round)