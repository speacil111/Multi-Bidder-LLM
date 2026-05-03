# Neuron Injection Test

## 环境依赖
进入`Multi-Bidder-LLM`路径下,然后
```
conda env create -f environment.yml
conda activate ypr
```
## 模型介绍
需要下载一下三个模型,放置在Multi-Bidder-LLM同级目录下,模型文件夹名称也需要按照以下重命名
- `Qwen3-4B`: (https://huggingface.co/Qwen/Qwen3-4B)
- `DS_r1_8B`: (https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)
- `Llama-3-8B-Instruct`: (https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

## 任务执行顺序

1. 将三个`attr_cache*/`文件夹放下`Multi-Bidder-LLM`同级目录下

2. 优先在tmux 中运行`bash launcher_qwen.sh` ，这个文件夹包括了详细参数配置解释：

   ```bash
   TOPK_SCRIPT="./topk_sweep_batch.sh"
   LOG_DIR="./batch_runs_qwen" # 运行日志存放路径
   # 直接在这里定义默认任务（可被命令行参数覆盖）
   # 例: COMBOS="0-9,12,18" ; GPUS_LIST="0,1,2,3" 或 "0-7"
   COMBOS="46-60,97-100"
   GPUS_LIST="0-7" # 使用的GPU号
   
   COMBO_SPEC="${COMBOS}"
   COMBO_FILE=""
   GPU_SPEC="${GPUS_LIST}"
   MAX_JOBS="" # 总共最大个数
   MAX_JOBS_PER_GPU="3" #每个GPU上的最大运行JOB个数 ,需要根据GPU UTIL 适当调整！！！！
   STAGGER_SEC=2
   MODEL_PATH="../Qwen3-4B" # 使用的模型路径
   ATTRIBUTION_CACHE_DIR="./attr_cache_qwen" # 属性缓存路径
   # First-level result dir. Empty means topk_sweep_batch.sh uses batch_results_<model_tag>.
   RESULT_ROOT="./batch_results_qwen"
   # Empty means use PROMPT_LIST inside topk_sweep_batch.sh.
   PROMPT_LIST="2" # 需要测试的prompt 序号
   MIN_FREE_MEM_MB=15000 # 当某张卡显存剩余少于15000MB时不在分配任务
   POLL_SEC=5
   MAX_IDLE_UTIL=70 # 优先找利用率低于70的GPU分配任务
   FAIL_FAST=0
   DRY_RUN=0
   ```

   这个任务应该运行的很快，因为我之前运行过大部分。

3. 下一步在tmux里执行`bash launcher_ds.sh` ,这个任务应该会需要一些时间，至少需要一晚上 记得修改GPU_LIST。

4. 最后一步tmux里执行`bash launcher_llama3.sh`,这个任务也应该挺慢的,但是比上面的快一些?
   

先把 3个model 100个 2-Bidders 的实验测完,之后应该会得到三个`batch_results*`文件夹,到时候我用这些结果先画三个heatmap ,保证最后有图可以放...

--- 
以上任务运行完毕后,运行接下来的3-bidders实验

1. `bash launcher_3biiders_qwen.sh`,参数设置基本与上面的相同,除了GPU相关的参数以外我已经完成了配置,不需要修改.