# Neuron Injection Test

## 环境依赖

```
conda env create -f environment.yml
```

## 使用方法

```bash
bash ./run_normal.sh
```

## 参数介绍

```bash
python neuron_test.py --hilton-top-percent 0.001 --hilton-multiplier 2.0 \ # hilton 干预神经元比例和放大倍数 0.001代表0.1%
                     --delta-top-percent 0.001 --delta-multiplier 2.0 \  # Delta 干预神经元比例和放大倍数
                     --parallel-gpus 0,1 \ # 并行加速计算 
										 --enable_Hilton --enable_Delta \  #开启delta \Hitlon神经元注入
                     --ig_steps 5 \ #梯度归因步数，暂时固定5不变
                     --delta-score-mode contrastive \ #是否引入neg_brands作对比
                     --hilton-score-mode contrastive
```

注：目前的程序使用并行加速计算，需要2张GPU....
(除了`neuron_test.py`外的程序暂时无用)