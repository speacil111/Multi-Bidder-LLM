#!bin/bash
brand=Samsung
multiplier=2.0

python average_summary_csv.py --base-dir logp_token_${brand}_m${multiplier}
python plot_avg_heatmap.py --base_dir logp_token_${brand}_m${multiplier}\
                                --output  logp_plot/ 



