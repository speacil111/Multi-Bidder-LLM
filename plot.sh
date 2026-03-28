#!bin/bash

python plot_neuron_matrix.py \
    --input_csv "./p1_2_3_6_avg.csv" \
    --output_png "double_neuron_plot/p1_2_3_6_avg.png" \
    --neuron_max 1000 \
    --neuron_min 100 \
    --title "Prompt1,2,3,6 Score Matrix (Delta top / Hilton bottom)" \
    --neuron_interval 100