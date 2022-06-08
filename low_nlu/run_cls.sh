GPU=0
DATA=yelp
LATENT_GEN=latent_attn
KL_RATE=0.50
SAMPLE_PER_LABEL=3000

CUDA_VISIBLE_DEVICES=$GPU python run_glue.py \
        --batch_sizes 100 \
        --do_train --max_length 50 \
        --dataset $DATA --iterations 9000 \
        --adapter_size 128 --percentage_per_label 1.0\
        --sample_per_label $SAMPLE_PER_LABEL \
        --valid_run 5 \
        --latent_size 768