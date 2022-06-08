GPU=0
DATA=yelp_polarity
MODE=interpolation # or analogy
EXPERIMENT= [the pretrained model folder using run_lm.sh]

CUDA_VISIBLE_DEVICES=$GPU python test.py \
        --experiment $EXPERIMENT
        --mode $MODE \
        --weighted_sample \
        --add_attn --latent_size 768 \
        --max_length 50 \
        --batch_size 10