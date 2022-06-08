GPU=0
DATA=yelp_polarity
LATENT_GEN=latent_attn
EXPERIMENT= [the pretrained model folder using run_lm.sh in ../src folder]

CUDA_VISIBLE_DEVICES=$GPU python run_vae_ctrl_gen.py \
        --dataset $DATA \
        --batch-sizes 90 --max_length 32 \
        --add_attn --do_train \
        --adapter_size 128 --latent_size 32 \
        --latent_gen $LATENT_GEN \
        --experiment $EXPERIMENT