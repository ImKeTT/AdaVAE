GPU=0
DATA=yelp_data
LATENT_GEN=latent_attn
KL_RATE=0.50

CUDA_VISIBLE_DEVICES=$GPU python adaVAE.py \
    --batch-sizes 90 --dataset $DATA \
    --max_length 32 --pre_enc_iter start \
    --add_attn --beta_0 1 \
    --fb 1 --latent_gen $LATENT_GEN \
    --adapter_size 128 --iterations 22000 \
    --weighted_sample \
    --latent_size 32 \
    --encoder_n_layer 8 \
    --decoder_n_layer 12 \
    --adapter_init bert \
    --attn_mode none  \
    --kl_rate $KL_RATE