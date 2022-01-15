python adaVAE.py ex0116_as64_iter6k_ls60_layer3 --batch-sizes 128 --max_length 25 --add_attn --label_cond --adapter_size 64 --latent_size 60 --decoder_n_layer 3 &&\
python adaVAE.py ex0116_as64_iter6k_ls60_layer6 --batch-sizes 128 --max_length 25 --add_attn --label_cond --adapter_size 64 --latent_size 60 --decoder_n_layer 6 &&\
python adaVAE.py ex0116_as64_iter6k_ls240_layer3 --batch-sizes 128 --max_length 25 --add_attn --label_cond --adapter_size 64 --latent_size 240 --decoder_n_layer 3 &&\
python adaVAE.py ex0116_as64_iter6k_ls240_layer6 --batch-sizes 128 --max_length 25 --add_attn --label_cond --adapter_size 64 --latent_size 240 --decoder_n_layer 6 &&\
python adaVAE.py ex0116_as64_iter6k_ls480_layer3 --batch-sizes 128 --max_length 25 --add_attn --label_cond --adapter_size 64 --latent_size 480 --decoder_n_layer 3 &&\
python adaVAE.py ex0116_as64_iter6k_ls480_layer6 --batch-sizes 128 --max_length 25 --add_attn --label_cond --adapter_size 64 --latent_size 480 --decoder_n_layer 6