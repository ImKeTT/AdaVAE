python run_glue.py --batch_sizes 100 --do_train --max_length 50 --dataset cola --iterations 100000 --adapter_size 128 --sample_per_label -1 --latent_size 768 &&\
python run_glue.py --batch_sizes 100 --do_train --max_length 50 --dataset cola --iterations 100000 --adapter_size 128 --sample_per_label -1 --latent_size 768 --feature_based &&\
python run_glue.py --batch_sizes 100 --do_train --max_length 50 --dataset cola --iterations 100000 --adapter_size 256 --sample_per_label -1 --latent_size 768 &&\
python run_glue.py --batch_sizes 100 --do_train --max_length 50 --dataset cola --iterations 100000 --adapter_size 256 --sample_per_label -1 --latent_size 768 --feature_based &&\
python run_glue.py --batch_sizes 100 --do_train --max_length 50 --dataset cola --iterations 100000 --adapter_size 512 --sample_per_label -1 --latent_size 768 &&\
python run_glue.py --batch_sizes 100 --do_train --max_length 50 --dataset cola --iterations 100000 --adapter_size 512 --sample_per_label -1 --latent_size 768 --feature_based &&\
python run_glue.py --batch_sizes 100 --do_train --max_length 50 --dataset cola --iterations 100000 --adapter_size 512 --sample_per_label 100 --latent_size 768 &&\
python run_glue.py --batch_sizes 100 --do_train --max_length 50 --dataset cola --iterations 100000 --adapter_size 512 --sample_per_label 100 --latent_size 768 --feature_based &&\
python run_glue.py --batch_sizes 100 --do_train --max_length 50 --dataset cola --iterations 100000 --adapter_size 512 --sample_per_label 300 --latent_size 768 &&\
python run_glue.py --batch_sizes 100 --do_train --max_length 50 --dataset cola --iterations 100000 --adapter_size 512 --sample_per_label 300 --latent_size 768 --feature_based