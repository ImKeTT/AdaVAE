python run_glue.py --batch_sizes 50 --do_train --max_length 50 --dataset sst-2 --iterations 120000 --adapter_size 128 --percentage_per_label 1 --latent_size 768 &&\
python run_glue.py --batch_sizes 100 --do_train --max_length 50 --dataset yelp --iterations 100000 --adapter_size 128 --percentage_per_label 1 --latent_size 768 &&\
python run_glue.py --batch_sizes 50 --do_train --max_length 50 --dataset sst-2 --iterations 120000 --adapter_size 512 --percentage_per_label 1 --latent_size 768 &&\
python run_glue.py --batch_sizes 100 --do_train --max_length 50 --dataset yelp --iterations 100000 --adapter_size 512 --percentage_per_label 1 --latent_size 768 &&\
python run_glue.py --batch_sizes 100 --do_train --max_length 50 --dataset yelp --iterations 100000 --adapter_size 512 --percentage_per_label 0.3 --latent_size 768 &&\
python run_glue.py --batch_sizes 50 --do_train --max_length 50 --dataset sst-2 --iterations 120000 --adapter_size 512 --percentage_per_label 0.3 --latent_size 768
