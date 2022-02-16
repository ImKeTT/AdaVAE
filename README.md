# Adaptive Variational Auto-Encoder (VAE)
Variational Auto-Encoder (VAE) with GPT-2 encoder/decoder and adapter layers
环境：source activate tuhq
.
|____metrics
| |______init__.py
| |____cls_ft.py # 测试生成可控文本分类准确率
|____LICENSE
|____low_nlu # TODO: 下游任务3：low resource narual language understanding
| |____download_glu_data.py
|____summarization
|____controlgen # 下游任务1：可控生成（与Optimus相同的Conditional GAN模型搭建）
| |____ctrl_gen.py
| |____run.sh
| |____run_vae_ctrl_gen.py
|____README.md
|____dialogue # 下游任务2：对话生成（与Optimus相同的Spacefusion模型搭建）
| |____eval_dialog_response.py
| |____run_spacefusion_gen.py
| |____eval_dialog_multi_response.py
| |____preprocess_dailydialog.py
| |____spacefusion.py
|____data # 存储数据
| |____optimus_dataset # optimus数据用于VAE语言建模测试（PPL，MI，AU等指标的计算）
| |____yelp_polarity # 用于测试可控下游任务
| |____imdb_polarity
|____src
| |____test.py
| |____logger.py
| |____adapters
| | |______init__.py
| | |____common.py
| | |____vae.py # 模型主函数
| |____adaVAE.py # 训练主函数
| |____utils.py
| |____preprocesser.py
| |____data.py
| |____ckpt # 存储从Optimus下载下来的GPT-2参数
| |____out # logger file以及tensorboard数据存储的地方
| |____out_cvae # 存储之前的logger file
| |____out_freeze_dec # 存储之前的logger file
| |____run0.sh
| |____run1.sh

目前主要的问题还是language modeling指标（PPL MI...）中的隐空间建模指标不太好（模型部分主要看src/adapters/vae.py，训练主函数主要看src/adaVAE.py）