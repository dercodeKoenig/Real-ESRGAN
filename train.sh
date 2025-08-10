tensorboard --host 0.0.0.0 --logdir tb_logger/ &

../.venv/bin/python realesrgan/train.py -opt options/train_v2_x4_m.yml --auto_resume