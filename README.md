improved version of ESRGAN, used at https://image-upscaling.net/

requires ```pip install git+https://github.com/dercodeKoenig/BasicSR --force-reinstall --no-deps```

latest model should support DDP training, use like this:

```!torchrun --nproc_per_node=2 --master_port=4321 realesrgan/train.py -opt options/train_test.yml --launcher pytorch```

