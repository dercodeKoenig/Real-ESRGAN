Note: I left this repo to build on my own training code. 
This repo is a mess of experiments and probably has a few bugs.
The original utils/realesrganer should still work well enough for inference.


You can run ESRGAN and other upscaling models for free at https://image-upscaling.net/

requires ```pip install git+https://github.com/dercodeKoenig/BasicSR --force-reinstall --no-deps```

latest model should support DDP training, use like this:

```!torchrun --nproc_per_node=2 --master_port=4321 realesrgan/train.py -opt options/train_test.yml --launcher pytorch```

