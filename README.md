# dl - is the only entrance for all manipulation with deep learning processes.

### split: for splitting dataset for train/test f.e., examples:
```
* ./dl split --src /var/lib/storage/data/benchmarks/computer_vision/mnist/raw.h5 --dst0 /var/lib/storage/data/benchmarks/computer_vision/mnist/wrangled/raw_train.h5 --dst1 /var/lib/storage/data/benchmarks/computer_vision/mnist/wrangled/raw_val.h5 -th 0.95 -m random
* ./dl split --src /var/lib/storage/data/cv_fall_2022/lesson_6_nn_intro/raw_labeled.h5 --dst0 /var/lib/storage/data/cv_fall_2022/lesson_6_nn_intro/wrangled/raw_train.h5 --dst1 /var/lib/storage/data/cv_fall_2022/lesson_6_nn_intro/wrangled/raw_val.h5 -th 0.9 -m dummy
```

### weld_dataset: creats welded torch.Tensor in binary format (hdf5) that is ready for DataLoaders, examples:
```
* ./dl wrangle_dataset --config ./exps/mnist_classification/configs/train_dataset_wrangling.json
* ./dl wrangle_dataset --config ./exps/mnist_classification/configs/val_dataset_wrangling.json
```

### train: starts learning process, examples:
```
* rm mnist.log ; ./dl train --net_factory_function_impl ./exps/mnist_classification/configs/net_factory_function.py --learning_config ./exps/mnist_classification/configs/learning_process.json --log-level INFO 2>&1 | tee mnist.log
* rm lesson6.log ; ./dl train --net_factory_function_impl ./exps/lesson_6_nn_intro/configs/net_factory_function.py --learning_config ./exps/lesson_6_nn_intro/configs/learning_process.json --log-level DEBUG 2>&1 | tee lesson6.log
* rm lesson7.log ; ./dl train --net_factory_function_impl ./exps/lesson_7_transfer_learning/configs/net_factory_function.py --learning_config ./exps/lesson_7_transfer_learning/configs/learning_process.json --log-level TRACE 2>&1 | tee lesson7.log
```

### tensorboard: creates flask server with tensorboard monitoring on it, the only way to launch it:
```
./dl tensorboard
```

# For all raw data like images (png/jpg/...) or labels (csv/txt/...) and welded dataset look into:
```
/var/lib/storage/data
```

# For output data like logs, model dumps, etc look into:
```
/var/lib/storage/resources
```

