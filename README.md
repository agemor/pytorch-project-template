# PyTorch Project Template

My favorite way to organize a PyTorch research project. It also offers boilerplate for multi-GPU distributed training, mixed-precision training (requires
[NVIDIA apex](https://github.com/NVIDIA/apex) API), and Tensorboard.

## Requirements

I personally prefer using the latest version of everything.

```
pip install -r requirements.txt
```

## Preprocessing

Often, we need to preprocess the downloaded dataset for our own use. `generate_dataset.py` should do the job of processing 'raw' datasets into 'tidy' ones. The
processed result should be stored under the `data/dataset` directory.

```shell script
python generate_dataset.py --dataset 'cifar10' --source 'downloades/raw_cifar10'
```

## Training

To start training, run ``train.py``.

```shell script
python train.py --dataset 'cifar10' --epoch 10 --batch_size 128
```

Or, the `.conf` files can be used instead of passing lengthy command-line arguments. `.conf` files should be stored under the `data/params` directory.

```shell script
python train.py @data/params/cifar10.conf
```

Upon completion of each epoch, the model checkpoint will be stored at `data/model_checkpoints/[label]`.
Also, [tensorboard](https://www.tensorflow.org/tensorboard)
can be used to monitor the training process.

## Evaluation

Once training is done, run `evaluate.py` to start the evaluation.

```shell script
python evaluate.py @data/params/cifar10.conf
```

## License

Public domain

