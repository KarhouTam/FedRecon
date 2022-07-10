# Federated Reconstruction: Partially Local Federated Learning [[Paper]](https://proceedings.neurips.cc/paper/2021/hash/5d44a2b0d85aa1a4dd3f218be6422c66-Abstract.html)

PyTorch Implementation of FedRecon

For simulating the Non-I.I.D scenario, the dataset is split according to [Dirichlet distribution($\alpha$)](https://arxiv.org/abs/1909.06335) by default. Furthermore, I offer another splitting method: allocating data that belong to a random subset of labels to each client.

I choose the more regular machine learning task for demonstrating algorithms, which is image classification instead of matrix factorization in the paper. And the experiment is just a toy demo, so please take it easy. 🤣

## Requirements
```
path~=16.4.0
numpy~=1.21.2
torch~=1.10.2
fedlab~=1.1.4
torchvision~=0.11.3
rich~=12.2.0
```

```
pip install -r requirements.txt
```

## Preprocess dataset

MNIST and CIFAR-10 is prepared.🌟

```
python data/preprocess.py
```

The way of preprocessing is adjustable, more details in `data/preprocess.py`.

And all splitting methods details are in `data/partition`



## Run the experiment

Before run the experiment, please make sure that the dataset is downloaded and preprocessed already.

It’s so simple.🤪 

```
python main.py
```



## Hyperparameters

`--global_epochs`: Num of communication rounds. Default: `20`

`--pers_epochs`: Num of local updating rounds. Default: `1`

`--recon_epochs`: Num of local reconstruction rounds. Default: `1`

`--pers_lr`: Learning rate used in client’s local updating phase. Default: `1e-2`

`--recon_lr`: Learning rate used in client’s reconstruction phase. Default:`1e-2`

`--server_lr`: Learning rate used in server’s aggregation phase. Default: `1.0`

`--dataset`: Name of experiment dataset. Default: `mnist`

`--client_num_per_round`: Num of clients that participating training at each communication round. Default: `5`

`--no_split`: Non-zero value for considering the whole client private dataset as the support set and the query set. Default: `0`

`--gpu`: Non-zero value for using CUDA; `0` for using CPU. Default: `1`

`--batch_size`: Batch size of client local dataset. Default: `32`.

`--eval_while_training`: Non-zero value for performing evaluation while in training phase. Default: `1`

`--valset_ratio`: Percentage of validation set in client local dataset. Default: `0.1`

`--seed`: Random seed for init model parameters and selected clients. Default: `17`
