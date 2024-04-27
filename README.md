# InfoVGAE-SL
The source code for InfoVGAE-SL model submitted for TKDD. We will publish this code and datasets after acceptance.

## Dataset
We uploaded the pre-processed datasets with smaller size, due to the file size limits of supplementary material.
The datasets are located in `dataset/election`, `dataset/eurovision`, `dataset/war`, and `dataset/bill`.
Please run the following to unzip the datasets:

```
cd dataset/bill
unzip bmap2.pkl.zip
cd ../bill
unzip data_80_115.pkl.zip
cd ../eurovision
unzip data.csv.zip
cd ../election
unzip data.csv.zip
cd ../war
unzip data.csv.zip
cd ../..
```

## Training

To run InfoVGAE-SL on Eurovision dataset:

```
python3 main.py --config_name InfoVGAE-SL_eurovision_3D
```

To run InfoVGAE-SL on Election dataset:

```
python3 main.py --config_name InfoVGAE-SL_election_3D
```

To run InfoVGAE-SL on Russia Ukraine War 2022 dataset:
```
python3 main.py --config_name InfoVGAE-SL_war_3D
```


To run InfoVGAE-SL on Voteview 105th Congress dataset:
```
python3 main.py --config_name InfoVGAE-SL_bill_3D
```

## Other arguments for training:

> General

`--use_cuda`: training with GPU

`--epochs`: iterations for training

`--learning_rate`: learning rate for training

`--device`: which gpu to use. empty for cpu.

`--num_process`: num process for pandas processing

> Data

`--data_path`: csv path for data file

`--stopword_path`: stopword path for text parsing

`--kthreshold`: tweet count threshold to filter not popular tweets.

`--uthreshold`: user count threshold to filter not popular users.

> For InfoVGAE-SL model

`--hidden1_dim`: the latent space dimension of first layer

`--hidden2_dim`: the latent space dimension of target layer

`--beta`: the parameter for sparsity regularization

`--pos_weight_lambda`: the parameter for local observation compensation

> Result

`--output_path` path to save the result
