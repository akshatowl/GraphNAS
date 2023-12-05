# Improvised Graph Neural Architecture Search Through Policy Optimization Methods
# Try it yourself!
## Setting up the codebase
Clone this repository on the local machine
```
git clone https://github.com/akshatowl/GraphNAS.git
```

After cloning, make sure all the dependencies are present. The suggested dependency versions are outdated and have depreciated functions. We installed every dependency around `CUDA 12` instead, as per our GPU requirements.

If using CUDA 12, we suggest running this command to make sure the correct version of pytorch and torchvision are present compatible with CUDA 12:
```
pip3 install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
or, if you are using a conda environment:

```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
Requirements can be installed manually or by running the requirements.txt script:
```
pip install -r requirements.txt
```
or

```
pip3 install dgl scipy torch_scatter torch-cluster torch-sparse torch_geometric numpy hyperopt scikit_learn requests
```
## Running the code
This is the main segment where reinforcement learning and our implementation will be reflected.
The reinforcement learning algorithm can be specified using the `--rlalgo` tag, the default will be the baseline algorithm with the original discount function.
This statement is run for designing an entire graph neural architecture.
Running the baseline algorithm in the existing paper.

```
cd ~/GraphNAS
python -m graphnas.main --dataset Citeseer
```

Running the training with the Proximal Policy Optimization (PPO) Algorithm:

```
cd ~/GraphNAS
python -m graphnas.main --dataset Citeseer --rlalgo ppo
```

Running the training with the Trust Region Policy Optimization (TRPO) Algorithm:

```
cd ~/GraphNAS
python -m graphnas.main --dataset Citeseer --rlalgo trpo
```
After the model is constructed it will be saved in a folder directory within the project dependencies `CiteSeer` with a .pth extension.

Identify the model that is generated based on the time of generation and then copy the path of the model file.

In the `semi` directory in `GraphNAS/semi/eval_designed_gnn.py` enter the correct path of the model in the `customer_architecture_path` string.  
```
custom_architecture_path = "/path_to_model/model.pth"

```

Then run the evaluation function to evaluate the training accuracy of the custom model with the `Citeseer` dataset.
```
cd ~/GraphNAS
```
Run the command:
```
python -m eval_scripts.semi.eval_designed_gnn

```
## Experimentation test accuracy results
PPO and TRPO algorithms have discount generation using a 4 order butterworth filter with a cutoff frequency of 0.1 Hz. 
The results are generated for a Citeseer academic paper dataset. 
| Techniques | Test Accuracy |
|-------------------|---------------|
| GraphNAS(Baseline)| 73.7+/-0.2    |
| APPNP             | 71.8+/-0.4    |
| simple-NAS        | 71.7+/-0.6    |
| GNAS with PPO     | 73.64+/-0.21  |
| GNAS with TRPO    | 73.64+/-0.21  |
 
    

#### Acknowledgements
This repo is modified based on [DGL](https://github.com/dmlc/dgl) and [PYG](https://github.com/rusty1s/pytorch_geometric).


