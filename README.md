# GraphNAS
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

In the semi directory in GraphNAS/semi/eval_designed_gnn.py enter the correct path of the model in the `customer_architecture_path` string.  
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


Model| Cora | Citeseer | Pubmed
|-|-|-|-|
GCN	| 81.5+/-0.4 | 70.9+/-0.5   | 79.0+/-0.4  
SGC	|  81.0+/-0.0 |   71.9+/-0.1   |  78.9+/-0.0   
GAT	|  83.0+/-0.7  |  72.5+/-0.7   | 79.0+/-0.3    
LGCN	|  83.3+/-0.5  | 73.0+/-0.6	|  79.5+/-0.2   
DGCN	|  82.0+/-0.2  | 72.2+/-0.3	|  78.6+/-0.1   
ARMA	|  82.8+/-0.6  | 72.3+/-1.1	|  78.8+/-0.3   
APPNP   |  83.3+/-0.6  | 71.8+/-0.4	|  80.2+/-0.2   
simple-NAS |  81.4+/-0.6  |  71.7+/-0.6	|  79.5+/-0.5  
GraphNAS | **84.3+/-0.4**  | **73.7+/-0.2**	| **80.6+/-0.2**  
   	 
Supervised node classification w.r.t. accuracy    

Model| Cora | Citeseer | Pubmed  
|-|-|-|-|
GCN	| 90.2+/-0.0  | 80.0+/-0.3   | 87.8+/-0.2  
SGC	| 88.8+/-0.0 |  80.6+/-0.0   |   86.5+/-0.1  
GAT	|  89.5+/-0.3  |  78.6+/-0.3	|  86.5+/-0.6   
LGCN	| 88.7+/-0.5  | 79.2+/-0.4 	|  OOM    
DGCN	|  88.4+/-0.2  |  78.0+/-0.2	|  88.0+/-0.9    
ARMA	|  89.8+/-0.1  |  79.9+/-0.6	|  88.1+/-0.2    
APPNP	| 90.4+/-0.2  | 79.2+/-0.4 	| 87.4+/-0.3    
random-NAS | 90.0+/-0.3   |  81.1+/-0.3	| 90.7+/-0.6    
simple-NAS | 90.1+/-0.3  |  79.6+/-0.5	|  88.5+/-0.2  
GraphNAS | **90.6+/-0.3**   |  **81.3+/-0.4**   | **91.3+/-0.3**    
    

#### Acknowledgements
This repo is modified based on [DGL](https://github.com/dmlc/dgl) and [PYG](https://github.com/rusty1s/pytorch_geometric).
The base repository used for this was:
