# DASNet

## Environment Setup
This code has been tested on Ubuntu18.04, Python3.6, Pytorch0.4.1, CUDA10.1, RTX 2080Ti GPUs
1. Clone the respository  
  git clone https://github.com/jwma0725/DASnet.git && cd DASnet
2. Setup python environment  
  pip install -r requirements.txt  
  bash make.sh  

## Test tracker
1. Before testing again, you need to create a directory /data and put the downloaded test data in it.  
2. python -u ./tools/test.py --snapshot model.pth --dataset VOT2016 --config config16.json  
The testing results will in current directory(test/VOT2016/model_name/)

## Train tracker
### Train base model
1. modify training set path in ./experiment/DASnet/config.json
2. python -u ./tools/train_dasnet.py --config ./experiment/DASnet/config.json  

### Train refine model
1. choose the best test base model as the training model
2. modify training set path in ./experiment/DASnet_sharp/config.json
3. python ./tools/train_dasnet_sharp.py --config ./experiment/DASnet/config.json --pretrain ./snapshot/best_model.pth
