## This repo contains the code to reproduce the results, figures and tables in the paper <>

- `prepare_data.ipynb`: the notebook contains code used to prepare the data for training and testing the model. 
    - Extract patches from orthomosaic and store them in `imagenet` format.
    - Augment training data with blurred version of the patches. 
- `lr_finder.ipynb`: Finds the optimal lr for the models. 
- `main_10.py` and `trainer_10m.py`: Scripts to train imagenet-pretrained on 10m data. The 'result' of running this file are the trained models `model_name-10.pth` will be stored
- `main_15.py` and `trainer_10m`: Script to fine-tune an 10m-pretrained model with 15m data. 
- `process_results.ipynb`: script to generate the tables and figures in the paper.
