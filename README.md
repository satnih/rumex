## This repo contains the code to reproduce the results, figures and tables in the paper <>

- `prepare_data.ipynb`: The notebook contains code used to prepare the data for training and testing the model. It extract image patches from orthomosaic and stores them in `imagenet` format.    
- `lr_finder.ipynb`: Finds the optimal lr for the models. 
- `main_10.py` and `trainer_10m.py`: Scripts to train models on 10m data with 5-fold CV. 
- `main_15.py` and `trainer_15m`: Script to fine-tune 10m model with 15m data. 
- `process_results.ipynb`: script to generate the tables and figures in the paper.
