import optuna
from optuna.exceptions import TrialPruned
import torch
import pytorch_lightning as pl
import wandb
import os
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from models.swin2sr import Swin2SRModule
from training import create_dataset
from configs.biosr_config import get_config
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.directory_setup_utils import get_workdir
from pytorch_lightning.callbacks import EarlyStopping
import json
def objective(trial):
    try:
        config = {
            'data_type': 'biosr',
            'learning_rate':  trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'upscale': 1,
            'in_chans': 1,
            'img_size': (256, 256),
            'window_size': trial.suggest_categorical('window_size', [4, 8, 16]),  # search space for window size
            'img_range': 1.,
            'depths': [
                trial.suggest_categorical('depth_stage_1', [2,3,4,6]),  # number of transformer blocks at stage 1
                trial.suggest_categorical('depth_stage_2',[2,3,4,6])   # number of transformer blocks at stage 2
            ],
            'embed_dim': trial.suggest_categorical('embed_dim', [60, 96, 120, 144]),  # embedding dimensions
            'num_heads': [
                trial.suggest_categorical('num_heads_stage_1', [2,3,4,6,8]),  # number of heads for stage 1
                trial.suggest_categorical('num_heads_stage_2', [2,3,4,6,8])   # number of heads for stage 2
            ],
            'mlp_ratio': trial.suggest_float('mlp_ratio', 1.5, 4.0, step=0.5),  # MLP expansion ratio
            'upsampler': 'pixelshuffledirect',
            'data':{
                'noisy_data': True,
                'poisson_factor': 0,
                'gaussian_factor':3400
            }
        }

        
        print(f"\nTrial {trial.number}")
        print(f"Hyperparameters: {config}\n")   
        # Initialize the model with the selected hyperparameters
        root_dir = "/group/jug/Michele/training/"
        model = Swin2SRModule(config)
        experiment_directory = get_workdir(config,root_dir)[0]
        # save the dictionary to file
        with open(os.path.join(experiment_directory,'config.json'), 'w') as f:
            json.dump(config, f)
            
        print('')
        print('------------------------------------')
        print('Experiment directory', experiment_directory)
        print('------------------------------------')
        config_str = f"{experiment_directory}"
        
        # Optional: Use a W&B logger or any other logger for logging
        wandb_logger = WandbLogger(save_dir=experiment_directory, project="gridsearch_tuning", name=config_str)
        wandb_logger.experiment.config.update(config, allow_val_change=True)
        model_filename = f'swin2sr'
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=50,    # How long to wait after last improvement
            #restore_best_weights=True,  # Automatically handled by PL's checkpoint system
            mode='min')
          # Define ModelCheckpoint callback to save the best model
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            save_top_k=1,
            save_last=True,
            mode='min',
            dirpath=experiment_directory,
            filename=model_filename + '_best',
        )
        checkpoint_callback.CHECKPOINT_NAME_LAST = model_filename + "_last"
        callbacks=[early_stopping, checkpoint_callback]
        
    # Define the Trainer
        trainer = pl.Trainer(
            max_epochs=300,
            logger=wandb_logger,
            log_every_n_steps=1,
            check_val_every_n_epoch=1,
            precision=16,
            enable_progress_bar= True,
            callbacks = callbacks
        )
        
        # config = get_config()
        
        train_loader, val_loader, _ = create_dataset(config, 
                    transform=True, 
                    noisy_data=config['data']['noisy_data'],
                    noisy_factor= config['data']['poisson_factor'],
                    gaus_factor=config['data']['gaussian_factor'],
                    patch_size=256)

        # Perform training
        trainer.fit(model, train_loader, val_loader)
        
        val_loss = trainer.callback_metrics["val_loss"].item() 
        
        print(f"\nTrial {trial.number} | Validation Loss = {val_loss:.4f}")
        
        
        # Save model
        saving_dir = experiment_directory
        model_filepath = os.path.join(saving_dir, model_filename)
        torch.save(model.state_dict(), model_filepath)
        wandb.finish()
        
        return val_loss   
    except RuntimeError as e:
        print(f"Run Time Error {trial.number}. Skipping...")
        # Clear GPU cache and prune the trial
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise TrialPruned()
        

from tqdm import tqdm

class TqdmCallback(object):
    def __init__(self, n_trials):
        self.pbar = tqdm(total=n_trials)

    def __call__(self, study, trial):
        self.pbar.update(1)

if __name__ == '__main__': 
    study = optuna.create_study(direction="minimize") 
    study.optimize(objective, n_trials=500, callbacks=[TqdmCallback(500)])
    print("Best hyperparameters: ", study.best_params)

    