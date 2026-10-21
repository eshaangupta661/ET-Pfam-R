import os
import torch as tr
import numpy as np
from torch import nn
from tqdm import tqdm
from src.basemodel import BaseModel
from src.dataset import PFamDataset
from src.utils import load_config, predict
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class EnsembleModel(nn.Module):
    def __init__(self, models_path, config, voting_strategy, ensemble_weights_path=None, 
                 exp_name=None):
        super(EnsembleModel, self).__init__()

        # Load model paths
        model_dirs = [os.path.join(models_path, d) for d in os.listdir(models_path) if os.path.isdir(os.path.join(models_path, d))]
        
        self.emb_path = config['emb_path']
        self.data_path = config['data_path']
        self.voting_strategy = voting_strategy
        self.path = models_path
        self.weights_file = None

        # Load categories
        cat_path = os.path.join(self.data_path, "categories.txt")
        with open(cat_path, 'r') as f:
            categories = [item.strip() for item in f]
        self.categories = categories

        # Initialize the ensemble of models
        self.models = nn.ModuleList()
        self.model_configs = []

        # Sort model directories to ensure consistent order
        model_dirs.sort()

        # Load each model's configuration and weights
        for model_dir in model_dirs:
            # Load the config.json to get the parameters
            config_path = os.path.join(model_dir, 'config.json')
            config = load_config(config_path)

            lr = config['lr']
            batch_size = config['batch_size']
            win_len = config['window_len']
            device = config['device']

            self.model_configs.append({
                'lr': lr,
                'batch_size': batch_size,
                'win_len': win_len
            })

            # Load the model weights
            weights_path = os.path.join(model_dir, 'weights.pk')
            print("loading weights from", model_dir)
            model = BaseModel(len(categories), lr=lr, device=device)
            model.load_state_dict(tr.load(weights_path))
            model.eval()
            self.models.append(model)
        
        # Initialize model weights based on voting strategy
        if self.voting_strategy in ['weighted_model', 'weighted_families']:
            weights, weights_file = self._initialize_weights(model_dirs, 
                                                             ensemble_weights_path,
                                                             exp_name)
            self.weights_file = weights_file
            if self.voting_strategy == 'weighted_model':
                self.model_weights = weights
            elif self.voting_strategy == 'weighted_families':
                self.family_weights = weights

    def fit(self):
        if self.voting_strategy in ['weighted_model', 'weighted_families']:
            # Collect predictions from each model
            all_preds = []
            for i, net in enumerate(self.models):
                config = self.model_configs[i]
                dev_data = PFamDataset(
                    f"{self.data_path}dev.csv",
                    self.emb_path,
                    self.categories,
                    win_len=config['win_len'],
                    is_training=False
                )
                dev_loader = tr.utils.data.DataLoader(dev_data, batch_size=config['batch_size'], num_workers=config.get("nworkers", 1))

                with tr.no_grad():
                    _, _, pred, ref, *_ = net.pred(dev_loader)
                    all_preds.append(pred)
            stacked_preds = tr.stack(all_preds)

        if self.voting_strategy == 'weighted_model':
            criterion = nn.CrossEntropyLoss()
            optimizer = tr.optim.Adam([self.model_weights], lr=0.01)

            for epoch in tqdm(range(500), desc="Epochs"):
                pred_avg = tr.sum(stacked_preds * self.model_weights.view(-1, 1, 1), dim=0)
                loss = criterion(pred_avg, tr.argmax(ref, dim=1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            tr.save(self.model_weights.detach().cpu(), self.weights_file)
            print(f"Saved model weights to {self.weights_file}")

        elif self.voting_strategy == 'weighted_families':
            criterion = nn.CrossEntropyLoss()
            optimizer = tr.optim.Adam([self.family_weights], lr=0.01)

            for epoch in tqdm(range(500), desc="Epochs"):
                pred_avg = tr.sum(stacked_preds * self.family_weights.view(len(self.models), 1, len(self.categories)), dim=0)
                loss = criterion(pred_avg, tr.argmax(ref, dim=1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            tr.save(self.family_weights.detach().cpu(), self.weights_file)
            print(f"Saved family weights to {self.weights_file}")

    def forward(self, batch):
        if isinstance(batch, (tr.Tensor, np.ndarray)):
            # Direct batch tensor forward pass: [batch_size, 1280, L]
            if isinstance(batch, np.ndarray):
                batch = tr.from_numpy(batch).float()
            all_preds = []
            for net in self.models:
                all_preds.append(net(batch))
            stacked = tr.stack(all_preds)
            preds, _ = self._combine_ensemble_predictions(stacked)
            return preds
        elif isinstance(batch, str):
            pred, _ = self.pred(partition=batch)
            return pred
        else:
            pred, _ = self.pred(partition='test')
            return pred

    def pred(self, partition='test'):
        # Predicts using the centered window method on the specified dataset.
        all_preds = []

        for i, net in enumerate(self.models):
            # Load the model's configuration and dataset
            config = self.model_configs[i]
            test_data = PFamDataset(
                f"{self.data_path}{partition}.csv",
                self.emb_path,
                self.categories,
                win_len=config['win_len'],
                is_training=False
            )
            test_loader = tr.utils.data.DataLoader(test_data, 
                                                   batch_size=config['batch_size'], 
                                                   num_workers=config.get("nworkers", 1))
            net_preds = []

            # Predict using the model
            with tr.no_grad():
                test_loss, test_errate, pred, *_ = net.pred(test_loader)
                net_preds.append(pred)
            print(f"win_len = {config['win_len']} - lr = {config['lr']} - test_loss {test_loss:.5f} - test_errate {test_errate:.5f}")
            net_preds = tr.cat(net_preds)
            all_preds.append(net_preds)

        stacked_preds = tr.stack(all_preds)
        preds, preds_bin = self._combine_ensemble_predictions(stacked_preds)
        return preds, preds_bin

    def pred_sliding(self, emb, step=4, use_softmax=False):
        all_preds = []
        all_centers = []
        
        for i, net in enumerate(self.models):
            config = self.model_configs[i]
            net_preds = []
            centers, pred = predict(net, emb, config['win_len'], 
                                    use_softmax=use_softmax, step=step)
            net_preds.append(pred)
            all_preds.append(tr.cat(net_preds))
            all_centers.append(centers)

        for c in all_centers:
            if not np.allclose(c, all_centers[0]):
                raise ValueError("Model predictions have misaligned window centers.")

        stacked_preds = tr.stack(all_preds) 
        preds, preds_bin = self._combine_ensemble_predictions(stacked_preds)
        return centers, preds.cpu().detach()

    def _initialize_weights(self, model_dirs, ensemble_weights_path, exp_name=None):
        """Initializes the weights for the ensemble based on the voting strategy."""
        # Define the file name based on the voting strategy and experiment name (if provided)
        if exp_name:
            file_name = f"{self.voting_strategy}_ensemble_{exp_name}.pt"
        else:
            file_name = f"{self.voting_strategy}_ensemble.pt"

        # If ensemble_weights_path is provided, use it to load weights
        if ensemble_weights_path:
            weights_file = f"{ensemble_weights_path}{file_name}"
        else:
            weights_file = f"{self.path}{file_name}"

        if self.voting_strategy == 'weighted_model':
            if ensemble_weights_path and os.path.exists(weights_file):
                weights = nn.Parameter(tr.load(weights_file))
                print(f"Loaded model weights from {weights_file}")
            else:
                weights = nn.Parameter(tr.rand(len(model_dirs)))
                if ensemble_weights_path:
                    print(f"Warning: {weights_file} not found, using random init.")
            return weights, weights_file

        elif self.voting_strategy == 'weighted_families':
            if ensemble_weights_path and os.path.exists(weights_file):
                weights = nn.Parameter(tr.load(weights_file))
                print(f"Loaded family weights from {weights_file}")
            else:
                weights = nn.Parameter(tr.rand(len(model_dirs), len(self.categories)))
                if ensemble_weights_path:
                    print(f"Warning: {weights_file} not found, using random init.")
            return weights, weights_file
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")

    def _combine_ensemble_predictions(self, stacked_preds):
        """ Combines predictions from the ensemble models based on the voting strategy."""
        if self.voting_strategy == 'score_voting':
            pred = tr.mean(stacked_preds, dim=0)
            pred_bin = tr.argmax(pred, dim=1)

        elif self.voting_strategy == 'weighted_model':
            pred = tr.sum(stacked_preds * self.model_weights.view(-1, 1, 1), dim=0)
            pred_bin = tr.argmax(pred, dim=1)

        elif self.voting_strategy == 'weighted_families':
            pred = tr.sum(stacked_preds * self.family_weights.view(len(self.models), 1, len(self.categories)), dim=0)
            pred_bin = tr.argmax(pred, dim=1)

        elif self.voting_strategy == 'simple_voting':
            pred_classes = tr.mode(tr.argmax(stacked_preds, dim=2), dim=0)[0]
            pred = tr.nn.functional.one_hot(pred_classes, num_classes=len(self.categories)).float()
            pred_bin = tr.argmax(pred, dim=1)

        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")
        return pred, pred_bin