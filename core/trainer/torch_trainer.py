import torch
from typing import Tuple
from core.abstracts import AbstractTrainer
from core.custom_opt.sls import Sls
from core.custom_opt.adam_sls import AdamSLS
import time
import os
import psutil
import pykeen

class TorchTrainer(AbstractTrainer):
    """
        TorchTrainer for using single GPU or multi CPUs on a single node

        Arguments
       ----------
       args: ?

       callbacks: list of Abstract callback instances

   """

    def __init__(self, args, callbacks):
        super().__init__(args, callbacks)
        self.use_closure = None
        self.loss_function = None
        self.optimizer = None
        self.model = None
        self.is_global_zero = True
        self.train_dataloaders = None
        torch.manual_seed(self.attributes.seed_for_computation)
        torch.cuda.manual_seed_all(self.attributes.seed_for_computation)
        if self.attributes.gpus:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'
        # https://psutil.readthedocs.io/en/latest/#psutil.Process
        self.process = psutil.Process(os.getpid())

    def _run_batch(self, i: int, x_batch, y_batch) -> float:
        """
            Forward anc Backward according to a mini-batch

            Arguments
           ----------
           i : index of a batch
           x_batch: torch.Tensor on selected device
           y_batch: torch.Tensor on selected device
           Returns
           -------
           batch loss (float)
       """
        if self.attributes.gradient_accumulation_steps > 1:
            # (1) Update parameters every gradient_accumulation_steps mini-batch.
            if i % self.attributes.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad(set_to_none=True)
        else:
            # (2) Do not accumulate gradient, zero the gradients per batch.
            self.optimizer.zero_grad(set_to_none=True)
        # (3) Loss Forward and Backward w.r.t the batch.
        return self.compute_forward_loss_backward(x_batch, y_batch).item()

    def _run_epoch(self, epoch: int) -> float:
        """
            Iterate over the training dataset

            Arguments
           ----------
           epoch:int
           -------
           average loss over the dataset
       """
        epoch_loss = 0
        i = 0
        construct_mini_batch_time = None
        for i, batch in enumerate(self.train_dataloaders):
            # (1) Extract Input and Outputs and set them on the dice
            x_batch, y_batch = self.extract_input_outputs_set_device(batch)
            start_time = time.time()
            if construct_mini_batch_time:
                construct_mini_batch_time = start_time - construct_mini_batch_time
            # (2) Forward-Backward-Update.
            batch_loss = self._run_batch(i, x_batch, y_batch)
            epoch_loss += batch_loss
            if construct_mini_batch_time:
                print(
                    f"Epoch:{epoch + 1} | Batch:{i + 1} | Loss:{batch_loss:.10f} |ForwardBackwardUpdate:{(time.time() - start_time):.2f}sec | BatchConst.:{construct_mini_batch_time:.2f}sec | Mem. Usage {self.process.memory_info().rss / 1_000_000: .5}MB  avail. {psutil.virtual_memory().percent} %")
            else:
                print(
                    f"Epoch:{epoch + 1} | Batch:{i + 1} | Loss:{batch_loss} |ForwardBackwardUpdate:{(time.time() - start_time):.2f}secs | Mem. Usage {self.process.memory_info().rss / 1_000_000: .5}MB")
            construct_mini_batch_time = time.time()
        return epoch_loss / (i + 1)

    def fit(self, *args, train_dataloaders, **kwargs) -> None:
        """
            Training starts

            Arguments
           ----------
           args:tuple
           (BASEKGE,)
           kwargs:Tuple
               empty dictionary
           Returns
           -------
           batch loss (float)
       """
        assert len(args) == 1
        model, = args
        self.model = model
        self.model.to(self.device)
        self.train_dataloaders = train_dataloaders
        if isinstance(model,pykeen.contrib.lightning.LitModule):
            self.loss_function = model.loss
        else:
            self.loss_function = model.loss_function
        self.optimizer = self.model.configure_optimizers()
        # (1) Start running callbacks
        self.on_fit_start(self, self.model)

        if isinstance(self.optimizer, Sls) or isinstance(self.optimizer, AdamSLS):
            self.use_closure = True
        else:
            self.use_closure = False

        print(
            f'NumOfDataPoints:{len(self.train_dataloaders.dataset)} | NumOfEpochs:{self.attributes.max_epochs} | LearningRate:{self.model.learning_rate} | BatchSize:{self.train_dataloaders.batch_size} | EpochBatchsize:{len(train_dataloaders)}')

        increment_ratio=0
        for epoch in range(self.attributes.max_epochs):
            start_time = time.time()
            # (1)
            avg_epoch_loss = self._run_epoch(epoch)
            print(f"Epoch:{epoch + 1} | Loss:{avg_epoch_loss:.8f} | Runtime:{(time.time() - start_time) / 60:.3f}mins")
            # Autobatch Finder: Increase the batch size at each epoch's end if memory allows
            #             mem=self.process.memory_info().rss
            if increment_ratio>1:
                self.train_dataloaders = torch.utils.data.DataLoader(dataset=self.train_dataloaders.dataset,
                                                                     batch_size=self.train_dataloaders.batch_size +self.train_dataloaders.batch_size,
                                                                     shuffle=True,
                                                                     collate_fn=self.train_dataloaders.dataset.collate_fn,
                                                                     num_workers=self.train_dataloaders.num_workers,
                                                                     persistent_workers=False)

            self.model.loss_history.append(avg_epoch_loss)
            self.on_train_epoch_end(self, self.model)
        self.on_fit_end(self, self.model)

    def compute_forward_loss_backward(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> torch.Tensor:
        """
            Compute forward, loss, backward, and parameter update

            Arguments
           ----------
           x_batch:(torch.Tensor) mini-batch inputs
           y_batch:(torch.Tensor) mini-batch outputs

           Returns
           -------
           batch loss (float)
       """
        if self.use_closure:
            batch_loss = self.optimizer.step(closure=lambda: self.loss_function(self.model(x_batch), y_batch))
            return batch_loss
        else:
            # (1) Forward and Backpropagate the gradient of (3) w.r.t. parameters.
            yhat_batch = self.model(x_batch)
            # (2) Compute the batch loss
            batch_loss = self.loss_function(yhat_batch, y_batch)
            # (3) Backward pass
            batch_loss.backward()
            # (4) Parameter update
            self.optimizer.step()
            return batch_loss

    def extract_input_outputs_set_device(self, batch: list) -> Tuple:
        """
            Construct inputs and outputs from a batch of inputs with outputs From a batch of inputs and put

            Arguments
           ----------
           batch: (list) mini-batch inputs on CPU

           Returns
           -------
           (tuple) mini-batch on select device
       """
        # (1) NegSample: x is a triple, y is a float
        if len(batch) == 2:
            x_batch, y_batch = batch
            return x_batch.to(self.device), y_batch.to(self.device)
        elif len(batch) == 3:
            x_batch, y_idx_batch, y_batch, = batch
            x_batch, y_idx_batch, y_batch = x_batch.to(self.device), y_idx_batch.to(self.device), y_batch.to(
                self.device)
            return (x_batch, y_idx_batch), y_batch
        else:
            print(len(batch))
            raise ValueError('Unexpected batch shape..')
