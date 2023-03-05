import os
import sys
import torch
import time
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.distributed as dist
import numpy as np
from dicee.abstracts import AbstractTrainer
from dicee.static_funcs_training import efficient_zero_grad
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import sys
from core.static_funcs_training import efficient_zero_grad
import platform
import GPUtil
import pykeen

# DDP with gradiant accumulation https://gist.github.com/mcarilli/bf013d2d2f4b4dd21ade30c9b52d5e2e


BACKEND_GLOO = "gloo"
BACKEND_NCCL = "nccl"


def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device)/(1024*1024)}MB ")


class TorchDDPTrainer(AbstractTrainer):
    """
     A Trainer based on torch.nn.parallel.DistributedDataParallel

     Arguments
    ----------
    train_set_idx
        Indexed triples for the training.
    entity_idxs
        mapping.
    relation_idxs
        mapping.
    form
        ?
    store
         ?
    label_smoothing_rate
         Using hard targets (0,1) drives weights to infinity.
         An outlier produces enormous gradients.

    Returns
    -------
    torch.utils.data.Dataset
    """

    def __init__(self, args, callbacks):
        super().__init__(args, callbacks)

    def fit(self, *args, **kwargs):
        """Train model"""
        assert len(args) == 1
        (model,) = args
        # (1) Fit start.
        self.on_fit_start(self, model)
        # nodes * gpus
        world_size = self.attributes.num_nodes * torch.cuda.device_count()

        # @TODO auto batch size implementation
        oom = True
        batch_size = self.attributes.batch_size  # batch size to increase
        initial_num_epochs = self.attributes.num_epochs
        num_of_tries = 1
        if "train_dataloaders" not in kwargs:
            # get the length of dataset from pykeen
            kwargs["train_dataloaders"] = model.train_dataloaders
            # size_of_train_data = len(model.dataset.training.triples)
        
        size_of_train_data = len(kwargs["train_dataloaders"].dataset)

        while oom:
            try:
                if (
                    num_of_tries == initial_num_epochs
                    or self.attributes.batch_size > size_of_train_data
                ):
                    # no rest of epochs are left to test the batch_size or
                    # bacth_size is already bigger than train dataset 
                    oom = False

                self.attributes.num_epochs = 1  # only run one epoch
                mp.spawn(
                    fn=distributed_training,
                    args=(
                        world_size,
                        model,
                        kwargs["train_dataloaders"],
                        self.callbacks,
                        self.attributes,
                    ),
                    nprocs=world_size,
                    join=True,
                )
                model.load_state_dict(
                    torch.load("model.pt", map_location=torch.device("cpu"))
                )
                os.remove("model.pt")
                self.attributes.batch_size += batch_size  # increase the batch size
                num_of_tries += 1
            except RuntimeError:
                # the batch_size here can be sure to fit in the memory of GPU
                # @TODO: there may be another method to find the batach size after oom
                self.attributes.batch_size += (
                    -batch_size
                )  # reset the batch size to the last add
                self.attributes = (
                    initial_num_epochs - num_of_tries
                )  # run the rest epochs
                mp.spawn(
                    fn=distributed_training,
                    args=(
                        world_size,
                        model,
                        kwargs["train_dataloaders"],
                        self.callbacks,
                        self.attributes,
                    ),
                    nprocs=world_size,
                    join=True,
                )
                model.load_state_dict(
                    torch.load("model.pt", map_location=torch.device("cpu"))
                )
                os.remove("model.pt")
                oom = False

        self.on_fit_end(self, model)


def distributed_training(
    rank: int, world_size, model, train_dataset_loader, callbacks,  attrubutes
    
):
    """
    distributed_training is called as the entrypoint of the spawned process.
    This function must be defined at the top level of a module so it can be pickled and spawned.
    This is a requirement imposed by multiprocessing.
    args: dictionary
    callbacks:list of callback objects
    The function is called as ``fn(i, *args)``, where ``i`` is the process index and ``args`` is the passed through tuple of arguments.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "1234"
    # oom = True
    if platform.system().lower() == "windows":
        backend = BACKEND_GLOO
    else:
        backend = BACKEND_NCCL

    # for test purpose(not sure if this simulation is correct???)
    # GPU memory managed by the caching allocator can now only allocate 0.01*total_memory memory
    # torch.cuda.set_per_process_memory_fraction(0.01)

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    # train_dataset_loader = DataLoader(
    #         dataset=train_dataset_loader.dataset,
    #         num_workers=attrubutes.num_core,
    #         pin_memory=True,
    #         # disable automatic batching
    #         batch_size=None,
    #         batch_sampler=None,
    #         shuffle=False,
    #         sampler=None,
    #     )



    # (1) Create DATA LOADER.
    # train_dataset_loader.sampler=torch.utils.data.distributed.DistributedSampler
    
    # collate_fn of the model of pykeen is None
    collate_fn=None
    if isinstance(model,pykeen.contrib.lightning.LitModule):
        collate_fn = model.train_dataloaders.dataset.get_collator()
    else:
        collate_fn = train_dataset_loader.dataset.collate_fn

    train_dataset_loader = DataLoader(
        train_dataset_loader.dataset,
        batch_size=attrubutes.batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=attrubutes.num_core,
        persistent_workers=True,
        collate_fn=collate_fn,
        sampler=torch.utils.data.distributed.DistributedSampler(
            train_dataset_loader.dataset
        ),
    )

    # (2) Initialize OPTIMIZER.
    optimizer = model.configure_optimizers()
    # (3) Create a static DDB Trainer.
    trainer = Trainer(
        model, train_dataset_loader, optimizer, rank, callbacks, attrubutes.num_epochs
    )
    # total_size=0
    # data_batch = next(iter(train_dataset_loader))
    # for data in data_batch:
    #     total_size += data.element_size() * data.nelement()
    # print(f"Total size of data in batch: {total_size} bytes")
    # print(f"{torch.cuda.memory_summary(0)}")
    trainer.train()

    if rank == 0:
        trainer.model.loss_history = trainer.loss_history
        torch.save(trainer.model.module.state_dict(), "model.pt")
    dist.destroy_process_group()

class DDPTrainer:
    def __init__(self,
                 model: torch.nn.Module,
                 train_dataset_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 gpu_id: int, callbacks, num_epochs) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_dataset_loader = train_dataset_loader
        self.loss_func = self.model.loss
        self.optimizer = optimizer
        self.callbacks = callbacks
        self.model = DDP(model, device_ids=[gpu_id])
        self.num_epochs = num_epochs
        print_peak_memory("Max memory allocated after creating DDP:", gpu_id)
        # print('GPU:{self.gpu_id')
        print(f"GPU:{torch.cuda.current_device()}")
        print(self.model)
        print(self.optimizer)
        print(
            f"NumOfDataPoints:{len(self.train_dataset_loader.dataset)} | NumOfEpochs:{self.num_epochs} | LearningRate:{self.model.module.learning_rate} | BatchSize:{self.train_dataset_loader.batch_size} | EpochBatchsize:{len(self.train_dataset_loader)}"
        )
        # print(f'.........max memory of GPU: {torch.cuda.get_device_properties(0).total_memory}')
        # print(torch.cuda.get_device_properties('cuda:0')) # 3221094400/(1024*1024) = 3071MB
        # for test purpose, manually decrease the memory of GPU

        self.loss_history = []

    def _run_batch(self, source, targets):

        # (1) Zero the gradients.
        # self.optimizer.zero_grad()
        efficient_zero_grad(self.model)
        output = self.model(source)
        loss = self.loss_func(output, targets)
        batch_loss = loss.item()
        loss.backward()
        self.optimizer.step()

        # @TODO: Tips to decrease mem usage
        #  https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        torch.cuda.empty_cache()

        return batch_loss

    def extract_input_outputs(self, z: list):
        if len(z) == 2:
            x_batch, y_batch = z
            return x_batch.to(self.gpu_id), y_batch.to(self.gpu_id)
        elif len(z) == 3:
            (
                x_batch,
                y_idx_batch,
                y_batch,
            ) = z
            x_batch, y_idx_batch, y_batch = (
                x_batch.to(self.gpu_id),
                y_idx_batch.to(self.gpu_id),
                y_batch.to(self.gpu_id),
            )
            return (x_batch, y_idx_batch), y_batch
        else:
            print(len(batch))
            raise ValueError("Unexpected batch shape..")

    def _run_epoch(self, epoch):
        
        self.train_dataset_loader.sampler.set_epoch(epoch)
        epoch_loss = 0
        i = 0
        construct_mini_batch_time = None
        for i, z in enumerate(self.train_dataset_loader):
            source, targets = self.extract_input_outputs(z)
            start_time = time.time()

            if construct_mini_batch_time:
                construct_mini_batch_time = start_time - construct_mini_batch_time
            batch_loss = self._run_batch(source, targets)
            epoch_loss += batch_loss
            if self.gpu_id == 0:
                if construct_mini_batch_time:
                    print(
                        f"Epoch:{epoch + 1} | Batch:{i + 1} | Loss:{batch_loss} |ForwardBackwardUpdate:{(time.time() - start_time):.2f}sec | BatchConst.:{construct_mini_batch_time:.2f}sec"
                    )
                else:
                    print(
                        f"Epoch:{epoch + 1} | Batch:{i + 1} | Loss:{batch_loss} |ForwardBackwardUpdate:{(time.time() - start_time):.2f}secs"
                    )
            construct_mini_batch_time = time.time()
        print(
            f"maximal alocated memory so far: {torch.cuda.memory_allocated(0)//1e6}MB"
        )
        print(f'batch size to be tried currently: {self.train_dataset_loader.batch_size}')
        return epoch_loss / (i + 1)

    def train(self):

        for epoch in range(self.num_epochs):
            start_time = time.time()
            epoch_loss = self._run_epoch(epoch)
            GPUtil.showUtilization()
            if self.gpu_id == 0:
                print(
                    f"Epoch:{epoch + 1} | Loss:{epoch_loss:.8f} | Runtime:{(time.time() - start_time) / 60:.3f}mins"
                )
                self.model.module.loss_history.append(epoch_loss)
                for c in self.callbacks:
                    c.on_train_epoch_end(None, self.model.module)
