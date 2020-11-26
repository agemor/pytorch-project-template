import os
import time
import argparse

import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.tensorboard as tb
import torch.cuda.amp as amp

import torchvision

from tqdm import tqdm

import utils
from model import DummyModel


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

    # Training environment configuration
    parser.add_argument('--num_gpus', type=int, default=2, help="Number of GPUs to use")
    parser.add_argument('--fp16', action='store_true', default=False,
                        help="Mixed precision training (requires NVIDIA apex)")
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help="Deterministic mode for reproducibility")
    parser.add_argument('--random_seed', type=int, default=7343, help="Random seed (optional)")
    parser.add_argument('--master_addr', type=str, default='localhost', help="Master address for distributed training")
    parser.add_argument('--master_port', type=str, default='7343', help="Master port for distributed training")

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=20, help="Number of total epochs to train")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size per GPU/CPU for training")
    parser.add_argument('--learning_rate', type=float, default=5e-5, help="Initial learning rate for optimizer")

    # Dataset and trained model (for resume training)
    parser.add_argument('--dataset', type=str, help="Name of the dataset to use")
    parser.add_argument('--model_checkpoint', type=str, help="Path to the model checkpoints (optional)")

    # Misc.
    parser.add_argument('--label', type=str, default='', help="Label of the run (used for the checkpoint file names)")
    parser.add_argument('--write_summary', action='store_true', default=False, help="Tensorboard training summary")

    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    mp.spawn(fn=init_process, nprocs=args.num_gpus, args=(args,), join=True)


def init_process(gpu, args):
    args.gpu = gpu

    # Init distributed processing
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.num_gpus, rank=args.gpu)
    torch.cuda.set_device(args.gpu)

    if args.deterministic:
        utils.set_deterministic(args.random_seed)

    run(args)


def run(args):
    # model

    model = DummyModel()

    rgb_mean = (0.4914, 0.4822, 0.4465)
    rgb_std = (0.2023, 0.1994, 0.2010)

    dataset_train = torchvision.datasets.CIFAR10('./data/datasets', train=True, download=True,
                                                 transform=torchvision.transforms.Compose([
                                                     torchvision.transforms.RandomCrop(32, padding=4),
                                                     torchvision.transforms.RandomHorizontalFlip(),
                                                     torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize(rgb_mean, rgb_std),
                                                 ]))
    dataset_test = torchvision.datasets.CIFAR10('./data/datasets', train=False, download=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(rgb_mean, rgb_std),
                                                ]))

    trainer = Trainer(args, model)

    if args.model_checkpoint is not None:
        trainer.load_checkpoint(args.model_checkpoint)

    trainer.train(dataset_train, dataset_test)


class Trainer:

    def __init__(self, args, model):
        self.args = args
        self.model = model

        self.is_primary = args.gpu == 0
        self.model = self.model.cuda(args.gpu)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.args.gpu])

        # Mixed-precision training
        if args.fp16:
            self.scaler = amp.GradScaler()

        # TensorBoard
        if args.write_summary and self.is_primary:
            self.writer = tb.SummaryWriter(comment=args.label, flush_secs=1)

        self.global_steps = 0

    def train(self, dataset_train, dataset_test):

        sampler_train = data.distributed.DistributedSampler(dataset_train, self.args.num_gpus, self.args.gpu,
                                                            shuffle=True)
        sampler_test = data.distributed.DistributedSampler(dataset_test, self.args.num_gpus, self.args.gpu)

        loader_train = data.DataLoader(dataset_train, self.args.batch_size, sampler=sampler_train)
        loader_test = data.DataLoader(dataset_test, self.args.batch_size, sampler=sampler_test)

        self.global_steps = 0

        for epoch in range(self.args.num_epochs):

            self.log(f"Epoch {epoch} started.")
            train_loss, train_acc = self.run_epoch(loader_train, is_train=True)
            test_loss, test_acc = self.run_epoch(loader_test, is_train=False)

            self.log(f"Epoch {epoch} completed.")
            self.log(f"    Train loss: {train_loss}, ")
            self.log(f"    Test loss: {test_loss}, ")
            self.log(f'    Train accuracy: {train_acc}')
            self.log(f'    Test accuracy.: {test_acc}')

            if self.is_primary and self.args.write_summary:
                self.writer.add_scalar('Test/loss', test_loss, epoch)
                self.writer.add_scalar('Test/accuracy', test_acc, epoch)
                self.save_checkpoint(f'{self.args.label}_epoch{epoch}_{time.strftime("%m%d_%H%M%S")}.pt')

            # self.scheduler.step()

    def run_epoch(self, loader, is_train=False):
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0
        total_correct_samples = 0
        total_samples = 0
        total_steps = 0

        for i, batch in enumerate(tqdm(loader, disable=False if self.is_primary else True)):

            images, labels = batch
            images = images.cuda(self.args.gpu)
            labels = labels.cuda(self.args.gpu)

            batch_size = labels.size(0)

            with amp.autocast(enabled=self.args.fp16):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            if is_train:
                self.optimizer.zero_grad()
                if self.args.fp16:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

            correct_samples = self.count_correct(logits, labels.data)

            total_loss += loss.item()
            total_correct_samples += correct_samples
            total_samples += batch_size
            total_steps += 1
            self.global_steps += 1

            if is_train and self.is_primary and self.args.write_summary:
                self.writer.add_scalar('Train/loss', loss.item(), self.global_steps)
                self.writer.add_scalar('Train/accuracy', correct_samples / batch_size, self.global_steps)

        return total_loss / total_steps, total_correct_samples / total_samples

    def count_correct(self, logits, labels):
        _, preds = torch.max(logits, 1)
        num_correct = preds.eq(labels).sum().item()
        return num_correct

    def load_checkpoint(self, checkpoint_name):
        path = utils.get_data_path('model_checkpoints', checkpoint_name)
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.args.gpu}
        self.model.load_state_dict(torch.load(path, map_location))

    def save_checkpoint(self, checkpoint_name):
        if self.is_primary:
            model_save_path = utils.get_data_path('model_checkpoints', checkpoint_name)
            torch.save(self.model.state_dict(), model_save_path)

    def log(self, message):
        if self.is_primary:
            print(message)


if __name__ == '__main__':
    main()
