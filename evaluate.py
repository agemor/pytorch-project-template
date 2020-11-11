import argparse

import torch
import torch.nn as nn
import torch.utils.data as data

import torchvision
from tqdm import tqdm

import utils
from model import DummyModel


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

    # Type of experiment
    parser.add_argument('--test_type', type=str, default='accuracy', help="Experiment type")

    # Training environment configuration
    parser.add_argument('--num_gpus', type=int, default=2, help="Number of GPUs to use")
    parser.add_argument('--deterministic', action='store_true', default=False, help="Deterministic mode for reproducibility")
    parser.add_argument('--random_seed', type=int, default=7343, help="Random seed (optional)")

    # Dataset and trained model (for resume training)
    parser.add_argument('--dataset', type=str, help="Name of the dataset to use")
    parser.add_argument('--model_checkpoint', type=str, help="Path to the model checkpoints")

    args = parser.parse_args()
    if args.deterministic:
        utils.set_deterministic(args.random_seed)

    run(args)


def run(args):
    model = DummyModel()

    rgb_mean = (0.4914, 0.4822, 0.4465)
    rgb_std = (0.2023, 0.1994, 0.2010)

    dataset_test = torchvision.datasets.CIFAR10('./data/datasets', train=False, download=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(rgb_mean, rgb_std),
                                                ]))
    evaluator = Evaluator(args, model)
    evaluator.load_checkpoint(args.model_checkpoint)
    evaluator.eval(dataset_test)


class Evaluator:

    def __init__(self, args, model):
        self.args = args
        self.model = nn.DataParallel(model, device_ids=[i for i in range(self.args.num_gpus)]).cuda(0)

    def eval(self, dataset):

        test_type = self.args.test_type.lower()

        if test_type == 'accuracy':
            self.test_accuracy(dataset)
        else:
            print('Unknown test type')

    def test_accuracy(self, dataset):

        print('Accuracy test started.')

        self.model.eval()

        criterion = nn.CrossEntropyLoss()
        loader = data.DataLoader(dataset, 256)

        total_loss = 0
        total_correct_samples = 0
        total_samples = 0
        total_steps = 0

        for i, batch in enumerate(tqdm(loader)):
            images, labels = batch
            images = images.cuda(0)
            labels = labels.cuda(0)

            batch_size = labels.size(0)

            logits = self.model(images)
            loss = criterion(logits, labels)

            correct_samples = self.count_correct(logits, labels.data)

            total_loss += loss.item()
            total_correct_samples += correct_samples
            total_samples += batch_size
            total_steps += 1

        total_loss = total_loss / total_steps
        total_acc = total_correct_samples / total_samples

        print('Accuracy test completed.')
        print(f'    Test loss: {total_loss}')
        print(f'    Test accuracy: {total_acc}')

    def count_correct(self, logits, labels):
        _, preds = torch.max(logits, 1)
        num_correct = preds.eq(labels).sum().item()
        return num_correct

    def load_checkpoint(self, checkpoint_name):
        path = utils.get_data_path('model_checkpoints', checkpoint_name)
        map_location = {'cuda:%d' % 0: 'cuda:%d' % 0}
        self.model.load_state_dict(torch.load(path, map_location))


if __name__ == '__main__':
    main()
