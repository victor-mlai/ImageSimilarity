import argparse
from pathlib import Path

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Evaluate Barlow Twins resnet50 features')
    parser.add_argument('--dataset-dir', default='./Totally-Looks-Like-Data/', type=Path,
                        metavar='DIR', help='path to the TTL dataset')
    parser.add_argument('--checkpoint-dir', default='./checkpoints/', type=Path,
                        metavar='DIR', help='path to checkpoint directory')
    parser.add_argument('--logs-dir', default='./logs/', type=Path,
                        metavar='DIR', help='path to logs directory')
    parser.add_argument('--model-filename', default='SiameseNetTriplet_v2.pch', type=str,
                        metavar='Name', help='The name of the file to save the model to')

    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                        help='mini-batch size')
    parser.add_argument('--lr', default=1e-3, type=float, metavar='W',
                        help='learning rate')
    parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight decay')
    parser.add_argument('--hidden-dim', default=1024, type=int, metavar='N',
                        help='projection head hidden dimension size')
    parser.add_argument('--output-dim', default=256, type=int, metavar='N',
                        help='projection head output dimension size')

    return parser
