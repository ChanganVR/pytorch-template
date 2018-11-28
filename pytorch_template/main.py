import sys
import logging
import os
import argparse
import shutil
import pprint
import importlib.util

import git
import torch


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--config', type=str, default='config.py')
    parser.add_argument('--model', type=str, default='dqn')
    parser.add_argument('--output_dir', type=str, default='data/output')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()

    # configure output directory
    make_new_dir = True
    if os.path.exists(args.output_dir):
        key = input('Output directory already exists! Overwrite the folder? (y/n)')
        if key == 'y':
            shutil.rmtree(args.output_dir)
        else:
            make_new_dir = False
            args.config = os.path.join(args.output_dir, args.config)
    if make_new_dir:
        os.makedirs(args.output_dir)

        # overwrite the default configuration value
        spec = importlib.util.spec_from_file_location('config', args.config)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)

        train_config = config.TrainConfig(args.debug)
        train_config.trainer.batch_size = args.batch_size
        train_config.trainer.learning_rate = args.learning_rate

        shutil.copy(args.config, args.output_dir)

    # configure logging
    log_file = os.path.join(args.output_dir, 'output.log')
    file_handler = logging.FileHandler(log_file, mode='a')
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    repo = git.Repo(search_parent_directories=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(sys.argv)
    logging.info('Current git head hash code: {}'.format(repo.head.object.hexsha))
    logging.info('Using device: %s', device)
    logging.info(pprint.pformat(vars(args), indent=4))

    # training
    if args.resume:
        logging.info('Resume training...')
    else:
        logging.info('Start training...')


if __name__ == '__main__':
    main()
