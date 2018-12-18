import argparse
import os
from os import path
import copy
from tqdm import tqdm
import torch
from torch import nn
from gan_training import utils
from gan_training.checkpoints import CheckpointIO
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.inputs import get_dataset
from gan_training.config import (
    load_config, build_models
)

# Arguments
parser = argparse.ArgumentParser(
    description='Test a trained GAN and create visualizations.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()

config = load_config(args.config)
is_cuda = (torch.cuda.is_available() and not args.no_cuda)

# Shorthands
nlabels = config['data']['nlabels']
out_dir = config['training']['out_dir']
batch_size = config['test']['batch_size']
sample_size = config['test']['sample_size']
sample_nrow = config['test']['sample_nrow']
fid_sample_size = config['training']['fid_sample_size']
checkpoint_dir = path.join(out_dir, 'chkpts')
img_dir = path.join(out_dir, 'test', 'img')
img_all_dir = path.join(out_dir, 'test', 'img_all')

# Creat missing directories
if not path.exists(img_dir):
    os.makedirs(img_dir)
if not path.exists(img_all_dir):
    os.makedirs(img_all_dir)

# Logger
checkpoint_io = CheckpointIO(
    checkpoint_dir=checkpoint_dir
)

device = torch.device("cuda:0" if is_cuda else "cpu")

generator, discriminator = build_models(config)
print(generator)
print(discriminator)

# Put models on gpu if needed
generator = generator.to(device)
discriminator = discriminator.to(device)

# Use multiple GPUs if possible
generator = nn.DataParallel(generator)
discriminator = nn.DataParallel(discriminator)

# Register modules to checkpoint
checkpoint_io.register_modules(
    generator=generator,
    discriminator=discriminator,
)

# Test generator
if config['test']['use_model_average']:
    generator_test = copy.deepcopy(generator)
    checkpoint_io.register_modules(generator_test=generator_test)
else:
    generator_test = generator

# Distributions
ydist = get_ydist(nlabels, device=device)
zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'],
                  device=device)

# Evaluator
fid_real_samples = None
if config['test']['compute_inception']:
    # Dataset
    train_dataset, nlabels = get_dataset(
        name=config['data']['type'],
        data_dir=config['data']['train_dir'],
        size=config['data']['img_size'],
        lsun_categories=config['data']['lsun_categories_train']
    )
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=config['training']['nworkers'],
            shuffle=True, pin_memory=True, sampler=None, drop_last=True
    )
    fid_real_samples, _ = utils.get_nsamples(train_loader, fid_sample_size)

evaluator = Evaluator(generator_test, zdist, ydist,
                      batch_size=batch_size, device=device,
                      fid_real_samples=fid_real_samples,
                      fid_sample_size=fid_sample_size)

# Load checkpoint if existant
it = checkpoint_io.load('model.pt')

# Inception score
if config['test']['compute_inception']:
    print('Computing inception score...')
    inception_mean, inception_std, fid = evaluator.compute_inception_score()
    print('Inception score: %.4f +- %.4f, FID: %.2f' % (inception_mean, inception_std, fid))

# Samples
print('Creating samples...')
ztest = zdist.sample((sample_size,))
x = evaluator.create_samples(ztest)
utils.save_images(x, path.join(img_all_dir, '%08d.png' % it),
                  nrow=sample_nrow)
if config['test']['conditional_samples']:
    for y_inst in tqdm(range(nlabels)):
        x = evaluator.create_samples(ztest, y_inst)
        utils.save_images(x, path.join(img_dir, '%04d.png' % y_inst),
                          nrow=sample_nrow)
