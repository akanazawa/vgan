# Variational Discriminator Bottleneck
Code for the image generation experiments in [Variational Discriminator Bottleneck: Improving Imitation Learning, Inverse RL, and GANs by Constraining Information Flow](https://xbpeng.github.io/projects/VDB/index.html).

# Bibtex
```
@inproceedings{
  VDBPeng18,
  title={Variational Discriminator Bottleneck: Improving Imitation Learning,
  Inverse RL, and GANs by Constraining Information Flow},
  author = {Peng, Xue Bin and Kanazawa, Angjoo and Toyer, Sam and Abbeel, Pieter
  and Levine, Sergey},
  booktitle={ICLR},
  year={2019}
}
```


# Acknowledgement
Our code is built on the GAN implmentation of
[Which Training Methods for GANs do actually Converge?](https://avg.is.tuebingen.mpg.de/publications/meschedericml2018)
\[Mescheder et al. ICML 2018\].
This repo adds the VGAN and instance noise implementations, along with FID computation.


# Usage
First download your data and put it into the `./data` folder.

To train a new model, first create a config script similar to the ones provided in the `./configs` folder.  You can then train you model using
```
python train.py PATH_TO_CONFIG
```

You can monitor the training with tensorboard:
```
tensorboard --logdir output/<MODEL_NAME>/monitoring/
```

# Experiments

To generate samples, use
```
python test.py PATH_TO_CONIFG
```

You can also create latent space interpolations using
```
python interpolate.py PATH_TO_CONFIG
```


# Pre-trained model
A pre-trained model for CelebA-HQ can be found [here](https://drive.google.com/file/d/1qwmQtkGm-i0IKFqSEVWugh4DVpJW4CQJ/view?usp=sharing)
