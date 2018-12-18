from gan_training.models import (
    resnet
)

generator_dict = {
    'resnet': resnet.Generator,
}

discriminator_dict = {
    'resnet': resnet.Discriminator,
}
