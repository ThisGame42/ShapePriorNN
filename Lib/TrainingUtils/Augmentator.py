import torch.nn as nn
import kornia as K
import kornia.constants as constants


class Augmentator(nn.Module):
    def __init__(self, prob):
        super(Augmentator, self).__init__()
        self.normalise = K.augmentation.Normalize(mean=0, std=1) # zero-mean, unit std
        self.k1 = K.augmentation.RandomAffine(p=prob,
                                              degrees=10.,
                                              translate=(0.1, 0.1),
                                              scale=(0.9, 1.1),
                                              shear=(-0.1, 0.1, -0.1, 0.1),
                                              padding_mode=constants.SamplePadding.REFLECTION,
                                              resample=constants.Resample.NEAREST)
        self.k2 = K.augmentation.RandomHorizontalFlip(p=prob)
        self.k3 = K.augmentation.RandomVerticalFlip(p=prob)

    def forward(self, img, mask):
        assert img.shape == mask.shape

        img = self.normalise(img)

        k1_params = self.k1.generate_parameters(img.shape)
        img = self.k1(img, k1_params)
        mask = self.k1(mask, k1_params)

        k2_params = self.k2.generate_parameters(img.shape)
        img = self.k2(img, k2_params)
        mask = self.k2(mask, k2_params)

        k3_params = self.k3.generate_parameters(img.shape)
        img = self.k3(img, k3_params)
        mask = self.k3(mask, k3_params)

        return img, mask
