"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import sys
sys.path.append('/opt/data/borghei/LAVIS/stable_control_representations/vc_models/src')
from vc_models.transforms.to_tensor_if_not import ToTensorIfNot

import re

from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor
from lavis.processors.randaugment import RandomAugment
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
# from stable_control_representations.vc_models.src.vc_models.transforms.to_tensor_if_not import ToTensorIfNot
# from vc_models.transforms.to_tensor_if_not import ToTensorIfNot
from PIL import Image
import numpy as np

class BlipImageBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)


@registry.register_processor("blip_caption")
class BlipCaptionProcessor(BaseProcessor):
    def __init__(self, prompt="", max_words=50):
        self.prompt = prompt
        self.max_words = max_words

    def __call__(self, caption):
        caption = self.prompt + self.pre_caption(caption)

        return caption

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 50)

        return cls(prompt=prompt, max_words=max_words)

    def pre_caption(self, caption):
        caption = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > self.max_words:
            caption = " ".join(caption_words[: self.max_words])

        return caption


@registry.register_processor("blip_question")
class BlipQuestionProcessor(BaseProcessor):
    def __init__(self, max_words=50):
        self.max_words = max_words

    def __call__(self, question):
        return self.pre_question(question)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        max_words = cfg.get("max_words", 50)

        return cls(max_words=max_words)

    def pre_question(self, question):
        question = re.sub(
            r"([.!\"()*#:;~])",
            "",
            question.lower(),
        )
        question = question.rstrip(" ")

        # truncate question
        question_words = question.split(" ")
        if len(question_words) > self.max_words:
            question = " ".join(question_words[: self.max_words])

        return question


@registry.register_processor("blip_image_train")
class BlipImageTrainProcessor(BlipImageBaseProcessor):
    def __init__(
        self, image_size=384, mean=None, std=None, min_scale=0.5, max_scale=1.0
    ):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(min_scale, max_scale),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                RandomAugment(
                    2,
                    5,
                    isPIL=True,
                    augs=[
                        "Identity",
                        "AutoContrast",
                        "Brightness",
                        "Sharpness",
                        "Equalize",
                        "ShearX",
                        "ShearY",
                        "TranslateX",
                        "TranslateY",
                        "Rotate",
                    ],
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 384)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
        )


@registry.register_processor("blip_image_eval")
class BlipImageEvalProcessor(BlipImageBaseProcessor):
    def __init__(self):#, image_size=384, mean=None, std=None, resize_size = 256, center_crop = True):
        # super().__init__(mean=mean, std=std)

        # self.transform = transforms.Compose(
        #     [
        #         # transforms.Resize(
        #         #     (image_size, image_size), interpolation=InterpolationMode.BICUBIC
        #         # ),
        #         # transforms.ToTensor(),
        #         # self.normalize,
        #         transforms.Resize(resize_size, interpolation=InterpolationMode.BILINEAR),
        #         transforms.CenterCrop(resize_size) if center_crop else transforms.Identity(),
        #         ToTensorIfNot(),
        #         transforms.Normalize([0.5], [0.5]),
        #     ]
        # )
        pass

    def __call__(self, item):
        # # return self.transform(item)
        # # Handle different input types
        # if isinstance(item, np.ndarray):
        #     # Convert numpy array to PIL Image, apply transforms, and add batch dimension
        #     return self.transform(Image.fromarray(item))#.unsqueeze(0)
        # else:
        #     # Original behavior for other input types (like PIL Images)
        #     return self.transform(item)
        return item

    @classmethod
    def from_config(cls, cfg=None):
        # if cfg is None:
        #     cfg = OmegaConf.create()

        # image_size = cfg.get("image_size", 384)

        # mean = cfg.get("mean", None)
        # std = cfg.get("std", None)

        return cls()#image_size=image_size, mean=mean, std=std)


@registry.register_processor("blip2_image_train")
class Blip2ImageTrainProcessor(BlipImageBaseProcessor):
    def __init__(self):
    #     self, image_size=364, mean=None, std=None, min_scale=0.5, max_scale=1.0, resize_size = 256, center_crop = True
    # ):
    #     super().__init__(mean=mean, std=std)

    #     self.transform = transforms.Compose(
    #         [
    #             # transforms.RandomResizedCrop(
    #             #     image_size,
    #             #     scale=(min_scale, max_scale),
    #             #     interpolation=InterpolationMode.BICUBIC,
    #             # ),
    #             # transforms.RandomHorizontalFlip(),
    #             # transforms.ToTensor(),
    #             # self.normalize,
    #             transforms.Resize(resize_size, interpolation=InterpolationMode.BILINEAR),
    #             transforms.CenterCrop(resize_size) if center_crop else transforms.Identity(),
    #             ToTensorIfNot(),
    #             transforms.Normalize([0.5], [0.5]),
    #         ]
        # )
        pass

    def __call__(self, item):
        # return self.transform(item)
        # Handle different input types
        # if isinstance(item, np.ndarray):
        #     # Convert numpy array to PIL Image, apply transforms, and add batch dimension
        #     return self.transform(Image.fromarray(item))#.unsqueeze(0)
        # else:
        #     # Original behavior for other input types (like PIL Images)
        #     return self.transform(item)
        return item

    @classmethod
    def from_config(cls, cfg=None):
        # if cfg is None:
        #     cfg = OmegaConf.create()

        # image_size = cfg.get("image_size", 364)

        # mean = cfg.get("mean", None)
        # std = cfg.get("std", None)

        # min_scale = cfg.get("min_scale", 0.5)
        # max_scale = cfg.get("max_scale", 1.0)

        return cls()
        #     image_size=image_size,
        #     mean=mean,
        #     std=std,
        #     min_scale=min_scale,
        #     max_scale=max_scale,
        # )