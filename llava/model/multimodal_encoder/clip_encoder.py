import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from .eva_clip import create_model_and_transforms,get_model_config,CLIPVisionCfg
from transformers.modeling_utils import get_parameter_device,get_parameter_dtype
import os
import logging
logging.basicConfig(level=logging.INFO)
class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.eva_ckpt_path = getattr(args, 'eva_ckpt_path', None)
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            if "eva" in vision_tower.lower():
                model_config = get_model_config("EVA02-CLIP-L-14-336")
                self.eva_config = CLIPVisionCfg(**model_config['vision_cfg'])
            else:
                self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        if 'eva' in  self.vision_tower_name.lower():
            assert self.vision_tower_name == "EVA02-CLIP-L-14-336", "eva model only support EVA02-CLIP-L-14-336"
            if getattr(self, 'eva_ckpt_path', None) is None:
                self.eva_ckpt_path = os.environ.get('EVA_CKPT')
                logging.info(f"eva model load ckpt from {self.eva_ckpt_path}")
            model_config = get_model_config(self.vision_tower_name)
            self.eva_config = CLIPVisionCfg(**model_config['vision_cfg'])
            # logging.info(f"loading /blob/hwq/data/ckpts/T_vit_1024x4_Vlr1e-6T1e-7_Tcc3m_0.5s0.5l_woWup-load_Rcc3m-2024_08_02-02/checkpoints/epoch_10/mp_rank_00_model_states.pt")
            model, _, _ = create_model_and_transforms(self.vision_tower_name,self.eva_ckpt_path,device="cuda")
            # model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336',"eva_clip",device="cuda")
            # import ipdb; ipdb.set_trace()
            self.vision_tower = model.visual
            self.vision_tower.eval()
            self.text_tower = model.text
            self.text_tower.eval()
            # self.clip = model
            # self.clip.eval()
            self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        else:
            self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(True)
        self.text_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images, text):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), text, output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), text, output_hidden_states=True)
            if not torch.is_tensor(image_forward_outs):
                image_features = self.feature_select(image_forward_outs).to(images.dtype)
            else: 
                image_features = image_forward_outs

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return get_parameter_dtype(self.vision_tower)

    @property
    def device(self):
        return get_parameter_device(self.vision_tower)

    @property
    def config(self):
        if self.is_loaded:
            if 'eva' in  self.vision_tower_name.lower():
                # logger.info(f"eva config: {self.eva_config}")
                return self.eva_config
            # else:
            #     raise ValueError('CLIPVisionTower does not have config attribute. Use `eva_config` instead.')
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.eva_config.width if 'eva' in  self.vision_tower_name.lower() else self.config.hidden_size
        # return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.eva_config.image_size // self.eva_config.patch_size if 'eva' in  self.vision_tower_name.lower() else self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.eva_config.image_size // self.eva_config.patch_size)**2 if 'eva' in  self.vision_tower_name.lower() else (self.config.image_size // self.config.patch_size) ** 2
        # return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    model = CLIPVisionTower("eva",None)
