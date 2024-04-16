from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from constants import DEVICE

class SamSegmentor:
    def __init__(self, checkpoint_path='sam_vit_h_4b8939.pth', model_type='vit_h'):
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.model = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path).to(device=DEVICE)
        self.mask_generator = SamAutomaticMaskGenerator(self.model)
