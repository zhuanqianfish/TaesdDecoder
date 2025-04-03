from typing import Tuple
import torch
import folder_paths
from .taesd import TAESD

# thanks for madebyollin's taesd code. https://github.com/madebyollin/taesd
# you need donwload taesd_decoder.pth and  taesdxl_decoder.pth to vae_approx folder first.

class TaesdVAEDecoder:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "latent": ("LATENT",),
                "vae": (folder_paths.get_filename_list("vae_approx"), {})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    OUTPUT_IS_LIST = (False,)

    CATEGORY = "latent"
    DESCRIPTION = "use TAESD decodes latent images back into pixel space images."

    def __init__(self):
        self.taesd = None

    def decode(self, latent: torch.Tensor, vae: str) -> Tuple[torch.Tensor]:
        dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        if self.taesd is None:
            self.taesd = TAESD(None, folder_paths.get_full_path("vae_approx", vae)).to(dev)
        
        # 获取latent samples并调整到正确设备，添加0.18215缩放因子
        latent_samples = latent['samples'].to(dev) * 0.18215
        
        # 解码latent到图像空间
        x_sample = self.taesd.decoder(latent_samples).detach()
        
        # 数值范围调整 [-1,1] -> [0,1]
        x_sample = x_sample.sub(0.5).mul(2)  # 从[0,1]到[-1,1]
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)  # 回到[0,1]范围
        
        # 调整维度顺序为ComfyUI需要的格式 [B,H,W,C]
        x_sample = x_sample.permute(0, 2, 3, 1)
        
        return (x_sample,)