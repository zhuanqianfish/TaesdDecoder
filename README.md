使用TAESD用来取代VAE进行潜空间解码
会损失一点图片质量，降低大量解码时的峰值显存

use TAESD decoded image.you need donwload taesd_decoder.pth and  taesdxl_decoder.pth to vae_approx folder first.\n It will result in a slight loss of image quality and a significant decrease in peak video memory during decoding.

需要先下载taesd_decoder.pth 和 taesdxl_decoder.pth 到 model/vae_approx 文件夹下
TAESD项目地址：https://github.com/madebyollin/taesd

you need donwload taesd_decoder.pth and  taesdxl_decoder.pth to vae_approx folder first.