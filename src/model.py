import torch 
import torch.nn as nn 
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torchvision import transforms
import matplotlib.pyplot as plt
import os 
from random_word import RandomWords
import random
DATADIR = "random_generated_images/"
MODELSDIR = "git/GEHENNUM/src/model_parameters/"
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]
    
def save_image(images,nmax=128,show_images=False,return_path=False):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()), nrow=8).permute(1, 2, 0))
    random_name = RandomWords().get_random_word()
    print(f"File saved at:{DATADIR + random_name}.png")
    plt.savefig(DATADIR + f"{random_name}.png")
    if show_images:
        plt.show()
    if return_path:
        return DATADIR + random_name + ".png"

class GEHENNUM:
    def __init__(self,latent_dim=128):
        self.latent_dim = latent_dim 
        self.stats = stats
        self.discriminator = nn.Sequential(
            # in: 3 x 64 x 64

            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 64 x 32 x 32

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 128 x 16 x 16

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 256 x 8 x 8

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 512 x 4 x 4

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # out: 1 x 1 x 1

            nn.Flatten(),
            nn.Sigmoid())

        self.generator = nn.Sequential(
            # in: latent_size x 1 x 1

            nn.ConvTranspose2d(128, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # out: 512 x 4 x 4

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # out: 256 x 8 x 8

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # out: 128 x 16 x 16

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # out: 64 x 32 x 32

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # out: 3 x 64 x 64
        )
        self.device = get_default_device()
        self.discriminator = to_device(self.discriminator,self.device)
        self.generator = to_device(self.generator,self.device)
        self.generator.load_state_dict(torch.load(MODELSDIR + f"gehennum_generator.ckpt"))
        self.discriminator.load_state_dict(torch.load(MODELSDIR + f"gehennum_discriminator.ckpt"))

    def generate_image(self,save_img = True):
        noise = torch.randn(128, self.latent_dim, 1, 1)
        images = self.generator.forward(noise.to(self.device))
        logits = self.discriminator.forward(images)
        best_img_idx = int(torch.argmax(logits).to("cpu"))
        if(save_img):
            save_image(images[best_img_idx,:,:,:].to("cpu"))
        return images[best_img_idx,:,:,:].to("cpu")

def main():
    print("Generating image...")
    gehennum = GEHENNUM()
    img = gehennum.generate_image()
    print("Done!")

if __name__ == "__main__":
    main()
