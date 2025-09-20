import torch 
import torch.nn as nn 

class PatchEmbedding(nn.Module):
    def __init__(self , image_size = 224 , patch_size = 16 , num_channels = 3 , emb_dims= 768) -> None :
        super().__init__()
        image_size = tuple([image_size , image_size])
        patch_size = tuple([patch_size , patch_size])
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.emb_dims = emb_dims
        self.num_patches = (image_size[0]//patch_size[0]) * (image_size[1]//patch_size[1])

        self.conv = nn.Conv2d(num_channels , emb_dims , kernel_size=patch_size, stride=patch_size)

    
    def forward(self , input_image):
        batch_size , num_channels , height , width = input_image.shape

        assert height == self.image_size[0] and width == self.image_size[1], f"Input image size ({height} , {width}) doesn't match model ({self.image_size[0]} , {self.image_size[1]})"

        return self.conv(input_image).flatten(2).transpose(1,2)
    

