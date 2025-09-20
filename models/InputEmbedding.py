import torch
import torch.nn as nn 

from .PatchEmbedding import PatchEmbedding


class InputEmbedding(nn.Module):
    """
    Takes input iamges converts to patch embedding with positional encoding and cls token
    """
    def __init__(self , image_size = 224 , patch_size = 16 , num_channels = 3 , emb_dims= 768) -> None:
        super().__init__()
        
        # convert to patches
        self.patch_embeddings = PatchEmbedding(
            image_size= image_size , patch_size= patch_size , num_channels = num_channels , emb_dims= emb_dims
        )
        num_patches = self.patch_embeddings.num_patches

        # cls token
        self.cls_token =  nn.Parameter(torch.zeros(1,1 , emb_dims))
        
        # positional encoding 
        self.pos_embeddings = nn.Parameter(torch.zeros(1 , num_patches + 1 , emb_dims))

        # dropout layer
        self.dropout = nn.Dropout(0.1)

    def forward(self , input_image):
        batch_size , num_channels , height , width = input_image.shape

        patches = self.patch_embeddings(input_image)
        cls_tokens = self.cls_token.expand(batch_size , -1 , -1)
        patches_with_cls = torch.cat((cls_tokens , patches) , dim = 1)

        patches_with_cls_positionalencoding = patches_with_cls  + self.pos_embeddings
        
        final_embedding  = self.dropout(patches_with_cls_positionalencoding)
        return final_embedding
        