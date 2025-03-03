import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import FashionMNIST
from PIL import Image
import sys
import argparse

def tokenizer(text, encode=True, mask=None, max_seq_length=32):
    if encode:
        out = chr(2) + text + chr(3) # Adding SOT and EOT tokens
        out = out + "".join([chr(0) for _ in range(max_seq_length-len(out))]) # Adding Padding
        out = torch.IntTensor(list(out.encode("utf-8"))) # Encoding Text
        mask = torch.ones(len(out.nonzero()))
        mask = torch.cat((mask,torch.zeros(max_seq_length-len(mask)))).type(torch.IntTensor)
    else:
        out = [chr(x) for x in text[1:len(mask.nonzero())-1]]
        out = "".join(out)
        mask = None

    return out, mask 

class PrepareDS(Dataset):
    def __init__(self, img_size, train=True):
        #self.dataset = load_dataset("fashion_mnist")
        
        self.transform= T.Compose([T.Resize(img_size), 
                                   T.ToTensor()
                        ])
        self.dataset= FashionMNIST(
            root='./data', 
            train=train, 
            download=True
        )
        if train:
            self.split="train"
        else:
            self.split= "test"

        self.captions = {0: "An image of a t-shirt/top",
                        1: "An image of trousers",
                        2: "An image of a pullover",
                        3: "An image of a dress",
                        4: "An image of a coat",
                        5: "An image of a sandal",
                        6: "An image of a shirt",
                        7: "An image of a sneaker",
                        8: "An image of a bag",
                        9: "An image of an ankle boot"}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img= self.dataset[idx][0]
        
        img= self.transform(img)

        cap, mask= tokenizer(self.captions[self.dataset[idx][1]])
        mask= mask.repeat(len(mask), 1)

        return {"image": img, "caption": cap, "mask":mask}
    
# text encoding
class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(InputEmbedding, self).__init__()
        self.embedding= nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)

class ImagePatching(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super(ImagePatching, self).__init__()
        self.projection= nn.Conv2d(in_channels, embed_dim, kernel_size= patch_size, stride= patch_size)
    
    ## what transformation happens here:
    ## lets say we pass an input image batch of shape: [batch_size, input_channels, Height, Width]
    ## then finally transforms it into: [batch_size, (image_size//patch_size)**2, d_model]
    def forward(self, x): 
        x= self.projection(x) 
        x= x.flatten(2)
        return x.transpose(-2, -1) 

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout):
        super(PositionalEncoding, self).__init__()

        pe= torch.zeros(max_len, d_model)

        for i in range(max_len):
            for j in range(d_model):
                if j%2==0:
                    pe[i][j]= np.sin(i/ (10000** (j/d_model)))
                else:
                    pe[i][j]= np.cos(i/(10000** ((j-1)/d_model)))
        self.register_buffer("pe", pe.unsqueeze(0))

        self.dropout= nn.Dropout(dropout)
    def forward(self, x):
        x= x+ self.pe
        return x
    

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_size):
        super(AttentionHead, self).__init__()
        self.head_size= head_size
        self.query= nn.Linear(embed_dim, head_size)
        self.key= nn.Linear(embed_dim, head_size)
        self.value= nn.Linear(embed_dim, head_size)

    def forward(self, x, mask=None):
        Q= self.query(x)
        K= self.key(x)
        V= self.value(x)

        attention= Q@K.transpose(-2, -1)
        attention= attention/(self.head_size**0.5)

        if mask is not None:
            attention= attention.masked_fill(mask==0, float("-inf"))
        attention= torch.softmax(attention, dim=-1)

        attention= attention@V
        return attention
        

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, nheads):
        super().__init__()
        self.head_size= embed_dim//nheads
        self.W_o= nn.Linear(embed_dim, embed_dim)
        self.attentionblocks= nn.ModuleList([AttentionHead(embed_dim, self.head_size) for _ in range(nheads)])

    def forward(self, x, mask= None):
        out= torch.cat([head(x, mask) for head in self.attentionblocks] ,dim =-1)
        out= self.W_o(out)
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nheads, r_mlp=4):
        super().__init__()
        self.norm1= nn.LayerNorm(d_model)
        self.mha= MultiHeadAttention(d_model, nheads)
        self.mlp= nn.Sequential(
            nn.Linear(d_model, d_model*r_mlp),
            nn.GELU(),
            nn.Linear(d_model* r_mlp, d_model)
        )
        self.norm2= nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x= x+ self.mha(self.norm1(x), mask)
        x= x+ self.mlp(self.norm2(x))
        return x

class ImageEncoder(nn.Module):
    def __init__(self, width, image_size, patch_size, in_channels, n_layers,nheads, embed_dim, dropout=0.1):
        super(ImageEncoder, self).__init__()
        assert (image_size[0] % patch_size[0]==0) and (image_size[1] % patch_size[1]==0), "image_size dim shall be divisible by patchsize"
        assert width%nheads==0, "width shall be divisible by nheads"
        
        self.image_embedding= ImagePatching(image_size, patch_size, in_channels, width)
        self.npatches= (image_size[0]*image_size[1]) // (patch_size[0]*patch_size[1])
        self.cls_token= nn.Parameter(torch.randn(1,1, width))

        self.positional_embedding= PositionalEncoding(width, self.npatches+1, dropout)
        self.encoder= nn.ModuleList([TransformerEncoder(width, nheads) for _ in range(n_layers)])

        self.projection= nn.Parameter(torch.randn(width ,embed_dim))

    def forward(self, x):
        x= self.image_embedding(x)
        #print(x.shape, self.cls_token.shape)
        x= torch.cat([self.cls_token.expand(x.size()[0], -1, -1), x], dim=1)
        x= self.positional_embedding(x)

        for encoder in self.encoder:
            x= encoder(x)

        x= x[:, 0, :]

        if self.projection is not None:
            x= x@self.projection

        x= x/torch.norm(x, dim=-1, keepdim=True)
        return x
    
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, width, max_seq_length, n_heads, n_layers, emb_dim):
        super().__init__()

        self.max_seq_length = max_seq_length

        self.encoder_embedding = nn.Embedding(vocab_size, width)

        self.positional_embedding = PositionalEncoding(width, max_seq_length, 0.1)

        self.encoder = nn.ModuleList([TransformerEncoder(width,n_heads) for _ in range(n_layers)])

        # learned proj of image to embed
        self.projection = nn.Parameter(torch.randn(width, emb_dim))

    def forward(self, text, mask=None):
        # Text Embedding
        x = self.encoder_embedding(text)

        # Positional Embedding
        x = self.positional_embedding(x)

        # Transformer Encoder
        for encoder_layer in self.encoder:
            x = encoder_layer(x, mask=mask)

        # Takes features from the EOT Embedding
        x = x[torch.arange(text.shape[0]), torch.sub(torch.sum(mask[:,0],dim=1),1)]

        # joint multimodal embedding
        if self.projection is not None:
            x = x @ self.projection

        x = x / torch.norm(x, dim=-1, keepdim=True)

        return x
    
# model

class CLIP(nn.Module):
    def __init__(self, emb_dim, vit_width, img_size, patch_size, n_channels, vit_layers, vit_heads, vocab_size, text_width, max_seq_length, text_heads, text_layers):
        super().__init__()

        self.image_encoder = ImageEncoder(vit_width, img_size, patch_size, n_channels, vit_layers, vit_heads, emb_dim)
        
        self.text_encoder = TextEncoder(vocab_size, text_width, max_seq_length, text_heads, text_layers, emb_dim)
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1/0.07))
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, image,text, mask=None):
        i_encode= self.image_encoder(image)
        t_encode= self.text_encoder(text, mask)
        #print(i_encode.shape, t_encode.shape)
        logits= (i_encode @ t_encode.T)/ torch.exp(self.temperature)
        labels= torch.arange(logits.shape[0]).to(self.device)

        #return logits

        loss_i= nn.functional.cross_entropy(logits, labels)
        loss_t= nn.functional.cross_entropy(logits.transpose(-2, -1), labels)

        loss= (loss_i+loss_t)/2
        return loss
    
emb_dim = 32
vit_width = 9
img_size = (28,28)
patch_size = (14,14)
n_channels = 1
vit_layers = 3
vit_heads = 3
vocab_size = 256
text_width = 32
max_seq_length = 32
text_heads = 8
text_layers = 4


test_set= PrepareDS(img_size, train=False)

test_dataloader= DataLoader(test_set, shuffle=False, batch_size= 32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

#model = CLIP(emb_dim, vit_width, img_size, patch_size, n_channels, vit_layers, vit_heads, vocab_size, text_width, max_seq_length, text_heads, text_layers).to(device)
from tqdm import tqdm

model = CLIP(emb_dim, vit_width, img_size, patch_size, n_channels, vit_layers, vit_heads, vocab_size, text_width, max_seq_length, text_heads, text_layers).to(device)
model.load_state_dict(torch.load("clip.pt", map_location=device))

def img2text(query, k):

    class_names =["t-shirt/top",
                        "trousers",
                        "pullover",
                        "dress",
                        "coat",
                        "sandal",
                        "shirt",
                        "sneaker",
                        "bag",
                        "ankle boot"]
    
    text = torch.stack([tokenizer(x)[0] for x in class_names]).to(device)
    mask = torch.stack([tokenizer(x)[1] for x in class_names])
    mask = mask.repeat(1,len(mask[0])).reshape(len(mask),len(mask[0]),len(mask[0])).to(device)

    
    idx= query
    img = test_set[idx]["image"][None,:]
    plt.imshow(img[0].permute(1, 2, 0)  ,cmap="gray")
    plt.title(tokenizer(test_set[idx]["caption"], encode=False, mask=test_set[idx]["mask"][0])[0])
    plt.show()
    img = img.to(device)
    model.eval()
    with torch.no_grad():
        image_features = model.image_encoder(img)
        text_features = model.text_encoder(text, mask=mask)


        similarity = ((100.0 * image_features @ text_features.T)/torch.exp(model.temperature)).softmax(dim=-1)
        values, indices = similarity[0].topk(k)

    # Print the result
        print("\nTop predictions:\n")
        for value, index in zip(values, indices):
            print(f"{class_names[int(index)]:>16s}: {100 * value.item():.2f}%")

def text2img(query, k):
    query_text, query_mask = tokenizer(query)
    query_text = query_text.unsqueeze(0).to(device)  # Add batch dimension
    query_mask = query_mask.unsqueeze(0)
    query_mask= query_mask.repeat(len(query_mask[0]), 1).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        text_encoded= model.text_encoder(query_text, query_mask)

        ## get the image 
        image_features= torch.cat([batch["image"] for batch in test_dataloader], dim=0).to(device)
        image_features= model.image_encoder(image_features)

        

        similarity= (text_encoded@ image_features.T)/torch.exp(model.temperature)
        similarity= similarity.softmax(dim=-1)

        confidence, top_indices= similarity[0].topk(k)

        for conf, i in zip(confidence, top_indices):
            image_= test_set[int(i)]["image"].permute(1, 2, 0).numpy()
            plt.imshow(image_)
            plt.axis("off")
            plt.show()


parser= argparse.ArgumentParser()
parser.add_argument( "--mode",required=True, choices= ["i2t", "t2i"], help= "for i2t give any index between 0 to 10000 (10000 non-inclusive); for t2i give text within the mentioned options in list" )
parser.add_argument("--k", required=True, help="add the value of top k required values")
parser.add_argument("--query", required= True, help="Enter the adequate number of retrieval as required")

args= parser.parse_args()

if args.mode=="i2t":
    img2text(int(args.query), int(args.k))
elif args.mode=="t2i":
    text2img(args.query, int(args.k))
