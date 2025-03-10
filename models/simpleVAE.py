import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleVAE(nn.Module):
    def __init__(self, sample, embed_dim, latent_dim):
        super(SimpleVAE, self).__init__()
        """
        Parameter settings:
        __init__
            - Sample: (batch, channels=1 or 3, height, width) or (channels, height, width)
            - embed_dim, latent_dim: integer
            
        forward:
            - x: (batch, channels=3, height, width)
        """
        # 
        self.sample = sample
        if sample.dim() == 3:
            self.sample = sample.unsqueeze(0)  # Now shape becomes (1, channels, height, width)
        
        # Encoder:
        SAMPLE_CHANNEL = self.sample.size(1)
        FINIAL_ENCODER_CHANNEL = 32
        self.encoder = nn.Sequential(
            nn.Conv2d(SAMPLE_CHANNEL, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, FINIAL_ENCODER_CHANNEL, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Latent: projection # Q: how to convert into embed_dim?
        encoder_out_dim = FINIAL_ENCODER_CHANNEL * self.sample.size(2) * self.sample.size(3) # similar to: - * math.prod(sample.shape[2:])
        self.proj_embed = nn.Linear(encoder_out_dim, embed_dim) 
        #print(f"encoder_out_dim: {encoder_out_dim}") # debug
        
        # Latent: mu and log variance
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)
        
        # Decoder # Q: how to change into the shape of self.sample 
        decoder_out_dim = self.sample.numel() // self.sample.shape[0] 
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, embed_dim), 
            nn.ReLU(),
            nn.Linear(embed_dim, decoder_out_dim), # how to convert into original image shape?
            nn.ReLU(),
            #
        )
        
    def reparameterize(self, mu, logvar):
        """ z = mu + eta * e^(1/2 logvar) """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        #
        
        # Encoder
        x = self.encoder(x) # shape: (batch, channels, height, width)
        x = x.view(x.size(0), -1) # shape: (batch, hid_dim), same with nn.flatten()
        #print("Encoder shape", x.shape) # debug
        
        # Projection
        x = self.proj_embed(x)
        #print("Projection shape", x.shape) # debug
        
        # Latent
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        # reparameterization -> z (latent)
        z = self.reparameterize(mu, logvar)
        
        # Decoder
        x_dec = self.decoder(z)
        
        #print("Decoder shape", x_dec.shape) # debug
        
        x_dec = x_dec.view(-1, *self.sample.shape[1:]) 
        
        # Advice: 將輸出壓縮到 [0, 1] 範圍，通常用於重構圖像
        x_dec = torch.sigmoid(x_dec)
        
        return x_dec, mu, logvar

if __name__ == "__main__":
    # requirement for model input: (channels, height, width) or (batch, channels, height, width)
    sample1 = torch.zeros(1, 64, 12)
    sample2 = torch.zeros(128, 1, 64, 12)
    
    model = SimpleVAE(sample1, embed_dim=512, latent_dim=128)
    model = SimpleVAE(sample2, embed_dim=512, latent_dim=128)
    #output, mu, logvar = model(sample1) # not allow
    output, mu, logvar = model(sample2)
    
    # check for: sample.shape == output.shape
    print("Output shape:", output.shape)
    print("Mu shape:", mu.shape)
    print("Logvar shape:", logvar.shape)
