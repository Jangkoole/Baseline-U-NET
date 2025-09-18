import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedVAE(nn.Module):
    def __init__(self, input_channels=3, latent_dim=128):  # 增大潜在维度
        super(ImprovedVAE, self).__init__()
        self.encoder = Encoder(input_channels, latent_dim)
        self.decoder = Decoder(latent_dim, input_channels)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar


class Encoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=128):
        super(Encoder, self).__init__()
        # 更深的网络结构
        self.conv1 = nn.Conv2d(input_channels, 64, 4, stride=2, padding=1)  # 256->128
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)  # 128->64
        self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)  # 64->32
        self.conv4 = nn.Conv2d(256, 512, 4, stride=2, padding=1)  # 32->16

        self.fc_mu = nn.Linear(512 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(512 * 16 * 16, latent_dim)

        # 添加dropout防止过拟合
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        h = F.leaky_relu(self.conv1(x), 0.2)
        h = F.leaky_relu(self.conv2(h), 0.2)
        h = F.leaky_relu(self.conv3(h), 0.2)
        h = F.leaky_relu(self.conv4(h), 0.2)
        h = self.dropout(h.view(h.size(0), -1))

        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -15, 15)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class Decoder(nn.Module):
    def __init__(self, latent_dim=128, output_channels=3):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 512 * 16 * 16)

        # 对称的解码器结构
        self.deconv1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)  # 16->32
        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)  # 32->64
        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)  # 64->128
        self.deconv4 = nn.ConvTranspose2d(64, output_channels, 4, stride=2, padding=1)  # 128->256

        self.dropout = nn.Dropout(0.2)

    def forward(self, z):
        h = F.leaky_relu(self.fc(z), 0.2)
        h = self.dropout(h.view(h.size(0), 512, 16, 16))

        h = F.leaky_relu(self.deconv1(h), 0.2)
        h = F.leaky_relu(self.deconv2(h), 0.2)
        h = F.leaky_relu(self.deconv3(h), 0.2)
        x_hat = torch.sigmoid(self.deconv4(h))
        return x_hat


# 改进的损失函数
def improved_vae_loss(x, x_hat, mu, logvar, beta=0.1):  # 降低KL权重
    # 使用MSE而不是BCE可能更适合RGB图像
    recon_loss = F.mse_loss(x_hat, x, reduction='mean')

    # 也可以试试BCE
    # x_hat_clamped = torch.clamp(x_hat, 1e-6, 1-1e-6)
    # recon_loss = F.binary_cross_entropy(x_hat_clamped, x, reduction='mean')

    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


