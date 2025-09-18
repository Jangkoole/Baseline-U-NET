import torch
from networks.VAE import ImprovedVAE,improved_vae_loss
import torch.nn.functional as F

# 训练代码
def train_improved_vae():
    # 数据加载
    from torch.utils.data import DataLoader
    from IRM2 import read_data, create_domain_dataloaders

    images, masks, domain = read_data()
    train_domains = [0, 1, 2]
    train_dataloaders = create_domain_dataloaders(
        images, masks, domain, train_domains, batch_size=16  # 增大batch size
    )
    train_dataset_all = torch.utils.data.ConcatDataset([dl.dataset for dl in train_dataloaders.values()])
    train_loader = DataLoader(train_dataset_all, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = ImprovedVAE(input_channels=3, latent_dim=128).to(device)

    # 使用不同的优化器设置
    optimizer = torch.optim.Adam(vae.parameters(), lr=2e-4, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # 预热训练 - 先只训练重建损失
    print("预热阶段: 只优化重建损失")
    for epoch in range(10):
        vae.train()
        total_recon = 0
        for batch in train_loader:
            images = batch['image'].to(device)
            images = torch.clamp(images, 0, 1)

            optimizer.zero_grad()
            x_hat, mu, logvar = vae(images)

            # 只用重建损失
            recon_loss = F.mse_loss(x_hat, images, reduction='mean')
            recon_loss.backward()

            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()
            total_recon += recon_loss.item()

        print(f"预热第{epoch + 1}轮, 重建损失: {total_recon / len(train_loader):.4f}")

    # 正式训练
    print("正式训练阶段")
    for epoch in range(100):
        vae.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0

        for batch in train_loader:
            images = batch['image'].to(device)
            images = torch.clamp(images, 0, 1)

            optimizer.zero_grad()
            x_hat, mu, logvar = vae(images)

            # 逐渐增加KL权重
            beta = min(0.1, epoch * 0.001)
            loss, recon_loss, kl_loss = improved_vae_loss(images, x_hat, mu, logvar, beta)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        avg_recon = total_recon / len(train_loader)
        avg_kl = total_kl / len(train_loader)

        print(f"第{epoch + 1}轮, 总损失: {avg_loss:.4f}, 重建: {avg_recon:.4f}, KL: {avg_kl:.4f}")

        # 保存最佳模型
        if epoch == 0 or avg_recon < best_recon:
            best_recon = avg_recon
            torch.save(vae.state_dict(), 'best_vae_model.pth')


if __name__ == "__main__":
    train_improved_vae()