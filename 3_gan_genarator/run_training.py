import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


# =================== 配置参数 ===================
class Config:
    # 数据路径
    tapian_dir = 'dataset/tapian'  # 拓片文件夹
    muben_dir = 'dataset/muben'  # 摹本文件夹

    # 训练参数
    batch_size = 16
    num_epochs = 200
    learning_rate = 0.0002
    beta1 = 0.5
    beta2 = 0.999
    lambda_l1 = 100  # L1损失权重

    # 模型参数
    ngf = 64  # 生成器特征数
    ndf = 64  # 判别器特征数

    # 图像参数
    image_size = 512
    input_channels = 1  # 灰度图
    output_channels = 1  # 灰度图

    # 保存参数
    save_interval = 10  # 每10个epoch保存一次
    sample_interval = 5  # 每5个epoch生成样本
    log_interval = 50  # 每50个batch记录一次

    # 输出目录
    output_dir = f'oracle_bone_gan_output_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    sample_dir = os.path.join(output_dir, 'samples')
    log_dir = os.path.join(output_dir, 'logs')

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 4

    # AMP
    use_amp = True


# =================== 数据集 ===================
class OracleBoneDataset(Dataset):
    def __init__(self, tapian_dir, muben_dir, transform=None):
        self.tapian_dir = tapian_dir
        self.muben_dir = muben_dir
        self.transform = transform

        # 获取所有图像文件名
        self.images = []
        for filename in os.listdir(tapian_dir):
            tapian_path = os.path.join(tapian_dir, filename)
            muben_path = os.path.join(muben_dir, filename)
            if os.path.exists(muben_path):
                self.images.append(filename)

        print(f"Found {len(self.images)} image pairs")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        filename = self.images[idx]

        # 加载拓片和摹本
        tapian_path = os.path.join(self.tapian_dir, filename)
        muben_path = os.path.join(self.muben_dir, filename)

        tapian = Image.open(tapian_path).convert('L')
        muben = Image.open(muben_path).convert('L')

        if self.transform:
            tapian = self.transform(tapian)
            muben = self.transform(muben)

        return tapian, muben


# =================== 生成器 (U-Net) ===================
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, ngf=64):
        super(UNetGenerator, self).__init__()

        # 编码器
        self.e1 = self.encoder_block(in_channels, ngf, normalize=False)
        self.e2 = self.encoder_block(ngf, ngf * 2)
        self.e3 = self.encoder_block(ngf * 2, ngf * 4)
        self.e4 = self.encoder_block(ngf * 4, ngf * 8)
        self.e5 = self.encoder_block(ngf * 8, ngf * 8)
        self.e6 = self.encoder_block(ngf * 8, ngf * 8)
        self.e7 = self.encoder_block(ngf * 8, ngf * 8)
        self.e8 = self.encoder_block(ngf * 8, ngf * 8, normalize=False)

        # 解码器
        self.d1 = self.decoder_block(ngf * 8, ngf * 8, dropout=True)
        self.d2 = self.decoder_block(ngf * 16, ngf * 8, dropout=True)
        self.d3 = self.decoder_block(ngf * 16, ngf * 8, dropout=True)
        self.d4 = self.decoder_block(ngf * 16, ngf * 8)
        self.d5 = self.decoder_block(ngf * 16, ngf * 4)
        self.d6 = self.decoder_block(ngf * 8, ngf * 2)
        self.d7 = self.decoder_block(ngf * 4, ngf)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def encoder_block(self, in_channels, out_channels, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, True))
        return nn.Sequential(*layers)

    def decoder_block(self, in_channels, out_channels, dropout=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(True))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 编码
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)

        # 解码 with skip connections
        d1 = self.d1(e8)
        d1 = torch.cat([d1, e7], 1)
        d2 = self.d2(d1)
        d2 = torch.cat([d2, e6], 1)
        d3 = self.d3(d2)
        d3 = torch.cat([d3, e5], 1)
        d4 = self.d4(d3)
        d4 = torch.cat([d4, e4], 1)
        d5 = self.d5(d4)
        d5 = torch.cat([d5, e3], 1)
        d6 = self.d6(d5)
        d6 = torch.cat([d6, e2], 1)
        d7 = self.d7(d6)
        d7 = torch.cat([d7, e1], 1)

        return self.final(d7)


# =================== 判别器 (PatchGAN) ===================
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=2, ndf=64):
        super(PatchGANDiscriminator, self).__init__()

        self.model = nn.Sequential(
            # 输入是拓片和摹本的拼接
            nn.Conv2d(in_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 1)
        )

    def forward(self, input, target):
        x = torch.cat([input, target], 1)
        return self.model(x)


# =================== 训练器 ===================
class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device

        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

        # 数据变换
        self.transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # 归一化到[-1, 1]
        ])

        # 数据集和数据加载器
        self.dataset = OracleBoneDataset(
            config.tapian_dir,
            config.muben_dir,
            self.transform
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )

        # 模型
        self.generator = UNetGenerator(
            config.input_channels,
            config.output_channels,
            config.ngf
        ).to(self.device)

        self.discriminator = PatchGANDiscriminator(
            config.input_channels + config.output_channels,
            config.ndf
        ).to(self.device)

        # 优化器
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2)
        )

        # 损失函数
        self.criterion_GAN = nn.BCEWithLogitsLoss()
        self.criterion_L1 = nn.L1Loss()

        # AMP
        self.scaler_G = GradScaler() if config.use_amp else None
        self.scaler_D = GradScaler() if config.use_amp else None

        # 训练历史
        self.history = {
            'epoch': [],
            'loss_G': [],
            'loss_D': [],
            'loss_GAN': [],
            'loss_L1': []
        }

        # 固定的验证样本
        try:
            self.fixed_input, self.fixed_target = next(iter(self.dataloader))
            self.fixed_input = self.fixed_input[:min(8, len(self.fixed_input))].to(self.device)
            self.fixed_target = self.fixed_target[:min(8, len(self.fixed_target))].to(self.device)
        except StopIteration:
            print("Warning: DataLoader is empty. Cannot fetch fixed samples.")
            self.fixed_input, self.fixed_target = None, None

    def train_epoch(self, epoch):
        self.generator.train()
        self.discriminator.train()

        epoch_loss_G = 0
        epoch_loss_D = 0
        epoch_loss_GAN = 0
        epoch_loss_L1 = 0

        pbar = tqdm(self.dataloader, desc=f'Epoch {epoch}/{self.config.num_epochs}')

        for i, (input_img, target_img) in enumerate(pbar):
            input_img = input_img.to(self.device)
            target_img = target_img.to(self.device)

            # ========== 训练判别器 ==========
            self.optimizer_D.zero_grad()

            if self.config.use_amp:
                with autocast():
                    # 真实图像
                    pred_real = self.discriminator(input_img, target_img)
                    # FIX: Create labels with the same size as the discriminator's output
                    real_label = torch.ones_like(pred_real, device=self.device)
                    loss_D_real = self.criterion_GAN(pred_real, real_label)

                    # 生成图像
                    fake_img = self.generator(input_img)
                    pred_fake = self.discriminator(input_img, fake_img.detach())
                    # FIX: Create labels with the same size as the discriminator's output
                    fake_label = torch.zeros_like(pred_fake, device=self.device)
                    loss_D_fake = self.criterion_GAN(pred_fake, fake_label)

                    loss_D = (loss_D_real + loss_D_fake) * 0.5

                self.scaler_D.scale(loss_D).backward()
                self.scaler_D.step(self.optimizer_D)
                self.scaler_D.update()
            else:
                pred_real = self.discriminator(input_img, target_img)
                real_label = torch.ones_like(pred_real, device=self.device)
                loss_D_real = self.criterion_GAN(pred_real, real_label)

                fake_img = self.generator(input_img)
                pred_fake = self.discriminator(input_img, fake_img.detach())
                fake_label = torch.zeros_like(pred_fake, device=self.device)
                loss_D_fake = self.criterion_GAN(pred_fake, fake_label)

                loss_D = (loss_D_real + loss_D_fake) * 0.5
                loss_D.backward()
                self.optimizer_D.step()

            # ========== 训练生成器 ==========
            self.optimizer_G.zero_grad()

            if self.config.use_amp:
                with autocast():
                    fake_img = self.generator(input_img)
                    pred_fake = self.discriminator(input_img, fake_img)

                    # Generator wants discriminator to think fake is real
                    loss_GAN = self.criterion_GAN(pred_fake, real_label)
                    loss_L1 = self.criterion_L1(fake_img, target_img)
                    loss_G = loss_GAN + self.config.lambda_l1 * loss_L1

                self.scaler_G.scale(loss_G).backward()
                self.scaler_G.step(self.optimizer_G)
                self.scaler_G.update()
            else:
                fake_img = self.generator(input_img)
                pred_fake = self.discriminator(input_img, fake_img)

                loss_GAN = self.criterion_GAN(pred_fake, real_label)
                loss_L1 = self.criterion_L1(fake_img, target_img)
                loss_G = loss_GAN + self.config.lambda_l1 * loss_L1

                loss_G.backward()
                self.optimizer_G.step()

            # 记录损失
            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()
            epoch_loss_GAN += loss_GAN.item()
            epoch_loss_L1 += loss_L1.item()

            # 更新进度条
            if i % 10 == 0:
                pbar.set_postfix({
                    'Loss_G': f'{loss_G.item():.4f}',
                    'Loss_D': f'{loss_D.item():.4f}',
                    'Loss_L1': f'{loss_L1.item():.4f}'
                })

        # 计算平均损失
        n_batches = len(self.dataloader)
        return {
            'loss_G': epoch_loss_G / n_batches,
            'loss_D': epoch_loss_D / n_batches,
            'loss_GAN': epoch_loss_GAN / n_batches,
            'loss_L1': epoch_loss_L1 / n_batches
        }

    def generate_samples(self, epoch):
        """生成并保存样本"""
        if self.fixed_input is None:
            return

        self.generator.eval()
        with torch.no_grad():
            fake_imgs = self.generator(self.fixed_input)

            # 创建对比图: Input | Generated | Target
            comparison = torch.cat([
                self.fixed_input,
                fake_imgs,
                self.fixed_target
            ], dim=3)

            # 反归一化
            comparison = comparison * 0.5 + 0.5

            # 保存图像
            save_path = os.path.join(self.config.sample_dir, f'epoch_{epoch:04d}.png')
            save_image(comparison, save_path, nrow=1)

            print(f"Saved samples to {save_path}")

    def save_checkpoint(self, epoch, losses):
        """保存模型检查点"""
        # Convert Config object to a dictionary for JSON serialization
        config_dict = {k: v for k, v in self.config.__dict__.items() if not k.startswith('__')}

        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'losses': losses,
            'config': config_dict
        }

        path = os.path.join(self.config.checkpoint_dir, f'checkpoint_epoch_{epoch:04d}.pth')
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

        # 保存最新模型
        latest_path = os.path.join(self.config.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)

    def plot_history(self):
        """绘制训练历史"""
        if not self.history['epoch']:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Training History', fontsize=16)

        # 生成器损失
        axes[0, 0].plot(self.history['epoch'], self.history['loss_G'], label='Generator Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Generator Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 判别器损失
        axes[0, 1].plot(self.history['epoch'], self.history['loss_D'], label='Discriminator Loss', color='orange')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Discriminator Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # GAN损失
        axes[1, 0].plot(self.history['epoch'], self.history['loss_GAN'], label='GAN Loss', color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('GAN Loss (Generator)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # L1损失
        axes[1, 1].plot(self.history['epoch'], self.history['loss_L1'], label='L1 Loss', color='red')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('L1 Reconstruction Loss (Generator)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(self.config.log_dir, 'training_history.png'))
        plt.close()

    def save_history(self):
        """保存训练历史到JSON"""
        history_path = os.path.join(self.config.log_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)

    def train(self):
        """主训练循环"""
        print(f"Starting training on {self.device}")
        print(f"Total training images: {len(self.dataset)}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Total epochs: {self.config.num_epochs}")
        print("-" * 50)

        for epoch in range(1, self.config.num_epochs + 1):
            # 训练一个epoch
            losses = self.train_epoch(epoch)

            # 记录历史
            self.history['epoch'].append(epoch)
            self.history['loss_G'].append(losses['loss_G'])
            self.history['loss_D'].append(losses['loss_D'])
            self.history['loss_GAN'].append(losses['loss_GAN'])
            self.history['loss_L1'].append(losses['loss_L1'])

            # 打印损失
            print(f"\nEpoch [{epoch}/{self.config.num_epochs}] "
                  f"Loss_G: {losses['loss_G']:.4f}, "
                  f"Loss_D: {losses['loss_D']:.4f}, "
                  f"Loss_GAN: {losses['loss_GAN']:.4f}, "
                  f"Loss_L1: {losses['loss_L1']:.4f}")

            # 生成样本
            if epoch % self.config.sample_interval == 0:
                self.generate_samples(epoch)

            # 保存检查点
            if epoch % self.config.save_interval == 0 or epoch == self.config.num_epochs:
                self.save_checkpoint(epoch, losses)
                self.plot_history()
                self.save_history()

        print("\nTraining completed!")
        self.plot_history()
        self.save_history()


# =================== 推理函数 ===================
def inference(model_path, input_image_path, output_path, device='cuda'):
    """使用训练好的模型进行推理"""
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)

    # FIX: 从checkpoint加载配置以确保模型和变换匹配
    config_dict = checkpoint['config']
    image_size = config_dict.get('image_size', 512)
    input_channels = config_dict.get('input_channels', 1)
    output_channels = config_dict.get('output_channels', 1)
    ngf = config_dict.get('ngf', 64)

    generator = UNetGenerator(input_channels, output_channels, ngf).to(device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

    # 数据变换 (使用保存的配置)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # 加载图像
    image = Image.open(input_image_path).convert('L')
    image = transform(image).unsqueeze(0).to(device)

    # 生成
    with torch.no_grad():
        output = generator(image)
        output = output * 0.5 + 0.5  # 反归一化

    # 保存结果
    save_image(output, output_path)
    print(f"Generated image saved to {output_path}")


# =================== 主函数 ===================
def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 创建配置
    config = Config()

    # 确保数据目录存在
    if not os.path.exists(config.tapian_dir) or not os.path.exists(config.muben_dir):
        print(f"Error: Data directories '{config.tapian_dir}' or '{config.muben_dir}' not found.")
        print("Please create these directories and place your image pairs in them.")
        # As a fallback for testing, create dummy directories and images
        print("Creating dummy data for demonstration purposes...")
        os.makedirs(config.tapian_dir, exist_ok=True)
        os.makedirs(config.muben_dir, exist_ok=True)
        for i in range(20):
            dummy_img = Image.new('L', (512, 512), color='white')
            dummy_img.save(os.path.join(config.tapian_dir, f'dummy_{i}.png'))
            dummy_img.save(os.path.join(config.muben_dir, f'dummy_{i}.png'))

    # 打印配置信息
    print("Configuration:")
    print("-" * 50)
    for key, value in config.__dict__.items():
        if not key.startswith('__'):
            print(f"{key}: {value}")
    print("-" * 50)

    # 创建训练器并开始训练
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
