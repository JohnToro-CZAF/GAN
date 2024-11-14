# GAN/trainer.py

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import itertools
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, models, dataloader, config, device, evaluation_metric=None):
        self.device = device
        self.config = config
        self.dataloader = dataloader
        self.models = models
        self.evaluation_metric = evaluation_metric
        self._setup()
        self._initialize_losses()

    def _setup(self):
        # Unpack models
        self.G_AB = self.models['G_AB'].to(self.device)
        self.G_BA = self.models['G_BA'].to(self.device)
        self.D_A = self.models['D_A'].to(self.device)
        self.D_B = self.models['D_B'].to(self.device)

        # Initialize optimizers
        lr = self.config['training']['learning_rate']
        beta1 = self.config['training']['beta1']
        self.optimizer_G = optim.Adam(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()),
            lr=lr, betas=(beta1, 0.999)
        )
        self.optimizer_D_A = optim.Adam(self.D_A.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_D_B = optim.Adam(self.D_B.parameters(), lr=lr, betas=(beta1, 0.999))

        # Learning rate schedulers
        self.lr_scheduler_G = lr_scheduler.LambdaLR(
            self.optimizer_G, lr_lambda=self._lambda_rule)
        self.lr_scheduler_D_A = lr_scheduler.LambdaLR(
            self.optimizer_D_A, lr_lambda=self._lambda_rule)
        self.lr_scheduler_D_B = lr_scheduler.LambdaLR(
            self.optimizer_D_B, lr_lambda=self._lambda_rule)

    def _initialize_losses(self):
        # Initialize loss functions
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()

    def _lambda_rule(self, epoch):
        total_epochs = self.config['training']['num_epochs']
        decay_epoch = self.config['training']['decay_epoch']
        lr_l = 1.0 - max(0, epoch - decay_epoch) / (total_epochs - decay_epoch)
        return lr_l

    def train(self):
        num_epochs = self.config['training']['num_epochs']
        for epoch in range(num_epochs):
            self._train_epoch(epoch)
            # Update learning rates
            self.lr_scheduler_G.step()
            self.lr_scheduler_D_A.step()
            self.lr_scheduler_D_B.step()
            # Save models periodically
            if (epoch + 1) % self.config['training']['save_interval'] == 0:
                self._save_models(epoch + 1)
            # Optionally, perform evaluation
            if self.evaluation_metric:
                self._evaluate(epoch)

    def _train_epoch(self, epoch):
        for i, data in enumerate(tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.config['training']['num_epochs']}")):
            self._train_step(data)

    def _train_step_cyclegan(self, data):
        real_A = data['A'].to(self.device)
        real_B = data['B'].to(self.device)

        valid = torch.ones((real_A.size(0), 1, 30, 30), device=self.device)
        fake = torch.zeros((real_A.size(0), 1, 30, 30), device=self.device)

        # ------------------
        #  Train Generators
        # ------------------
        self.optimizer_G.zero_grad()

        # Identity loss
        loss_id_A = self.criterion_identity(self.G_BA(real_A), real_A) * self.config['training']['lambda_identity']
        loss_id_B = self.criterion_identity(self.G_AB(real_B), real_B) * self.config['training']['lambda_identity']

        # GAN loss
        fake_B = self.G_AB(real_A)
        pred_fake = self.D_B(fake_B)
        loss_GAN_AB = self.criterion_GAN(pred_fake, valid)

        fake_A = self.G_BA(real_B)
        pred_fake = self.D_A(fake_A)
        loss_GAN_BA = self.criterion_GAN(pred_fake, valid)

        # Cycle loss
        recov_A = self.G_BA(fake_B)
        loss_cycle_A = self.criterion_cycle(recov_A, real_A) * self.config['training']['lambda_cycle']

        recov_B = self.G_AB(fake_A)
        loss_cycle_B = self.criterion_cycle(recov_B, real_B) * self.config['training']['lambda_cycle']

        # Total loss
        loss_G = loss_id_A + loss_id_B + loss_GAN_AB + loss_GAN_BA + loss_cycle_A + loss_cycle_B
        loss_G.backward()
        self.optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------
        self.optimizer_D_A.zero_grad()

        # Real loss
        pred_real = self.D_A(real_A)
        loss_D_real = self.criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = self.D_A(fake_A.detach())
        loss_D_fake = self.criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()
        self.optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------
        self.optimizer_D_B.zero_grad()

        # Real loss
        pred_real = self.D_B(real_B)
        loss_D_real = self.criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = self.D_B(fake_B.detach())
        loss_D_fake = self.criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()
        self.optimizer_D_B.step()

        # Logging losses (optional)
        if self.config['training']['verbose'] and i % self.config['training']['log_interval'] == 0:
            print(f"Epoch [{epoch+1}/{self.config['training']['num_epochs']}] Batch [{i}/{len(self.dataloader)}] "
                  f"Loss_D: {loss_D_A.item() + loss_D_B.item():.4f}, Loss_G: {loss_G.item():.4f}")

    def _train_step(self, data):
        model_name = self.config['model']['name'].lower()
        if model_name == 'cyclegan':
            self._train_step_cyclegan(data)
        elif model_name == 'pix2pix':
            self._train_step_pix2pix(data)
        elif model_name == 'stargan':
            self._train_step_stargan(data)
        else:
            raise ValueError(f"Model {model_name} not recognized.")

    def _train_step_pix2pix(self, data):
        # Implement training steps for Pix2Pix
        real_A = data['A'].to(self.device)
        real_B = data['B'].to(self.device)

        # Generate fake images
        fake_B = self.G_AB(real_A)

        # Train Discriminator
        self.optimizer_D_B.zero_grad()
        pred_real = self.D_B(real_A, real_B)
        pred_fake = self.D_B(real_A, fake_B.detach())
        loss_D_real = self.criterion_GAN(pred_real, torch.ones_like(pred_real))
        loss_D_fake = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        self.optimizer_D_B.step()

        # Train Generator
        self.optimizer_G.zero_grad()
        pred_fake = self.D_B(real_A, fake_B)
        loss_G_GAN = self.criterion_GAN(pred_fake, torch.ones_like(pred_fake))
        loss_G_L1 = self.criterion_cycle(fake_B, real_B) * self.config['training']['lambda_L1']
        loss_G = loss_G_GAN + loss_G_L1
        loss_G.backward()
        self.optimizer_G.step()


    def _save_models(self, epoch):
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(self.G_AB.state_dict(), f'checkpoints/G_AB_epoch_{epoch}.pth')
        torch.save(self.G_BA.state_dict(), f'checkpoints/G_BA_epoch_{epoch}.pth')
        torch.save(self.D_A.state_dict(), f'checkpoints/D_A_epoch_{epoch}.pth')
        torch.save(self.D_B.state_dict(), f'checkpoints/D_B_epoch_{epoch}.pth')
        print(f"Saved models at epoch {epoch}")

    def _evaluate(self, epoch):
        # Implement evaluation using self.evaluation_metric
        # For example, compute FID on a validation set
        pass
