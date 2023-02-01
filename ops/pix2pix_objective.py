from .gan_objective import GANObjective
import torch


class Pix2pixObjective(GANObjective):
    def __init__(self, lambda_rec, **kwargs):
        self.lambda_rec = lambda_rec

        self.gan_loss = torch.nn.BCEWithLogitsLoss()
        self.l1_loss = torch.nn.L1Loss()

    def practice_discrimination(self, generator, discriminator, data, device):
        batch_src_cpu, batch_tgt_cpu = data
        batch_size, channels = batch_src_cpu.size(0), batch_src_cpu.size(1)
        real_translation_cpu = torch.cat((batch_src_cpu, batch_tgt_cpu), dim=1)
        real_translation = real_translation_cpu.to(device)

        scores_for_real = discriminator(real_translation)
        loss_real = self.gan_loss(scores_for_real, torch.ones_like(scores_for_real, requires_grad=False, device=device))

        fake_translation = torch.cat((real_translation[:,:channels],generator(real_translation[:,:channels])), dim=1)
        scored_for_fake = discriminator(fake_translation)
        loss_fake = self.gan_loss(scored_for_fake, torch.zeros_like(scored_for_fake, requires_grad=False, device=device))
        loss_total = (loss_real + loss_fake) / 2

        return loss_total, {
            "real": loss_real.item(),
            "fake": loss_fake.item(),
            "total": loss_total.item()
        }

    def practice_generation(self, generator, discriminator, data, device):
        batch_src_cpu, batch_tgt_cpu = data
        batch_src, batch_tgt = batch_src_cpu.to(device), batch_tgt_cpu.to(device)

        fake_batch = generator(batch_src)
        fake_translation = torch.cat((batch_src, fake_batch), dim=1)

        scores_for_fake = discriminator(fake_translation)
        loss_fake = self.gan_loss(scores_for_fake, torch.ones_like(scores_for_fake, requires_grad=False, device=device))
        loss_reconstruction = self.l1_loss(fake_batch, batch_tgt)

        loss_total = loss_fake + self.lambda_rec * loss_reconstruction
        return loss_total, {
            "fake": loss_fake.item(),
            "reconstruction": loss_reconstruction.item(),
            "total": loss_total.item()
        }
