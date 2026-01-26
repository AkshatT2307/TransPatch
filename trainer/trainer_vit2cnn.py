from torch.cuda.amp import autocast
from transformers import AutoConfig, SegformerForSemanticSegmentation
# assumes available in your repo
from pretrained_models.PIDNet.model import get_pred_model
from torch.optim.lr_scheduler import ExponentialLR
from patch.create import Patch
from metrics.loss import PatchLoss
from metrics.performance import SegmentationMetric
from dataset.cityscapes import Cityscapes
from tqdm.auto import tqdm
import numpy as np
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch
import sys
import time
import datetime
import random
import copy
from typing import Tuple
original_sys_path = sys.path.copy()
sys.path.append("/kaggle/working/adversarial-patch-transferability/")


# Surrogate CNN (optional: for grad alignment)

# Hugging Face SegFormer (ViT-backbone)

# Restore original sys.path
sys.path = original_sys_path


class PatchTrainer:
    """
    ViT-target patch training with Model Parallelism across 2 GPUs:
      - GPU 0 (cuda:0): SegFormer (Target Model)
      - GPU 1 (cuda:1): PIDNet (Surrogate), Patch Parameters, Optimizer, All Losses
    
    Features:
      - Attention Hijack (ViT)
      - Boundary Disruption (clean vs patched edge divergence)
      - Frequency Shaping (encourage mid/high frequency energy)
      - Optional Surrogate-CNN Gradient Alignment (2nd-order)
      - EOT, TV, PGD/Adam

    Loss (minimize):
      total = -L_attack
              + tv_w * TV
              + attn_w * L_attnHijack
              + boundary_w * L_boundary      (returns negative to maximize edge divergence)
              + freq_w * L_freq              (returns negative to maximize ring energy)
              + grad_align_w * L_grad_align  (cosine-based; needs surrogate)
    """

    def __init__(self, config, main_logger):
        self.cfg = config
        self.log = main_logger
        
        # ============ DEVICE INITIALIZATION ============
        # GPU 0: SegFormer only
        # GPU 1: PIDNet, Patch, Optimizer, Losses
        self.device0 = torch.device("cuda:0")
        self.device1 = torch.device("cuda:1")
        
        self.log.info(f"[Multi-GPU] Device 0 (SegFormer): {self.device0}")
        self.log.info(f"[Multi-GPU] Device 1 (PIDNet/Patch/Losses): {self.device1}")

        # ---------------- Dataloaders ----------------
        cityscape_train = Cityscapes(
            root=config.dataset.root,
            list_path=config.dataset.train,
            num_classes=config.dataset.num_classes,
            multi_scale=config.train.multi_scale,
            flip=config.train.flip,
            ignore_label=config.train.ignore_label,
            base_size=config.train.base_size,
            crop_size=(config.train.height, config.train.width),
            scale_factor=config.train.scale_factor,
        )
        cityscape_val = Cityscapes(
            root=config.dataset.root,
            list_path=config.dataset.val,
            num_classes=config.dataset.num_classes,
            multi_scale=False,
            flip=False,
            ignore_label=config.train.ignore_label,
            base_size=config.test.base_size,
            crop_size=(config.test.height, config.test.width),
        )
        self.train_dl = torch.utils.data.DataLoader(
            cityscape_train, batch_size=config.train.batch_size,
            shuffle=config.train.shuffle, num_workers=config.train.num_workers,
            pin_memory=config.train.pin_memory, drop_last=config.train.drop_last)
        self.val_dl = torch.utils.data.DataLoader(
            cityscape_val, batch_size=config.test.batch_size,
            shuffle=False, num_workers=config.test.num_workers,
            pin_memory=config.test.pin_memory, drop_last=config.test.drop_last)

        self.iters_per_epoch = len(self.train_dl)
        self.start_epoch = config.train.start_epoch
        self.end_epoch = config.train.end_epoch
        self.total_epochs = self.end_epoch - self.start_epoch
        assert self.total_epochs > 0

        # ---------------- SegFormer (ViT target) on GPU 0 ----------------
        hf_name = getattr(config.model, "hf_name",
                          "nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
        hf_cfg = AutoConfig.from_pretrained(
            hf_name, num_labels=config.dataset.num_classes,
            output_attentions=True, output_hidden_states=False
        )
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            hf_name, config=hf_cfg)
        self.model.to(self.device0).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.log.info(f"[Multi-GPU] SegFormer loaded on {self.device0}")

        # ---------------- Optional Surrogate CNN on GPU 1 ----------------
        self.use_surrogate = bool(
            getattr(getattr(config, "surrogate", object()), "enable", False))
        self.surrogate = None
        if self.use_surrogate:
            s_name = getattr(config.surrogate, "name", "pidnet_s")
            try:
                self.surrogate = get_pred_model(s_name, num_classes=config.dataset.num_classes)
                
                # Load pretrained weights from Kaggle
                if '_s' in s_name:
                    ckpt = torch.load('/kaggle/input/pidnet-s-weights/PIDNet_S_Cityscapes_test.pt', 
                                     map_location=self.device1)
                elif '_m' in s_name:
                    ckpt = torch.load('/kaggle/input/pidnet-m-weights/PIDNet_M_Cityscapes_test.pt', 
                                     map_location=self.device1)
                else:  # _l
                    ckpt = torch.load('/kaggle/input/pidnet-l-weights/PIDNet_L_Cityscapes_test.pt', 
                                     map_location=self.device1)
                
                if 'state_dict' in ckpt:
                    ckpt = ckpt['state_dict']
                model_dict = self.surrogate.state_dict()
                ckpt = {k[6:]: v for k, v in ckpt.items() if k[6:] in model_dict.keys()}  # strip 'model.' prefix
                self.surrogate.load_state_dict(ckpt)
                
                self.surrogate.to(self.device1).eval()
                for p in self.surrogate.parameters(): 
                    p.requires_grad_(False)
                self.log.info(f"[Multi-GPU] Surrogate CNN ({s_name}) loaded on {self.device1}")
            except Exception as e:
                self.log.info(
                    f"[Surrogate] Failed to load '{s_name}': {e}. Disabling grad alignment.")
                self.use_surrogate = False

        # ---------------- Losses / patch / optimizer on GPU 1 ----------------
        self.criterion = PatchLoss(config, main_logger)

        self.S = config.patch.size
        self.lr = config.optimizer.init_lr
        self.use_pgd = bool(getattr(config.optimizer, "use_pgd", False))
        self.pgd_steps = getattr(config.optimizer, "pgd_steps", 7)
        self.pgd_alpha = getattr(config.optimizer, "pgd_alpha", 2.0/255.0)

        # tanh-param patch initialized on GPU 1
        self.patch_param = self.init_lowfreq_tanh(
            (3, self.S, self.S), cutoff=0.2).to(self.device1)
        self.log.info(f"[Multi-GPU] Patch parameters initialized on {self.device1}")
        
        if not self.use_pgd:
            self.opt = torch.optim.Adam([self.patch_param], lr=self.lr)
            self.log.info(f"[Multi-GPU] Adam optimizer created on {self.device1}")
        else:
            self.opt = None
            self.log.info(f"[Multi-GPU] Using PGD optimization")

        self.lr_scheduler = config.optimizer.exponentiallr
        self.gamma = config.optimizer.exponentiallr_gamma
        self.scheduler = ExponentialLR(self.opt, gamma=self.gamma) if (
            self.lr_scheduler and self.opt) else None

        # weights
        self.tv_weight = getattr(
            getattr(config, "loss", object()), "tv_weight", 1e-4)
        self.attn_w = getattr(
            getattr(config, "loss", object()), "attn_hijack_w", 0.10)
        self.boundary_w = getattr(
            getattr(config, "loss", object()), "boundary_w", 0.20)
        self.freq_w = getattr(
            getattr(config, "loss", object()), "freq_w", 0.05)
        self.grad_align_w = getattr(
            getattr(config, "loss", object()), "grad_align_w", 0.10)
        # frequency ring
        self.r_low = getattr(getattr(config, "freq", object()), "r_low", 0.12)
        self.r_high = getattr(
            getattr(config, "freq", object()), "r_high", 0.45)

        self.metric = SegmentationMetric(config)
        self.log_every = config.train.log_per_iters

        self.apply_patch = Patch(config).apply_patch
        self.ignore_index = config.train.ignore_label
        self.num_classes = config.dataset.num_classes

        # --- Patch placement options ---
        self.loc = getattr(getattr(config, "patch", object()),
                           "loc", "class")
        
        if (self.loc == "class"):
            self.class_id = getattr(config.patch, "class_id", 5)
            self.class_entropy_bias = bool(getattr(
                getattr(config, "patch", object()), "class_entropy_bias", True))
            self.class_topk_frac = float(getattr(
                getattr(config, "patch", object()), "class_topk_frac", 0.20))
            self.class_dilate = int(getattr(
                getattr(config, "patch", object()), "class_dilate", 5))
            self.mask_patch_labels = bool(getattr(getattr(config, "patch", object(
            )), "mask_patch_labels", True))

    def _softmax_entropy(self, logits, dim=1, eps=1e-8):
        """Compute entropy on GPU 1"""
        p = torch.softmax(logits, dim=dim).clamp_min(eps)
        return -(p * p.log()).sum(dim=dim)  # (B,H,W)

    def _dilate_mask(self, mask_2d, k=5):
        """Dilate mask on GPU 1"""
        if k <= 1:
            return mask_2d
        m = mask_2d.float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        m = F.max_pool2d(m, kernel_size=k, stride=1, padding=k//2)
        return (m[0, 0] > 0.5)

    def _choose_patch_topleft_from_mask(self, mask_2d, S, entropy_2d=None, topk_frac=0.2):
        """Choose patch location on GPU 1"""
        H, W = mask_2d.shape
        valid = mask_2d.clone()
        if S > 1:
            valid[:S//2, :] = False
            valid[H-(S-S//2):, :] = False
            valid[:, :S//2] = False
            valid[:, W-(S-S//2):] = False
        idx = valid.nonzero(as_tuple=False)  # (K,2): [y,x]
        if idx.numel() == 0:
            return None
        if entropy_2d is not None:
            ent = entropy_2d[idx[:, 0], idx[:, 1]]
            k = max(1, int(topk_frac * idx.size(0)))
            _, topk_inds = torch.topk(ent, k, largest=True)
            idx = idx[topk_inds]
        yx = idx[torch.randint(0, idx.size(0), (1,), device=idx.device).item()]
        yc, xc = int(yx[0]), int(yx[1])
        y0 = max(0, min(yc - S//2, H - S))
        x0 = max(0, min(xc - S//2, W - S))
        return y0, x0

    def _paste_patch(self, img_bchw, patch_3chw, y0, x0):
        """Paste patch on GPU 1"""
        out = img_bchw.clone()
        S = patch_3chw.shape[-1]
        out[:, :, y0:y0+S, x0:x0+S] = patch_3chw[None]
        return out

    # ---------------- Patch parametrization on GPU 1 ----------------
    def get_patch(self):
        """Get patch from parameters on GPU 1"""
        return 0.5 * (torch.tanh(self.patch_param) + 1.0) * 0.999

    def init_lowfreq_tanh(self, shape, cutoff=0.2):
        """Initialize patch parameters on GPU 1"""
        C, H, W = shape
        spec = torch.randn(C, H, W, dtype=torch.complex64, device=self.device1)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=self.device1),
                                torch.linspace(-1, 1, W, device=self.device1), indexing="ij")
        rad = (xx**2 + yy**2).sqrt()
        spec = spec * (rad <= cutoff)
        img = torch.fft.ifft2(spec).real
        img = (img - img.amin(dim=(-2, -1), keepdim=True))
        img = img / (img.amax(dim=(-2, -1), keepdim=True) -
                     img.amin(dim=(-2, -1), keepdim=True) + 1e-8)
        z = (img*2 - 1).clamp(-0.999, 0.999)
        param = torch.atanh(z).detach()
        param.requires_grad_(True)
        return param

    # ---------------- EOT on patch (GPU 1) ----------------
    def eot_patch(self, patch_3chw):
        """Apply EOT transformations on GPU 1"""
        angle = random.uniform(-20, 20)
        scale = random.uniform(0.85, 1.15)
        shear = [random.uniform(-5, 5), random.uniform(-5, 5)]
        out = TF.affine(patch_3chw, angle=angle, translate=[
                        0, 0], scale=scale, shear=shear)
        out = TF.adjust_brightness(out, random.uniform(0.85, 1.15))
        out = TF.adjust_contrast(out,  random.uniform(0.85, 1.15))
        return out.clamp(0, 1)

    # ---------------- Regularizers (GPU 1) ----------------
    def tv_loss(self, x):
        """Total variation loss on GPU 1"""
        if x.dim() == 3:
            return (x[:, 1:, :]-x[:, :-1, :]).abs().mean() + (x[:, :, 1:]-x[:, :, :-1]).abs().mean()
        elif x.dim() == 4:
            return (x[:, :, 1:, :]-x[:, :, :-1, :]).abs().mean() + (x[:, :, :, 1:]-x[:, :, :, :-1]).abs().mean()
        else:
            raise ValueError("tv_loss expects 3D/4D")

    @torch.no_grad()
    def _estimate_patch_mask(self, clean_img, patched_img, thresh=1e-3):
        """Estimate patch mask on GPU 1"""
        diff = (patched_img - clean_img).abs().sum(dim=1, keepdim=True)  # (B,1,H,W)
        return (diff > thresh).float()

    def saliency_hijack_loss(self, logits_adv, patched_image, patch_mask) -> torch.Tensor:
        """
        Fallback when attentions are unusable (GPU 1)
        Maximize gradient magnitude inside the patch vs outside.
        """
        probs = torch.softmax(logits_adv, dim=1)
        entropy = -(probs * (probs.clamp_min(1e-8)).log()).sum(dim=1).mean()

        if not patched_image.requires_grad:
            patched_image.requires_grad_(True)

        grad = torch.autograd.grad(
            entropy, patched_image, create_graph=True, retain_graph=True, allow_unused=False
        )[0]  # (B,3,H,W)

        gmag = grad.abs().mean(dim=1, keepdim=True)  # (B,1,H,W)

        inside = (gmag * patch_mask).mean()
        outside = (gmag * (1.0 - patch_mask)).mean()
        return -(inside - outside)

    ## -------------------------- ViT CNN alignment (GPU 1) ------------------- #
    def logit_agreement_loss(self, vit_logits, cnn_logits):
        """
        First-order surrogate shaping on GPU 1
        Maximize cosine similarity between ViT and CNN logits
        """
        v = vit_logits.flatten(1)
        c = cnn_logits.flatten(1)
        v = F.normalize(v, dim=1)
        c = F.normalize(c, dim=1)
        return - (v * c).sum(dim=1).mean()

    def kl_align(self, p_logits, q_logits, T=1.0):
        """Alternative first-order option on GPU 1: KL(p || q)"""
        p = F.log_softmax(p_logits / T, dim=1)
        q = F.softmax(q_logits / T, dim=1)
        return F.kl_div(p, q, reduction='batchmean') * (T * T)

    def patch_entropy_loss(self, logits, patch_mask):
        """Optional attention-free hijack on GPU 1"""
        probs = torch.softmax(logits, dim=1)
        ent = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        return - (ent * patch_mask).mean()

    def attn_hijack_loss(self, attentions, patch_mask, H: int, W: int) -> torch.Tensor:
        """
        Robust attention hijack on GPU 1
        Try to maximize attention mass to patch tokens
        """
        if (attentions is None) or (len(attentions) == 0):
            return torch.zeros((), device=self.device1)

        B = patch_mask.size(0)
        strides = [4, 8, 16, 32]
        hijack_sum, used = 0.0, 0

        for att in attentions:
            if att is None or not torch.is_tensor(att) or att.ndim != 4:
                continue
            B_a, n_heads, Nq, Nk = att.shape
            if (B_a != B) or (Nq != Nk) or (Nq <= 0):
                continue

            stride = None
            for s in strides:
                if (H // s) * (W // s) == Nq:
                    stride = s
                    break
            if stride is None:
                root = int(round(Nq ** 0.5))
                if root * root != Nq:
                    continue
                h_s = w_s = root
            else:
                h_s, w_s = H // stride, W // stride

            pmask = F.interpolate(patch_mask, size=(h_s, w_s), mode="nearest").view(B, -1)
            att_mean = att.mean(dim=1)

            batch_loss, valid = 0.0, 0
            for b in range(B):
                cols = pmask[b] > 0.5
                if cols.sum() == 0:
                    continue
                mass_to_patch = att_mean[b][:, cols].mean()
                batch_loss += (-mass_to_patch)
                valid += 1

            if valid > 0:
                hijack_sum += (batch_loss / valid)
                used += 1

        if used == 0:
            return torch.zeros((), device=self.device1)
        return hijack_sum / used

    def boundary_disruption_loss(self, clean_logits, adv_logits) -> torch.Tensor:
        """Boundary disruption loss on GPU 1"""
        with torch.no_grad():
            clean = clean_logits.argmax(dim=1, keepdim=True).float()
        adv = adv_logits.argmax(dim=1, keepdim=True).float()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          device=self.device1, dtype=adv.dtype).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                          device=self.device1, dtype=adv.dtype).view(1, 1, 3, 3)
        e_c = F.conv2d(clean, kx, padding=1).abs() + F.conv2d(clean, ky, padding=1).abs()
        e_a = F.conv2d(adv,   kx, padding=1).abs() + F.conv2d(adv,   ky, padding=1).abs()
        return - F.l1_loss(e_a, e_c)

    def freq_shaping_loss(self, patch_3chw) -> torch.Tensor:
        """Frequency shaping loss on GPU 1"""
        C, H, W = patch_3chw.shape
        x = patch_3chw - patch_3chw.mean(dim=(-2, -1), keepdim=True)
        spec = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"), dim=(-2, -1))
        mag = spec.abs().mean(dim=0)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=self.device1),
                                torch.linspace(-1, 1, W, device=self.device1), indexing="ij")
        rad = (xx**2 + yy**2).sqrt()
        ring = (rad >= self.r_low) & (rad <= self.r_high)
        if ring.sum() == 0:
            return torch.zeros((), device=self.device1)
        band_energy = mag[ring].mean()
        return - band_energy

    # ---------------- Forwards ----------------
    def segformer_forward(self, img_4bhwc, out_size=None):
        """
        SegFormer forward on GPU 0
        Input must be on GPU 0, outputs returned on GPU 0
        """
        # Ensure input is on GPU 0
        if img_4bhwc.device != self.device0:
            img_4bhwc = img_4bhwc.to(self.device0)
            
        out = self.model(img_4bhwc, output_attentions=True)
        logits = out.logits  # (B,C,h,w) on GPU 0
        
        if out_size is not None and logits.shape[-2:] != out_size:
            logits = F.interpolate(logits, size=out_size,
                                   mode="bilinear", align_corners=False)
        
        # Return on GPU 0 (will be moved to GPU 1 by caller)
        return logits, out.attentions

    def _pick_highres_4d(self, tensors):
        """Pick the 4D tensor with the largest H*W"""
        best = None
        best_area = -1
        for t in tensors:
            if torch.is_tensor(t) and t.ndim == 4:
                h, w = t.shape[-2], t.shape[-1]
                area = int(h) * int(w)
                if area > best_area:
                    best = t
                    best_area = area
        return best

    def _extract_logits(self, out):
        """Extract logits from various model output formats"""
        if torch.is_tensor(out):
            return out

        if hasattr(out, "logits") and torch.is_tensor(out.logits):
            return out.logits

        if isinstance(out, dict):
            for k in ("logits", "out", "main_out", "pred", "aux", "output"):
                v = out.get(k, None)
                if torch.is_tensor(v):
                    return v
            return self._pick_highres_4d([v for v in out.values() if torch.is_tensor(v)])

        if isinstance(out, (list, tuple)):
            return self._pick_highres_4d(list(out))

        return None

    def surrogate_forward_logits(self, img_4bhwc, target_size):
        """
        Surrogate forward on GPU 1
        Input must be on GPU 1, output returned on GPU 1
        """
        if self.surrogate is None:
            return None

        # Ensure input is on GPU 1
        if img_4bhwc.device != self.device1:
            img_4bhwc = img_4bhwc.to(self.device1)

        out = self.surrogate(img_4bhwc)
        logits = self._extract_logits(out)
        
        if logits is None:
            return None

        if logits.ndim == 3:
            logits = logits.unsqueeze(1)
        elif logits.ndim != 4:
            return None

        if logits.shape[-2:] != target_size:
            logits = F.interpolate(
                logits, size=target_size, mode="bilinear", align_corners=False)
        
        return logits

    # ---------------- Train ----------------
    def train(self):
        desired_total = int(getattr(self.cfg.train, "total_epochs", 20))
        self.end_epoch = self.start_epoch + desired_total
        start_epoch, end_epoch = self.start_epoch, self.end_epoch
        total_epochs = end_epoch - start_epoch
        switch_epoch = start_epoch + (total_epochs // 2)

        start_time = time.time()
        self.log.info(
            f"Start training | Total Epochs: {total_epochs} "
            f"(Stage-1: {start_epoch}–{switch_epoch-1}, Stage-2(JS): {switch_epoch}–{end_epoch-1}) | "
            f"Iterations/epoch: {self.iters_per_epoch}"
        )

        max_batches = int(getattr(self.cfg.train, "max_batches_per_epoch", 300))
        max_epochs = int(getattr(self.cfg.train, "max_epochs", 10))
        self.end_epoch = min(self.end_epoch, self.start_epoch + max_epochs)
        self.log.info(
            f"[Limiter] max_batches_per_epoch={max_batches}, max_epochs={max_epochs}")
        surrogate_every = int(getattr(self.cfg.train, "surrogate_every", 4))

        IoU_over_epochs = []
        H, W = self.cfg.train.height, self.cfg.train.width

        for ep in range(start_epoch, end_epoch):
            self.metric.reset()
            if(ep == switch_epoch):
                patch_stage1 = self.get_patch().detach().cpu()

            use_stage1 = (ep < switch_epoch)
            stage = "Stage-1" if use_stage1 else "Stage-2(JS)"
            self.log.info(f"Epoch {ep}: using {stage}")
            epoch_iter_limit = min(self.iters_per_epoch, max_batches) if max_batches else self.iters_per_epoch
            pbar = tqdm(total=epoch_iter_limit,
                        desc=f"Epoch {ep}/{end_epoch-1}", leave=False, dynamic_ncols=True)

            cum_attack_loss = 0.0
            cum_attn_loss = 0.0
            cum_tv_loss = 0.0
            cum_bound_loss = 0.0
            cum_freq_loss = 0.0
            cum_ga_loss = 0.0
            iters_run = 0
            epoch_start = time.time()

            for it, batch in enumerate(self.train_dl):
                if it >= epoch_iter_limit:
                    break
                    
                # ============ MOVE DATA TO GPU 1 (preprocessing device) ============
                image, true_label, _, _, _ = batch
                image = image.to(self.device1)
                true_label = true_label.to(self.device1).long()

                # ============ PATCH PREPARATION ON GPU 1 ============
                base_patch = self.get_patch()  # Already on GPU 1
                patch = self.eot_patch(base_patch)  # EOT on GPU 1
                S = patch.shape[-1]
                target_hw = true_label.shape[-2:]
                patched_imgs = []
                patched_label = true_label.clone().long()

                if (self.loc == "class"):
                    # Downscale for SegFormer
                    vit_ds = float(getattr(self.cfg.train, "vit_downscale", 0.75))
                    if vit_ds < 1.0:
                        ds_hw = (int(image.shape[-2] * vit_ds), int(image.shape[-1] * vit_ds))
                        image_ds = F.interpolate(image, size=ds_hw, mode="bilinear", align_corners=False)
                    else:
                        image_ds = image

                    # ============ SEGFORMER FORWARD ON GPU 0 ============
                    with torch.no_grad(), autocast(dtype=torch.float16):
                        # Move to GPU 0 for forward pass
                        clean_logits_gpu0, _ = self.segformer_forward(image_ds, out_size=target_hw)
                        # Move back to GPU 1 for processing
                        clean_logits = clean_logits_gpu0.to(self.device1)

                    # Build class mask on GPU 1
                    clean_pred = clean_logits.argmax(dim=1)
                    class_mask = (clean_pred == self.class_id)

                    ent = self._softmax_entropy(clean_logits) if self.class_entropy_bias else None

                    if self.class_dilate > 1:
                        class_mask = torch.stack(
                            [self._dilate_mask(m, self.class_dilate) for m in class_mask], dim=0)

                    # Paste patch on GPU 1
                    for b in range(image.size(0)):
                        yx = self._choose_patch_topleft_from_mask(
                            class_mask[b], S,
                            entropy_2d=(ent[b] if ent is not None else None),
                            topk_frac=self.class_topk_frac
                        )
                        if yx is None:
                            Htot, Wtot = target_hw
                            y0 = random.randint(0, max(0, Htot - S))
                            x0 = random.randint(0, max(0, Wtot - S))
                        else:
                            y0, x0 = yx

                        patched_imgs.append(self._paste_patch(image[b:b+1], patch, y0, x0))

                        if self.mask_patch_labels:
                            patched_label[b, y0:y0+S, x0:x0+S] = self.ignore_index

                elif (self.loc == "center"):
                    Htot, Wtot = target_hw
                    y0 = (Htot-self.S)//2
                    x0 = (Wtot-self.S)//2

                    for b in range(image.size(0)):
                        patched_imgs.append(self._paste_patch(image[b:b+1], patch, y0, x0))

                        if self.mask_patch_labels:
                            patched_label[b, y0:y0+S, x0:x0+S] = self.ignore_index

                patched_image = torch.cat(patched_imgs, dim=0)  # On GPU 1

                # ============ DOWNSCALE FOR SEGFORMER ============
                vit_ds = float(getattr(self.cfg.train, "vit_downscale", 0.75))
                if vit_ds < 1.0:
                    ds_hw = (int(image.shape[-2] * vit_ds), int(image.shape[-1] * vit_ds))
                    image_ds = F.interpolate(image, size=ds_hw, mode="bilinear", align_corners=False)
                    patched_image_ds = F.interpolate(patched_image, size=ds_hw, mode="bilinear", align_corners=False)
                else:
                    image_ds, patched_image_ds = image, patched_image

                # ============ SEGFORMER FORWARD ON GPU 0, RESULTS TO GPU 1 ============
                with autocast(dtype=torch.float16):
                    # Forward on GPU 0
                    logits_adv_gpu0, atts_adv_gpu0 = self.segformer_forward(patched_image_ds, out_size=target_hw)
                    # Move to GPU 1 for loss computation
                    logits_adv = logits_adv_gpu0.to(self.device1)
                    atts_adv = [att.to(self.device1) if att is not None else None for att in atts_adv_gpu0]
                    
                with torch.no_grad(), autocast(dtype=torch.float16):
                    # Forward on GPU 0
                    logits_clean_gpu0, _ = self.segformer_forward(image_ds, out_size=target_hw)
                    # Move to GPU 1
                    logits_clean = logits_clean_gpu0.to(self.device1)

                # ============ ALL LOSSES ON GPU 1 ============
                if use_stage1:
                    attack_loss = self.criterion.compute_loss_transegpgd_stage1(
                        logits_adv, patched_label, logits_clean
                    )
                else:
                    attack_loss = self.criterion.compute_loss_transegpgd_stage2_js(
                        logits_adv, patched_label, logits_clean
                    )

                # Regularizers on GPU 1
                with torch.no_grad():
                    patch_mask = self._estimate_patch_mask(image, patched_image)

                ah_loss = self.attn_hijack_loss(atts_adv, patch_mask, H, W)

                if ah_loss.numel() == 0 or (ah_loss.detach().abs() < 1e-12):
                    patched_image.requires_grad_(True)
                    ah_loss = self.saliency_hijack_loss(logits_adv, patched_image, patch_mask)

                tv = self.tv_loss(base_patch)
                b_loss = self.boundary_disruption_loss(logits_clean, logits_adv)
                f_loss = self.freq_shaping_loss(base_patch)

                # Surrogate alignment on GPU 1
                ga_loss = torch.zeros((), device=self.device1)
                use_sur_now = (self.use_surrogate and self.grad_align_w > 0.0 and
                               (surrogate_every <= 1 or (it % surrogate_every == 0)))
                if use_sur_now:
                    sur_logits = self.surrogate_forward_logits(patched_image, target_size=target_hw)
                    if sur_logits is not None:
                        ga_loss = self.logit_agreement_loss(logits_adv.detach(), sur_logits)

                # Total loss on GPU 1
                total = (-attack_loss) \
                    + self.tv_weight * tv \
                    + self.attn_w * ah_loss \
                    + self.boundary_w * b_loss \
                    + self.freq_w * f_loss \
                    + self.grad_align_w * ga_loss

                # ============ OPTIMIZATION ON GPU 1 ============
                if not self.use_pgd:
                    self.model.zero_grad(set_to_none=True)
                    if self.surrogate is not None:
                        self.surrogate.zero_grad(set_to_none=True)
                    if self.patch_param.grad is not None:
                        self.patch_param.grad.zero_()
                    total.backward()
                    if self.opt is not None:
                        self.opt.step()
                    if self.scheduler:
                        self.scheduler.step()
                else:
                    # PGD loop
                    K, alpha = self.pgd_steps, self.pgd_alpha
                    for _ in range(K):
                        base_patch = self.get_patch()
                        patch = self.eot_patch(base_patch)
                        patched_image, patched_label = self.apply_patch(image, true_label, patch)
                        patched_label = patched_label.long()

                        target_hw = patched_label.shape[-2:]

                        vit_ds = float(getattr(self.cfg.train, "vit_downscale", 0.75))
                        if vit_ds < 1.0:
                            ds_hw = (int(image.shape[-2] * vit_ds), int(image.shape[-1] * vit_ds))
                            image_ds = F.interpolate(image, size=ds_hw, mode="bilinear", align_corners=False)
                            patched_image_ds = F.interpolate(patched_image, size=ds_hw, mode="bilinear", align_corners=False)
                        else:
                            image_ds, patched_image_ds = image, patched_image

                        # Forward on GPU 0, move to GPU 1
                        logits_adv_gpu0, atts_adv_gpu0 = self.segformer_forward(patched_image_ds, out_size=target_hw)
                        logits_adv = logits_adv_gpu0.to(self.device1)
                        atts_adv = [att.to(self.device1) if att is not None else None for att in atts_adv_gpu0]
                        
                        with torch.no_grad():
                            logits_clean_gpu0, _ = self.segformer_forward(image_ds, out_size=target_hw)
                            logits_clean = logits_clean_gpu0.to(self.device1)

                        if use_stage1:
                            attack_loss = self.criterion.compute_loss_transegpgd_stage1(
                                logits_adv, patched_label, logits_clean)
                        else:
                            attack_loss = self.criterion.compute_loss_transegpgd_stage2_js(
                                logits_adv, patched_label, logits_clean)

                        with torch.no_grad():
                            patch_mask = self._estimate_patch_mask(image, patched_image)

                        ah_loss = self.attn_hijack_loss(atts_adv, patch_mask, H, W)

                        if ah_loss.numel() == 0 or (ah_loss.detach().abs() < 1e-12):
                            patched_image.requires_grad_(True)
                            ah_loss = self.saliency_hijack_loss(logits_adv, patched_image, patch_mask)
                            
                        tv = self.tv_loss(base_patch)
                        b_loss = self.boundary_disruption_loss(logits_clean, logits_adv)
                        f_loss = self.freq_shaping_loss(base_patch)

                        ga_loss = torch.zeros((), device=self.device1)
                        if use_sur_now:
                            sur_logits = self.surrogate_forward_logits(patched_image, target_size=target_hw)
                            if sur_logits is not None:
                                ga_loss = self.logit_agreement_loss(logits_adv.detach(), sur_logits)

                        total_inner = (-attack_loss) \
                            + self.tv_weight * tv \
                            + self.attn_w * ah_loss \
                            + self.boundary_w * b_loss \
                            + self.freq_w * f_loss \
                            + self.grad_align_w * ga_loss

                        self.model.zero_grad(set_to_none=True)
                        if self.surrogate is not None:
                            self.surrogate.zero_grad(set_to_none=True)
                        if self.patch_param.grad is not None:
                            self.patch_param.grad.zero_()
                        total_inner.backward()

                        with torch.no_grad():
                            self.patch_param += alpha * self.patch_param.grad.sign()
                            vis = self.get_patch().clamp(0, 1)
                            z = (vis*2 - 1).clamp(-0.999, 0.999)
                            self.patch_param.copy_(torch.atanh(z))

                # ============ METRICS ON GPU 1 ============
                self.metric.update(logits_adv, patched_label)
                _, mIoU = self.metric.get()
                atk_val = attack_loss.item()
                cum_attack_loss += atk_val
                cum_attn_loss += ah_loss.item()
                cum_tv_loss += tv.item()
                cum_bound_loss += b_loss.item()
                cum_freq_loss += f_loss.item()
                cum_ga_loss += ga_loss.item()
                iters_run += 1

                if pbar is not None:
                    pbar.update(1)

            pixAcc, meanIoU = self.metric.get()
            if pbar is not None:
                pbar.close()

            avg_attack = cum_attack_loss / max(1, iters_run)
            avg_attn = cum_attn_loss / max(1, iters_run)
            avg_tv = cum_tv_loss / max(1, iters_run)
            avg_bound = cum_bound_loss / max(1, iters_run)
            avg_freq = cum_freq_loss / max(1, iters_run)
            avg_ga = cum_ga_loss / max(1, iters_run)
            epoch_elapsed = time.time() - epoch_start
            epochs_done = (ep - start_epoch) + 1
            epochs_left = max(0, (end_epoch - ep - 1))
            eta_sec = (epoch_elapsed * epochs_left) if epochs_done == 0 else (
                (time.time() - start_time) / epochs_done) * epochs_left
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            elapsed_str = str(datetime.timedelta(seconds=int(epoch_elapsed)))
            self.log.info("-"*100)
            self.log.info(
                f"Epoch {ep}/{end_epoch} | {stage} | "
                f"Atk:{avg_attack:.4f} | Attn:{avg_attn:.4f} | TV:{avg_tv:.4f} | "
                f"Bound:{avg_bound:.4f} | Freq:{avg_freq:.4f} | GA:{avg_ga:.4f} | "
                f"mIoU:{meanIoU:.4f} | pixAcc:{pixAcc:.4f} | "
                f"EpochTime:{elapsed_str} | ETA:{eta_str}"
            )
            IoU_over_epochs.append(self.metric.get(full=True))

        return self.get_patch().detach(), np.array(IoU_over_epochs), patch_stage1