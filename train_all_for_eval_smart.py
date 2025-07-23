import math
import time
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
import sys
from PIL import Image
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from gaussian_models.utils import *
from tqdm import tqdm
import random
import torchvision.transforms as transforms
import shutil
import glob
from fused_ssim import fused_ssim

Image.MAX_IMAGE_PIXELS = 10000000000

class SimpleTrainer2d:
    """Trains random 2d gaussians to fit an image."""
    def __init__(
        self,
        image_path: Path,
        num_points: int = 2000,
        compression_ratio: int = - 1,
        model_name:str = "GaussianImage_Cholesky",
        iterations:int = 30000,
        model_path = None,
        log_path = "None",
        args = None,
    ):
        self.device = torch.device("cuda")
        self.gt_image = image_path_to_tensor(image_path).to(self.device) # [1, C, H, W]
        print("Image Shape: ", self.gt_image.shape[-2:])
        self.compression_ratio = compression_ratio
        image_path = Path(image_path)
        self.image_name = image_path.stem
        BLOCK_H, BLOCK_W = 16, 16
        self.H, self.W = self.gt_image.shape[2], self.gt_image.shape[3]
        self.iterations = iterations
        self.save_iter_img = args.save_iter_img
        self.data_name = args.data_name
        self.model_name = model_name

        # compute compress ratio
        if compression_ratio > 0:
            # 量化7， 不量化32
            self.num_points = int((self.H * self.W * 3) / (7 * self.compression_ratio))
            self.log_dir = Path(os.path.join(log_path, self.image_name))
            os.makedirs(self.log_dir, exist_ok=True)
            # if self.log_dir.exists() and self.log_dir.is_dir():
            #     shutil.rmtree(self.log_dir)
            self.training_render_dir = Path(os.path.join(self.log_dir, "renders"))
            os.makedirs(self.training_render_dir, exist_ok=True)
        else:
            self.num_points = num_points
            self.log_dir = Path(f"./logs/{args.data_name}/{model_name}/{model_name}_{args.iterations}_{num_points}/{self.image_name}")
            if self.log_dir.exists() and self.log_dir.is_dir():
                shutil.rmtree(self.log_dir)
            self.training_render_dir = Path(f"./logs/{args.data_name}/{model_name}/{model_name}_{args.iterations}_{num_points}/{self.image_name}/renders")
            os.makedirs(self.training_render_dir, exist_ok=True)

        print(f"Num Points: {self.num_points} | Compression Ratio: {self.compression_ratio}")

        # Baselines
        if model_name == "GaussianImage_Cholesky":
            from gaussian_models.gaussianimage_cholesky import GaussianImage_Cholesky
            self.gaussian_model = GaussianImage_Cholesky(loss_type="L2", opt_type="adan", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, lr=args.lr, quantize=False).to(self.device)
        elif model_name == "GaussianImage_RS":
            from gaussian_models.gaussianimage_rs import GaussianImage_RS
            self.gaussian_model = GaussianImage_RS(loss_type="L2", opt_type="adan", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, lr=args.lr, quantize=False).to(self.device)
        elif model_name == "3DGS":
            from gaussian_models.gaussiansplatting_3d import Gaussian3D
            self.gaussian_model = Gaussian3D(loss_type="L2", opt_type="adan", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, sh_degree=args.sh_degree, lr=args.lr).to(self.device)  
        # Image_GS
        elif model_name == "ImageGS_RS":
            from gaussian_models.image_gs_rs import ImageGS_RS
            self.gaussian_model = ImageGS_RS(loss_type="L1+SSIM", opt_type="adam", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, sh_degree=args.sh_degree, lr=args.lr, init_image=self.gt_image).to(self.device)

        elif model_name == "GaussianImage_Cov2D":
            # adopt from LIG
            from gaussian_models.gaussianimage_cov2d import GaussianImage_Cov2D
            self.gaussian_model = GaussianImage_Cov2D(loss_type="L2", opt_type="adam", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, sh_degree=args.sh_degree, lr=args.lr, init_image=self.gt_image).to(self.device)
        
        # Ours
        elif model_name == "GaussianImage_RS_Sample":
            from gaussian_models.gaussianimage_rs_sample import GaussianImage_RS_Sample
            self.gaussian_model = GaussianImage_RS_Sample(loss_type="L2", opt_type="adam", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, sh_degree=args.sh_degree, lr=args.lr, init_image=self.gt_image).to(self.device)
        
        elif model_name == "SmartSplat":
            from gaussian_models.smartsplat import SmartSplat
            self.gaussian_model = SmartSplat(loss_type="L1-SSIM",
                                            opt_type="adam",
                                            num_points=self.num_points,
                                            H=self.H,
                                            W=self.W,
                                            BLOCK_H=BLOCK_H,
                                            BLOCK_W=BLOCK_W, 
                                            device=self.device,
                                            sh_degree=args.sh_degree,
                                            lr=args.lr,
                                            init_image=self.gt_image).to(self.device)
        
        self.logwriter = LogWriter(self.log_dir)

        if model_path is not None:
            print(f"loading model path:{model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            model_dict = self.gaussian_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.gaussian_model.load_state_dict(model_dict)

    def train(self):     
        psnr_list, iter_list = [], []
        progress_bar = tqdm(range(1, self.iterations+1), desc="Training progress")
        best_psnr = 0
        self.gaussian_model.train()
        start_time = time.time()

        for iter in range(0, self.iterations):
            if self.model_name == "ImageGS_RS":
                loss, psnr = self.gaussian_model.train_iter(self.gt_image, iter)
            else:
                loss, psnr = self.gaussian_model.train_iter(self.gt_image)
            if psnr > best_psnr:
                best_psnr = psnr
            # if iter >  1000 and  psnr < 10.:
            #     break
            psnr_list.append(psnr)
            iter_list.append(iter)
            with torch.no_grad():
                if (iter+1) % 10 == 0:
                    progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f}", f"Best PSNR":f"{best_psnr:.{4}f}",})
                    progress_bar.update(10)
                if self.save_iter_img > 0:
                    if iter==0 or (iter + 1) % self.save_iter_img == 0:
                        self.training_test(iter)
                
        end_time = time.time() - start_time
        progress_bar.close()
        psnr_value, ms_ssim_value = self.test()
        with torch.no_grad():
            self.gaussian_model.eval()
            test_start_time = time.time()
            for i in range(100):
                _ = self.gaussian_model()
            test_end_time = (time.time() - test_start_time)/100

        self.logwriter.write("Training Complete in {:.4f}s, Eval time:{:.8f}s, FPS:{:.4f}, Best PSNR: {:.4f}".format(end_time, test_end_time, 1/test_end_time, best_psnr))
        torch.save(self.gaussian_model.state_dict(), self.log_dir / "gaussian_model.pth.tar")
        np.save(self.log_dir / "training.npy", {"iterations": iter_list, "training_psnr": psnr_list, "training_time": end_time, 
        "psnr": psnr_value, "ms-ssim": ms_ssim_value, "rendering_time": test_end_time, "rendering_fps": 1/test_end_time})
        return psnr_value, ms_ssim_value, end_time, test_end_time, 1/test_end_time

    def test(self):
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model()
        mse_loss = F.mse_loss(out["render"].float(), self.gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        # ms_ssim_value = ms_ssim(out["render"].float(), self.gt_image.float(), data_range=1, size_average=True).item()
        # MS-SSIM OOM
        if self.data_name in ["DIV16K"]:
            ms_ssim_value = fused_ssim(out["render"].float(), self.gt_image.float(), train=False).item()
        else:  
            ms_ssim_value = ms_ssim(out["render"].float(), self.gt_image.float(), data_range=1, size_average=True).item()
        
        self.logwriter.write("Test PSNR:{:.4f}, MS_SSIM:{:.6f}".format(psnr, ms_ssim_value))
        if self.save_iter_img > 0:
            transform = transforms.ToPILImage()
            img = transform(out["render"].float().squeeze(0))
            name = self.image_name + "_fitting.png" 
            img.save(str(self.log_dir / name))
        return psnr, ms_ssim_value

    def training_test(self, iter):
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model()
        mse_loss = F.mse_loss(out["render"].float(), self.gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        # ms_ssim_value = ms_ssim(out["render"].float(), self.gt_image.float(), data_range=1, size_average=True).item()

        # MS-SSIM OOM
        if self.data_name in ["DIV16K"]:
            ms_ssim_value = fused_ssim(out["render"].float(), self.gt_image.float(), train=False).item()
        else:
            ms_ssim_value = ms_ssim(out["render"].float(), self.gt_image.float(), data_range=1, size_average=True).item()
        
        self.logwriter.write("Iter: {:d}, Test PSNR:{:.4f}, MS_SSIM:{:.6f}".format(iter, psnr, ms_ssim_value), train=True)
        
        if self.save_iter_img > 0:
            transform = transforms.ToPILImage()
            img = transform(out["render"].float().squeeze(0))
            name = self.image_name + f"_fitting_{iter}.jpg" 
            img.save(str(self.training_render_dir / name))
        return psnr, ms_ssim_value

def image_path_to_tensor(image_path: Path):
    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0) #[1, C, H, W]
    return img_tensor

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, default='./datasets/kodak/', help="Training dataset"
    )
    parser.add_argument(
        "--log_dir", type=str, default='./logs/', help="Training dataset"
    )
    parser.add_argument(
        "--data_name", type=str, default='kodak', help="Training dataset"
    )
    parser.add_argument(
        "--iterations", type=int, default=50000, help="number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--model_name", type=str, default="GaussianImage_Cholesky", help="model selection: GaussianImage_Cholesky, GaussianImage_RS, 3DGS"
    )
    parser.add_argument(
        "--sh_degree", type=int, default=3, help="SH degree (default: %(default)s)"
    )
    parser.add_argument(
        "--compression_ratio", type=int, default=-1, help="Compression Ratio (default: %(default)s)"
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=50000,
        help="2D GS points (default: %(default)s)",
    )
    parser.add_argument("--model_path", type=str, default=None, help="Path to a checkpoint")
    parser.add_argument("--seed", type=float, default=1, help="Set random seed for reproducibility")
    parser.add_argument(
        "--save_iter_img", 
        type=int, 
        default=-1, 
        help="Save intermediate images every N iterations (-1 means disabled, default: -1)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)
    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

    log_path = os.path.join(args.log_dir, args.data_name, args.model_name, f"{args.model_name}_Iter{args.iterations}_CR{args.compression_ratio}")
    logwriter = LogWriter(Path(log_path))
    psnrs, ms_ssims, training_times, eval_times, eval_fpses = [], [], [], [], []
    image_h, image_w = 0, 0

    image_files = sorted(glob.glob(str(Path(args.dataset) / '*.png')))
    image_length = len(image_files)
    print(f"Training with {args.model_name} Gaussian Image Model.......")
    print(f"Founding {image_length} Images in {args.data_name} Dataset .........")
    for i, img_path in enumerate(image_files):
        torch.cuda.empty_cache()
        image_path = Path(img_path)  # 转换为 Path 对象
        print(f"Processing image {i+1}: {image_path.name}")
        trainer = SimpleTrainer2d(image_path=image_path,
                                num_points=args.num_points, 
                                compression_ratio=args.compression_ratio,
                                iterations=args.iterations,
                                model_name=args.model_name,
                                model_path=args.model_path,
                                log_path=log_path,
                                args=args)
        
        psnr, ms_ssim, training_time, eval_time, eval_fps = trainer.train()
        
        psnrs.append(psnr)
        ms_ssims.append(ms_ssim)
        training_times.append(training_time) 
        eval_times.append(eval_time)
        eval_fpses.append(eval_fps)
        image_h += trainer.H
        image_w += trainer.W
        image_name = image_path.stem
        logwriter.write("{}: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
            image_name, trainer.H, trainer.W, psnr, ms_ssim, training_time, eval_time, eval_fps))

    avg_psnr = torch.tensor(psnrs).mean().item()
    avg_ms_ssim = torch.tensor(ms_ssims).mean().item()
    avg_training_time = torch.tensor(training_times).mean().item()
    avg_eval_time = torch.tensor(eval_times).mean().item()
    avg_eval_fps = torch.tensor(eval_fpses).mean().item()
    avg_h = image_h//image_length
    avg_w = image_w//image_length

    logwriter.write("Average: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
        avg_h, avg_w, avg_psnr, avg_ms_ssim, avg_training_time, avg_eval_time, avg_eval_fps))    

if __name__ == "__main__":
    main(sys.argv[1:])
