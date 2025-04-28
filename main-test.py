import torch.optim as optim
from net.network import SwinJSCC
from data.datasets import get_loader
from utils import *
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from datetime import datetime
import torch.nn as nn
import argparse
from loss.distortion import *
import time
import torchvision

class args:
    training = False  # Default values, can be changed manually
    testset = 'test'
    trainset = 'DIV2K'
    distortion_metric = 'MSE'
    model = 'SwinJSCC_w/_SAandRA'
    channel_type = 'awgn'
    C = '96'    # Number of channels in the encoder
    multiple_snr = '10'
    model_size = 'small'
    model_path = 'models/SwinJSCC w- SA&RA/pretrained_SwinJSCC_w_SAandRA_small_AWGN_HRimage_cbr_psnr_snr13.model'
    device_type = 'cpu'

class config():
    seed = 42
    pass_channel = True
    CUDA = True
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    norm = False
    # logger
    filename = 'log'#datetime.now().__str__()[:-7]
    workdir = './history/{}'.format(filename)
    log = workdir + '/Log_{}.log'.format(filename)
    samples = workdir + '/samples'
    models = workdir + '/models'
    logger = None

    # training details
    normalize = False


    image_dims = (3, 256, 256)
    if args.testset == 'test':
        test_data_dir = ["datasets/cpu_inference_set"]

    batch_size = 16
    downsample = 4
    if args.model == 'SwinJSCC_w/o_SAandRA' or args.model == 'SwinJSCC_w/_SA':
        channel_number = int(args.C)
    else:
        channel_number = None

    if args.model_size == 'small':
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 192, 256, 320], depths=[2, 2, 2, 2], num_heads=[4, 6, 8, 10], C=channel_number,
            window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[320, 256, 192, 128], depths=[2, 2, 2, 2], num_heads=[10, 8, 6, 4], C=channel_number,
            window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True
        )
    elif args.model_size == 'base':
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10], C=channel_number,
            window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4], C=channel_number,
            window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
    elif args.model_size =='large':
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 192, 256, 320], depths=[2, 2, 18, 2], num_heads=[4, 6, 8, 10], C=channel_number,
            window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[320, 256, 192, 128], depths=[2, 18, 2, 2], num_heads=[10, 8, 6, 4], C=channel_number,
            window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )

CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).to(args.device_type)

def load_weights(model_path, net):
    
    pretrained = torch.load(model_path, map_location=torch.device(args.device_type))
    net.load_state_dict(pretrained, strict=True)
    del pretrained
    """
    pretrained_dict = torch.load(model_path, map_location=torch.device(args.device_type))
    model_dict = net.state_dict()
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'scalable_adapter' not in k}
    model_dict.update(filtered_dict)
    net.load_state_dict(model_dict)
    del pretrained_dict
    """

def test():
    config.isTrain = False
    net.eval()
    elapsed, psnrs, msssims, snrs, cbrs = [AverageMeter() for _ in range(5)]
    metrics = [elapsed, psnrs, msssims, snrs, cbrs]
    multiple_snr = args.multiple_snr.split(",")
    for i in range(len(multiple_snr)):
        multiple_snr[i] = int(multiple_snr[i])
    channel_number = args.C.split(",")
    for i in range(len(channel_number)):
        channel_number[i] = int(channel_number[i])
    results_snr = np.zeros((len(multiple_snr), len(channel_number)))
    results_cbr = np.zeros((len(multiple_snr), len(channel_number)))
    results_psnr = np.zeros((len(multiple_snr), len(channel_number)))
    results_msssim = np.zeros((len(multiple_snr), len(channel_number)))
    for i, SNR in enumerate(multiple_snr):
        for j, rate in enumerate(channel_number):
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    input, names = batch
                    start_time = time.time()
                    input = input.to(args.device_type) #.cuda()
                    recon_image, CBR, SNR, mse, loss_G = net(input, SNR, rate)
                    torchvision.utils.save_image(recon_image,
                                                    os.path.join("results", f"recon/{names[0]}"))
                    elapsed.update(time.time() - start_time)
                    cbrs.update(CBR)
                    snrs.update(SNR)
                    if mse.item() > 0:
                        psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                        psnrs.update(psnr.item())
                        msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                        msssims.update(msssim)
                        MSSSIM = -10 * np.emath.log10(1 - msssim)
                        print(MSSSIM)
                    else:
                        psnrs.update(100)
                        msssims.update(100)
                    log = (' | '.join([
                        f'Time {elapsed.val:.3f}',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f}',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                    ]))
                    logger.info(log)
            results_snr[i, j] = snrs.avg
            results_cbr[i, j] = cbrs.avg
            results_psnr[i, j] = psnrs.avg
            results_msssim[i, j] = msssims.avg
            for t in metrics:
                t.clear()

    print("SNR: {}".format(results_snr.tolist()))
    print("CBR: {}".format(results_cbr.tolist()))
    print("PSNR: {}".format(results_psnr.tolist()))
    print("MS-SSIM: {}".format(results_msssim.tolist()))
    print("Finish Test!")

if __name__ == '__main__':
    seed_torch()
    logger = logger_configuration(config, save_log=True)
    logger.info(config.__dict__)
    torch.manual_seed(seed=config.seed)
    net = SwinJSCC(args, config)
    #model_path = "models/SwinJSCC w- SA&RA/pretrained_SwinJSCC_w_SAandRA_large_AWGN_HRimage_cbr_psnr_snr13.model"
    load_weights(args.model_path, net)
    net = net.to(args.device_type)#.cuda()
    _train_loader, test_loader = get_loader(args, config)

    if args.training:
        print('This file is for cpu inference only! please use main.py for training!')
    else:
        os.makedirs('results', exist_ok=True,)
        os.makedirs('results/recon', exist_ok=True,)
        test()
