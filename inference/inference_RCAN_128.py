import argparse
import cv2
import glob
import numpy as np
import os
import torch

from basicsr.models.archs.rcan_arch import RCAN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa: E251
        'experiments/pretrained_models/ESRGAN/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth'  # noqa: E501
    )
    parser.add_argument(
        '--folder',
        type=str,
        default='datasets/Set14/LRbicx4',
        help='input test image folder')
    parser.add_argument(
        '--output_folder',
        type=str,
        default='results/MSRResNet_z_288_15000/',
        help='output test image folder')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = RCAN(num_in_ch=3, num_out_ch=3, num_feat=128, num_group=10, num_block=20, squeeze_factor=16, upscale=3, res_scale=1, img_range=255., rgb_mean=[0.4488, 0.4371, 0.4040])
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    #result_path = 'results/MSRResNet_z_288_15000/'
    result_path = args.output_folder
    os.makedirs(result_path, exist_ok=True)
    for idx, path in enumerate(
            sorted(glob.glob(os.path.join(args.folder, '*')))):
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing', idx, imgname)
        # read image
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]],
                                            (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)
        # inference
        with torch.no_grad():
            output = model(img)
        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        cv2.imwrite(result_path+f'{imgname}.png', output)


if __name__ == '__main__':
    main()
