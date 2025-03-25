import argparse
import cv2
import glob
import numpy as np
import os
import torch
import time  # 添加时间测量模块
from thop import profile
from torchcam.methods import GradCAM
from torchvision.transforms import ToTensor
from mysr.archs.My_arch import MyArch
#python inference/inference_mysr.py        --input datasets/Set14/LR_bicubic/X2 --output results/mysr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa: E251
        'experiments/MyArch_x2_Large_del5_archived_20250226_203047/models/net_g_195000.pth'  # noqa: E501
    )
    parser.add_argument('--input', type=str, default='datasets/Urban100/LR_bicubic/X2', help='input test image folder')
    parser.add_argument('--output', type=str, default='x2Large_13', help='output folder')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = MyArch(num_in_ch=3, num_out_ch=3, num_feat=180, num_block=36, upscale=2, dygroup=12, factor=45)
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    # 计算 FLOPs
    input_shape = (1, 3, 640, 360)  # 模型输入张量的形状 (批大小, 通道数, 高度, 宽度)  输出统一为1280x720
    # x2 input_shape = (1, 3, 640, 360)
    # x3 input_shape = (1, 3, 432, 240)
    # x4 input_shape = (1, 3, 320, 180)

    flops, params = profile(model, inputs=(torch.randn(input_shape).to(device),), verbose=False)
    print(f'\nModel FLOPs: {flops / 1e9:.2f} G, Model Parameters: {params / 1e6:.2f} M\n')

    os.makedirs(args.output, exist_ok=True)
    latency_list = []
    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing', idx, imgname)
        # read image
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)
        # inference
        try:
            # 计算 Latency
            start_time = time.time()
            with torch.no_grad():
                output = model(img)
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # 转换为毫秒
            latency_list.append(latency)
        except Exception as error:
            print('Error', error, imgname)
        else:
            # save image
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
            cv2.imwrite(os.path.join(args.output, f'{imgname}_MyArch.png'), output)
    # 打印平均 Latency
    if latency_list:
        average_latency = sum(latency_list) / len(latency_list)
        print(f'\nAverage Latency per Image: {average_latency:.2f} ms\n')

if __name__ == '__main__':
    main()
