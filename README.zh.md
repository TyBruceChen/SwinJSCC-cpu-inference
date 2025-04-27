## 分支修改 (Forked Modification)
修改了 [SwinJSCC](https://github.com/semcomm/SwinJSCC) 以便仅在CPU设备上部署。（如果需要在支持CUDA的设备上运行CPU模式，请修改```net```文件夹下的```device```变量。）

通过CPU运行：```python main-test.py```; 在[Google Drive](https://drive.google.com/drive/folders/1_EouRY4yYvMCtamX2ReBzEd5YBQbyesc?usp=sharing)下载由官方提供的预训练模型; 修改```args.model_path``` and ```args.model_size```

文件夹结构：
```
  -main-test.py (新增)
  -data
    --datasets.py (修改)
  -datasets
    --cpu_inference_set (将测试图片放在此处)
  -results
    --recon (恢复的图片会存储在此处)
  -models (预训练模型存储在此处)
  -其他文件 (未修改)
```

参数说明： 
- ```C```： 表示通道数量或速率，这意味着在```embed_dim[-1]```处会保存前C个特征图最大值通道用于传输。
- 压缩率: ```(1/4) ** (stage数量)``` times ```C/3 (C is from embed_dim[-1])```.

<b>测试展示 :</b>
![residential roof snow removal](https://github.com/user-attachments/assets/ae42ebfa-95bc-4320-8c50-d6cbbda92b8f)
```
Message: '【Channel】: Built awgn channel, SNR 10 dB.'
Arguments: ()
Transmitted Vector shape as:torch.Size([1, 6912, 320]) with mask selecting 96.0 channels
2025-04-24 13:36:07,467 - INFO] Time 9.673 | CBR 0.0625 (0.0625) | SNR 10.0 | PSNR 25.471 (25.471) | MSSSIM 0.945 
(0.945)
SNR: [[10.0]]
CBR: [[0.0625]]
PSNR: [[25.470993041992188]]
MS-SSIM: [[0.9454551339149475]]
Finish Test!
```
