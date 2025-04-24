## 分支修改 (Forked Modification)
修改了 [SwinJSCC](https://github.com/semcomm/SwinJSCC) 以便仅在CPU设备上部署。（如果需要在支持CUDA的设备上运行CPU模式，请修改```net```文件夹下的```device```变量。）

通过CPU运行： ```python main-test.py```

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

参数说明： C 表示通道数量或速率，这意味着在```embed_dim[-1]```处会保存前C个特征图最大值通道用于传输。
