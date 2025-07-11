# BLEDNet：Bio-inspired lightweight neural network for edge detection

This repository contains the PyTorch implementation for "[BLEDNet：Bio-inspired lightweight neural network for edge detection](https://doi.org/10.1016/j.engappai.2023.106530)" 

by 
Zhengqiao Luo,
Chuan Lin ,
Fuzhang Li ,
Yongcai Pan


## Generating edge images
```bash
python generate.py --custompath /path/to/data --save_path ./results # --invert # generate inverse edge map
```

## Acknowledgements
In the process of building the code, we also consulted the following open-source repositories:<br>
- [Piotr's Structured Forest matlab toolbox](https://github.com/pdollar/edges)
- [HED Implementation](https://github.com/xwjabc/hed)
- [Original HED](https://github.com/s9xie/hed)
- [PiDiNet](https://github.com/hellozhuo/pidinet)<br>
- [RCF](https://github.com/yun-liu/rcf)<br>



## Citation
~~~
@article{luo2023blednet,
  title={Blednet: bio-inspired lightweight neural network for edge detection},
  author={Luo, Zhengqiao and Lin, Chuan and Li, Fuzhang and Pan, Yongcai},
  journal={Engineering Applications of Artificial Intelligence},
  volume={124},
  pages={106530},
  year={2023},
  publisher={Elsevier}
}
~~~