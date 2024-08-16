# TransformCamK

**Transform** **Cam**era Intrinsic(**K**) has operations that handle changes in camera intrinsic values along with image resizing or cropping.
The provided operations are compatible with `torchvision.transforms.Compose`.

## Install

```bash
pip install git+https://github.com/KIMGEONUNG/TransformCamK
```

## Usage

```python
from TransformCamK import Resize_with_K, CenterCrop_with_K
from torchvision.transforms import Compose
from PIL import Image


transform = Compose([
    Resize_with_K(512),
    CenterCrop_with_K(512),
])

W, H = 400, 600
fx, fy = 800, 800
cx, cy = W // 2, H // 2

K = np.array([
    [fx, 0., cx],
    [0., fy, cy],
    [0., 0., 1.],
])

image = Image.open('image/file/path')
rs = transform({"img": image, "K": K})
image_hat = rs["img"]
K_hat = rs["K"]
```
