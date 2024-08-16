from torchvision.transforms import Resize, CenterCrop, Compose


class Resize_with_K(object):

    def __init__(self, size):
        self.resize = Resize(size)

    def __call__(self, arg):
        # process image
        x = arg["img"]
        w_src, h_src = x.size
        x = self.resize(x)
        w_dst, h_dst = x.size

        # process intrinsic
        K = arg["K"]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        fx, fy, cx, cy, *_ = resize_intrinsic_param(
            fx,
            fy,
            cx,
            cy,
            w_src,
            h_src,
            w_dst,
            h_dst,
        )
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = fx, fy, cx, cy

        arg["img"] = x
        arg["K"] = K

        return arg


class CenterCrop_with_K(object):

    def __init__(self, size):
        self.ccrop = CenterCrop(size)

    def __call__(self, arg):
        x = arg["img"]
        w_src, h_src = x.size
        x = self.ccrop(x)
        w_dst, h_dst = x.size

        w_diff_half = (w_src - w_dst) // 2
        h_diff_half = (h_src - h_dst) // 2

        x_min = w_diff_half
        x_max = w_src - w_diff_half

        y_min = h_diff_half
        y_max = h_src - h_diff_half

        K = arg["K"]
        cx, cy = K[0, 2], K[1, 2]
        cx, cy, w, h = crop_intrinsic_param(
            cx,
            cy,
            w_src,
            h_src,
            x_min,
            x_max,
            y_min,
            y_max,
        )

        # assert w == w_dst and h == h_dst
        K[0, 2], K[1, 2] = cx, cy

        arg["img"] = x
        arg["K"] = K

        return arg


def resize_intrinsic_param(fx, fy, cx, cy, w_src, h_src, w_dst, h_dst):

    fx = fx * w_dst / w_src
    fy = fy * h_dst / h_src

    cx = cx * w_dst / w_src
    cy = cy * h_dst / h_src

    return fx, fy, cx, cy, w_dst, h_dst


def crop_intrinsic_param(cx, cy, w, h, x_min, x_max, y_min, y_max):

    w_hat = w - (x_min + (w - x_max))
    h_hat = h - (y_min + (h - y_max))
    cx_hat = cx - x_min
    cy_hat = cy - y_min

    assert w_hat > 0 and h_hat > 0 and cx_hat > 0 and cy_hat > 0

    return cx_hat, cy_hat, w_hat, h_hat
