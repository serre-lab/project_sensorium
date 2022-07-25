import torch
import torch.nn as nn
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def visualize_map(map):
    map = map.detach().numpy()
    plt.imshow(map)


def get_gabor(l_size, l_div, n_ori, aspect_ratio):
    """generate the gabor filters

    Args
    ----
        l_size: float
            gabor sizes
        l_div: floats
            normalization value to be used
        n_ori: type integer
            number of orientations
        aspect_ratio: type float
            gabor aspect ratio

    Returns
    -------
        gabor: type nparray
            gabor filter

    Example
    -------
        aspect_ratio  = 0.3
        l_gabor_size = 7
        l_div        = 4.0
        n_ori         = 4
        get_gabor(l_gabor_size, l_div, n_ori, aspect_ratio)

    """

    la = l_size * 2 / l_div
    si = la * 0.8
    gs = l_size

    # TODO: inverse the axes in the begining so I don't need to do swap them back
    # thetas for all gabor orientations
    th = np.array(range(n_ori)) * np.pi / n_ori + np.pi / 2.
    th = th[sp.newaxis, sp.newaxis, :]
    hgs = (gs - 1) / 2.
    yy, xx = sp.mgrid[-hgs: hgs + 1, -hgs: hgs + 1]
    xx = xx[:, :, sp.newaxis];
    yy = yy[:, :, sp.newaxis]

    x = xx * np.cos(th) - yy * np.sin(th)
    y = xx * np.sin(th) + yy * np.cos(th)

    filt = np.exp(-(x ** 2 + (aspect_ratio * y) ** 2) / (2 * si ** 2)) * np.cos(2 * np.pi * x / la)
    filt[np.sqrt(x ** 2 + y ** 2) > gs / 2.] = 0

    # gabor normalization (following cns hmaxgray package)
    for ori in range(n_ori):
        filt[:, :, ori] -= filt[:, :, ori].mean()
        filt_norm = fastnorm(filt[:, :, ori])
        if filt_norm != 0: filt[:, :, ori] /= filt_norm
    filt_c = np.array(filt, dtype='float32').swapaxes(0, 2).swapaxes(1, 2)

    filt_c = torch.Tensor(filt_c)
    filt_c = filt_c.view(n_ori, 1, gs, gs)
    # filt_c = filt_c.repeat((1, 3, 1, 1))
    # filt_c = filt_c.repeat((1, 3, 1, 1))

    return filt_c


def fastnorm(in_arr):
    arr_norm = np.dot(in_arr.ravel(), in_arr.ravel()).sum() ** (1. / 2.)

    return arr_norm


def get_sp_kernel_sizes_C(scales, num_scales_pooled, scale_stride):
    '''
    Recursive function to find the right relative kernel sizes for the spatial pooling performed in a C layer.
    The right relative kernel size is the average of the scales that will be pooled. E.g, if scale 7 and 9 will be
    pooled, the kernel size for the spatial pool is 8 x 8

    Parameters
    ----------
    scales
    num_scales_pooled
    scale_stride

    Returns
    -------
    list of sp_kernel_size

    '''

    if len(scales) < num_scales_pooled:
        return []
    else:
        average = int(sum(scales[0:num_scales_pooled]) / len(scales[0:num_scales_pooled]))
        return [average] + get_sp_kernel_sizes_C(scales[scale_stride::], num_scales_pooled, scale_stride)


def pad_to_size(a, size):
    current_size = (a.shape[-2], a.shape[-1])
    total_pad_h = size[0] - current_size[0]
    pad_top = total_pad_h // 2
    pad_bottom = total_pad_h - pad_top

    total_pad_w = size[1] - current_size[1]
    pad_left = total_pad_w // 2
    pad_right = total_pad_w - pad_left

    a = nn.functional.pad(a, (pad_left, pad_right, pad_top, pad_bottom))

    return a


class S1(nn.Module):
    # TODO: make more flexible such that forward will accept any number of tensors
    def __init__(self, scales, n_ori, padding, trainable_filters, divs):

        super(S1, self).__init__()
        assert (len(scales) == len(divs))
        self.scales = scales
        self.divs = divs

        for scale, div in zip(self.scales, self.divs):
            setattr(self, f's_{scale}', nn.Conv2d(1, n_ori, scale, padding=padding))
            s1_cell = getattr(self, f's_{scale}')
            gabor_filter = get_gabor(l_size=scale, l_div=div, n_ori=n_ori, aspect_ratio=0.3)
            # print('gabor_filter : ',gabor_filter.shape)
            s1_cell.weight = nn.Parameter(gabor_filter, requires_grad=trainable_filters)

            # For normalization
            setattr(self, f's_uniform_{scale}', nn.Conv2d(1, n_ori, scale, bias=False))
            s1_uniform = getattr(self, f's_uniform_{scale}')
            nn.init.constant_(s1_uniform.weight, 1)
            for param in s1_uniform.parameters():
                param.requires_grad = False

    def forward(self, x):
        s1_maps = []
        # Loop over scales, normalizing.
        for scale in self.scales:
            s1_cell = getattr(self, f's_{scale}')
            s1_map = torch.abs(s1_cell(x))  # adding absolute value

            s1_unorm = getattr(self, f's_uniform_{scale}')
            s1_unorm = torch.sqrt(s1_unorm(x** 2))
            s1_unorm.data[s1_unorm == 0] = 1  # To avoid divide by zero
            s1_map /= s1_unorm

            s1_maps.append(s1_map)

        # Padding (to get s1_maps in same size)
        # TODO: figure out if we'll ever be in a scenario where we don't need the same padding left/right or top/down
        # Or if the size difference is an odd number
        ori_size = (x.shape[-2], x.shape[-1])
        for i, s1_map in enumerate(s1_maps):
            s1_maps[i] = pad_to_size(s1_map, ori_size)
        s1_maps = torch.stack(s1_maps, dim=4)

        return s1_maps


class C(nn.Module):
    # TODO: make more flexible such that forward will accept any number of tensors
    def __init__(self, sp_kernel_size=list(range(8, 38, 4)), sp_stride_factor=None, n_in_sbands=None,
                 num_scales_pooled=2, scale_stride=2,image_subsample_factor=1):

        super(C, self).__init__()
        self.sp_kernel_size = sp_kernel_size
        self.num_scales_pooled = num_scales_pooled
        self.sp_stride_factor = sp_stride_factor
        self.scale_stride = scale_stride
        self.n_in_sbands = n_in_sbands
        self.n_out_sbands = int(((n_in_sbands - self.num_scales_pooled) / self.scale_stride) + 1)
        self.img_subsample = image_subsample_factor 
        # Checking
        if type(self.sp_kernel_size) == int:
            # Apply the same sp_kernel_size everywhere
            self.sp_kernel_size = [self.sp_kernel_size] * self.n_out_sbands

        if len(self.sp_kernel_size) != self.n_out_sbands:
            raise ValueError('wrong number of sp_kernel_sizes provided')

        # Compute strides
        if self.sp_stride_factor is None:
            self.sp_stride = [1] * self.n_out_sbands
        else:
            self.sp_stride = [int(np.ceil(self.sp_stride_factor * kernel_size)) for kernel_size in self.sp_kernel_size]

    def forward(self, x):
        # TODO - make this whole section more memory efficient

        # Evaluate input
        if x.ndim != 5:
            raise ValueError('expecting 5D input: BXCxHxWxS, where S is number of scalebands')

        # Group scale bands to be pooled together
        # TODO: deal with the scenario in which x cannot be split even
        groups = []
        for i in range(self.num_scales_pooled):
            groups.append(x[:, :, :, :, i::self.scale_stride][:, :, :, :, 0:self.n_out_sbands])
        x = torch.stack(groups, dim=5)

        # Maxpool over scale groups
        x, _ = torch.max(x, dim=5)

        # Maxpool over positions
        # TODO: deal with rectangular images if needed
        c_maps = []
        ori_size = x.shape[2:4]
        for i, (kernel_size, stride) in enumerate(zip(self.sp_kernel_size, self.sp_stride)):
            to_append = x[:, :, :, :, i]
            if kernel_size >= 0:
                to_append = nn.functional.max_pool2d(to_append, kernel_size, stride)
                to_append = pad_to_size(to_append, (int(ori_size[0] /self.img_subsample), int(ori_size[1] / self.img_subsample)))
            else:
                # Negative kernel_size indicating we want global
                to_append = nn.functional.max_pool2d(to_append, to_append.shape[-2], 1)
            # TODO: think about whether the resolution gets too small here
            # to_append = nn.functional.interpolate(to_append, orig_size)  # resizing to ensure all maps are same size
            c_maps.append(to_append)

        c_maps = torch.stack(c_maps, dim=4)

        return c_maps


class HMAX(nn.Module):
    def __init__(self,
                 s1_scales=range(3, 39, 2),
                 s1_divs=np.arange(4, 3.1, -0.05),
                 n_ori=8,
                 s1_trainable_filters=False,
                 n_scales = 9):
        super(HMAX, self).__init__()

        # A few settings
        s1_scales = s1_scales[:n_scales]
        s1_divs = s1_divs[:n_scales]

        self.s1_scales = s1_scales
        self.s1_divs = s1_divs
        self.n_ori = n_ori
        self.s1_trainable_filters = s1_trainable_filters

        self.c_scale_stride = 1
        self.c_num_scales_pooled = 2

        self.c1_scale_stride = self.c_scale_stride
        self.c1_num_scales_pooled = self.c_num_scales_pooled
        self.c1_sp_kernel_sizes = get_sp_kernel_sizes_C(self.s1_scales, self.c1_num_scales_pooled, self.c1_scale_stride)

        # Feature extractors (in the order of the table in Figure 1)
        self.s1 = S1(scales=self.s1_scales, n_ori=n_ori, padding='valid', trainable_filters=s1_trainable_filters,
                     divs=self.s1_divs)
        self.c1 = C(sp_kernel_size=self.c1_sp_kernel_sizes, sp_stride_factor=0.125, n_in_sbands=len(s1_scales),
                    num_scales_pooled=self.c1_num_scales_pooled, scale_stride=self.c1_scale_stride)
        

    def forward(self, x):
        s1_maps = self.s1(x)
        c1_maps = self.c1(s1_maps)  # BxCxHxWxS with S number of scales

        # print('c1_maps before : ',c1_maps.shape)

        c1_maps = c1_maps.permute(0,1,4,2,3)
        c1_maps = c1_maps.reshape(c1_maps.shape[0], -1, c1_maps.shape[3], c1_maps.shape[4])

        # print('c1_maps after : ',c1_maps.shape)

        return c1_maps
