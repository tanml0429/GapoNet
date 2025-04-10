
from GapoNet.apis.yolov8_api import (
    DetectionModel, BaseModel, v8DetectionLoss,
    yaml_model_load, initialize_weights,
    make_divisible, colorstr, scale_img
    )
from GapoNet.apis.yolov8_api import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    OBB,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    AConv,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Pose,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
    Segment,
    WorldDetect,
    v10Detect,
)
from GapoNet.apis.lymo_blocks import (
    C2fCA,
    C2fST,
    C2f_MHSA,
    C3k2CA,
    C3k2CA2,
    C3k2_MHSA,
    C3k2_MHSA2,
    PatchMerging, PatchEmbed, SwinStage,
    BiLevelRoutingAttention,
    BiFPN_Add2, BiFPN_Add3,
    GSConv, VoVGSCSP,
    CARAFE, ODConv2d,
    BiLevelRoutingAttention,
    DetectWithRecoveryBlock,
    RepNCSPELAN4,
    SPPELAN,
    Down_wt,
    DySample,
    SS2D,
    C2f_KAN,
    C2f_VSS,
    PSA
    )

from GapoNet.apis.lymo_blocks import LymoDetectionLoss

import damei as dm
from copy import deepcopy
import torch
import torch.nn as nn
import contextlib
from GapoNet.apis import lymo_blocks

LOGGER = dm.get_logger('tasks.py')


class LymoDetectionModel(BaseModel):
    
    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=True):
        super().__init__()    
        # super().__init__(cfg, ch, nc, verbose)
        
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override YAML value
            
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        # self.model = self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # default names dict
        self.inplace = self.yaml.get('inplace', True)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment, Pose, DetectWithRecoveryBlock)):
            s = 256  # 2x min stride
            m.inplace = self.inplace

            # forward = lambda x: self.forward(x)[0] if isinstance(m, (Segment, Pose)) else self.forward(x)
            if isinstance(m, (Segment, Pose)):
                forward = lambda x: self.forward(x)[0]
            else:
                forward = lambda x: self.forward(x)
            
            # res = forward(torch.zeros(1, ch, s, s))  # forward
            x = torch.zeros(1, ch, s, s)
            try:
                res = forward(x)  # forward
            except Exception as e:  # support mamba causal_conv_1d
                x = x.to('cuda' if torch.cuda.is_available() else 'cpu')
                self.model.to(x.device)
                res = forward(x)  # forward  
            if isinstance(m, DetectWithRecoveryBlock):  # lymo
                res = res[1]  # res[0] 是 RB 的输出， res[1] 是 detect head 的输出
            m.stride = torch.tensor([s / x.shape[-2] for x in res])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info('')
            
    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs."""
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation)."""
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLOv5 augmented inference tails."""
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        # return v8DetectionLoss(self)
        return LymoDetectionLoss(self)
            

def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    legacy = True
    max_channels = float('inf')
    nc, act, scales = (d.get(x) for x in ('nc', 'activation', 'scales'))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ('depth_multiple', 'width_multiple', 'kpt_shape'))
    if scales:
        scale = d.get('scale')
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        if 'lymo.' in m:  # [-1, 3, lymo.C2f_MHSA, [1024, True]]
            block_name = m[5:]  # C2f_MHSA
            m = getattr(lymo_blocks, block_name)
        else:
            m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]  # get module
        
        for j, a in enumerate(args):  # 1024, None, nearest, True etc
            if isinstance(a, str):
                try:
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
                except ValueError:
                    pass
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in (lymo_blocks.Classify, Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
                 BottleneckCSP, C1, C2, C2f, C2fPSA, C2PSA, C2fAttn, C3, C3k2, C3k2CA, C3k2CA2, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3, PSA, SCDown, C2fCIB, 
                 C2fCA, C2fST, C2f_MHSA,C3k2_MHSA, C3k2_MHSA2, GSConv, VoVGSCSP, CARAFE, ODConv2d, PatchEmbed, PatchMerging, SwinStage, RepNCSPELAN4, ELAN1, SPPELAN, ADown, AConv, Down_wt, C2f_KAN, C2f_VSS, PSA, SPPELAN):
            c1, c2 = ch[f], args[0]  # for C2f_MHSA, c1=上一层, c2=1024
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)  # embed channels
                args[2] = int(
                    max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2]
                )  # num heads
            args = [c1, c2, *args[1:]]  # 重写参数
            if m in (BottleneckCSP, C1, C2, C2f, C3, C3k2, C2fAttn, C3TR, C3Ghost, C3x, RepC3, C2fCIB, C2fPSA, C2PSA, C2fCA, C2fST, C2f_MHSA, GSConv, VoVGSCSP, C2f_KAN, C2f_VSS, PSA, C3k2CA, C3k2_MHSA, C3k2CA2, C3k2_MHSA2):
                args.insert(2, n)  # number of repeats
                n = 1
            if m in (C3k2, C3k2CA, C3k2_MHSA, C3k2CA2, C3k2_MHSA2):  # for M/L/X sizes
                legacy = False
                if scale in "mlx":
                    args[3] = True
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in (HGStem, HGBlock):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        #新加
        elif m in [BiFPN_Add2, BiFPN_Add3]:
            c2 = max([ch[x] for x in f])
        elif m in [BiLevelRoutingAttention]:
            c2 = ch[f]
            args = [c2, *args[0:]]

        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in (Detect, WorldDetect, Segment, Pose, OBB, ImagePoolingAttn, v10Detect):
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {Detect, Segment, Pose, OBB}:
                m.legacy = legacy
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        elif m in [DySample]:
            c1 = ch[f]
            args = [c1, *args[0:]]
        elif m in [SS2D]:
            args[0] = ch[f]
            c2 = ch[f]
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)