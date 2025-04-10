
import os, sys
from pathlib import Path
here = Path(__file__).parent

try:
    from ultralytics import YOLO  # 使用Ultralytics库的YOLO
except:
    local_yolo_path = f'{here.parent}/ultralytics'
    if local_yolo_path not in sys.path:
        sys.path.insert(1, local_yolo_path)
    from ultralytics import YOLO  # 使用本地的YOLO库
    print("Use local Ultralytics backend in `repos/ultralytics`, or you can install it by running `pip install ultralytics`")



from ultralytics import YOLO
from ..ultralytics.ultralytics.nn.tasks import (
    parse_model, 
    DetectionModel,
    BaseModel,
    v8DetectionLoss,
    yaml_model_load,
    initialize_weights,
    make_divisible,
    colorstr,
    scale_img,
    ClassificationModel,
    )
from ..ultralytics.ultralytics.nn.modules import (
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
from ..ultralytics.ultralytics.models.yolo.detect.train import (
    DetectionTrainer,
    )
from ..ultralytics.ultralytics.models.yolo.detect.val import (
    DetectionValidator,
    )
from ..ultralytics.ultralytics.models.yolo.detect.predict import (
    DetectionPredictor,
    )

from ..ultralytics.ultralytics.utils import (
    RANK,
    DEFAULT_CFG,
    DEFAULT_CFG_DICT,
    IterableSimpleNamespace,
    callbacks,
    ops,
    )
from ..ultralytics.ultralytics.utils.torch_utils import (
    de_parallel,
    torch_distributed_zero_first,
)
from ..ultralytics.ultralytics.utils.tal import (
    make_anchors,
    dist2bbox,
    TaskAlignedAssigner,
)
from ..ultralytics.ultralytics.utils.ops import (
    crop_mask, xywh2xyxy, xyxy2xywh
)
from ..ultralytics.ultralytics.utils.loss import (
    BboxLoss,
)
from ..ultralytics.ultralytics.utils.instance import (
    Instances,
)
from ..ultralytics.ultralytics.data.dataset import (
    YOLODataset,
    BaseDataset,
)
from ..ultralytics.ultralytics.data.utils import (
     check_det_dataset, check_cls_dataset, clean_url, emojis
)


from ..ultralytics.ultralytics.data.augment import (
    Compose,
    LetterBox,
    Format,
    Mosaic,
    CopyPaste,
    RandomPerspective,
    MixUp,
    Albumentations,
    RandomHSV,
    RandomFlip,
    v8_transforms,
)
from ..ultralytics.ultralytics.utils.metrics import (
    bbox_iou,
    bbox_ioa,
    ClassifyMetrics,
    ConfusionMatrix,
    ap_per_class,
    compute_ap,
    plot_mc_curve,
    plot_pr_curve,
    smooth,
    DetMetrics,
)
from ..ultralytics.ultralytics.models.yolo.classify.val import (
    ClassificationValidator,
    )
from ..ultralytics.ultralytics.models.yolo.classify.train import (
    ClassificationTrainer
    )
from ..ultralytics.ultralytics.models.yolo.classify.predict import (
    ClassificationPredictor
    )
from ..ultralytics.ultralytics.engine.validator import (
    BaseValidator,
)
from ..ultralytics.ultralytics.cfg import (
    get_save_dir, cfg2dict, check_dict_alignment, LOGGER,
    CFG_FLOAT_KEYS, CFG_FRACTION_KEYS, CFG_INT_KEYS, CFG_BOOL_KEYS,
    )
from ..ultralytics.ultralytics.utils.checks import check_imgsz
