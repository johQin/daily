# yolov8

# 1 helloworld

## 数据集下载

```bash
# coco128，从train2017随即选取的128张图片
https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip

wget https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip
```



## 权重下载

在ultralytics github的readme.md下方找到Models栏目，那里的表格直接点击对应的权重即可下载

![](./legend/weight_download.png)

## [训练](https://blog.csdn.net/weixin_42166222/article/details/129391260)

新建一个配置.yaml

```bash
# 在项目根目录下
# 法一：直接复制default.yaml
cp /home/mango/ultralytics/ultralytics/yolo/cfg/default.yaml ./default_copy.yaml
# 法二
yolo copy-cfg

# 在default_copy.yaml文件的基础上，按需修改配置
# eg：
...
model:  weights/yolo8m.pt # (str, optional) path to model file, i.e. yolov8n.pt, yolov8n.yaml
data:  datasets/coco128/coco128.yaml			# 这里修改为自己的yaml，
epochs: 20  # number of epochs to train for
batch: 8  # number of images per batch (-1 for AutoBatch)
...
```

修改：`gitRepository/ultralytics/ultralytics/engine/model.py`

```python
def train(self, trainer=None, **kwargs):
    ...
    overrides = yaml_load(check_yaml(kwargs['cfg'])) if kwargs.get('cfg') else self.overrides
        custom = {'data': TASK2DATA[self.task]}  # method defaults
        # 源码
        # args = {**overrides, **custom, **kwargs, 'mode': 'train'}
        
        # 修改为下面，调整一下优先级，否则会读到custom就会覆盖default_copy.yaml的配置
        # 要修改配置尽量在配置文件中修改
        args = {**custom, **overrides, **kwargs, 'mode': 'train'}  # highest priority args on the right
    ...
```

在根目录下创建train.py。

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("weights/yolov8n.pt")  # 加载预训练模型（推荐用于训练）

# Use the model
# 指定使用的配置文件
results = model.train(cfg='default_copy.yaml')  # 训练模型
```

训练完成后：`/home/buntu/gitRepository/ultralytics/runs/detect/train/weights/best.pt`



配置样例：

```yaml
# default_copy.yaml，里面有部分已修改

# Ultralytics YOLO 🚀, AGPL-3.0 license
# Default training settings and hyperparameters for medium-augmentation COCO training

task: detect  # (str) YOLO task, i.e. detect, segment, classify, pose
mode: train  # (str) YOLO mode, i.e. train, val, predict, export, track, benchmark

# Train settings -------------------------------------------------------------------------------------------------------
model:  weights/yolo8m.pt # (str, optional) path to model file, i.e. yolov8n.pt, yolov8n.yaml
data:  datasets/coco128/coco128.yaml # (str, optional) path to data file, i.e. coco128.yaml
epochs: 16  # (int) number of epochs to train for
patience: 8  # (int) epochs to wait for no observable improvement for early stopping of training
batch: 16  # (int) number of images per batch (-1 for AutoBatch)
imgsz: 640  # (int | list) input images size as int for train and val modes, or list[w,h] for predict and export modes
save: True  # (bool) save train checkpoints and predict results
save_period: -1 # (int) Save checkpoint every x epochs (disabled if < 1)
cache: False  # (bool) True/ram, disk or False. Use cache for data loading
device:  # (int | str | list, optional) device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
workers: 8  # (int) number of worker threads for data loading (per RANK if DDP)
project:  # (str, optional) project name
name:  # (str, optional) experiment name, results saved to 'project/name' directory
exist_ok: False  # (bool) whether to overwrite existing experiment
pretrained: True  # (bool | str) whether to use a pretrained model (bool) or a model to load weights from (str)
optimizer: auto  # (str) optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
verbose: True  # (bool) whether to print verbose output
seed: 0  # (int) random seed for reproducibility
deterministic: True  # (bool) whether to enable deterministic mode
single_cls: False  # (bool) train multi-class data as single-class
rect: False  # (bool) rectangular training if mode='train' or rectangular validation if mode='val'
cos_lr: False  # (bool) use cosine learning rate scheduler
close_mosaic: 10  # (int) disable mosaic augmentation for final epochs (0 to disable)
resume: False  # (bool) resume training from last checkpoint
amp: True  # (bool) Automatic Mixed Precision (AMP) training, choices=[True, False], True runs AMP check
fraction: 1.0  # (float) dataset fraction to train on (default is 1.0, all images in train set)
profile: False  # (bool) profile ONNX and TensorRT speeds during training for loggers
freeze: None  # (int | list, optional) freeze first n layers, or freeze list of layer indices during training
# Segmentation
overlap_mask: True  # (bool) masks should overlap during training (segment train only)
mask_ratio: 4  # (int) mask downsample ratio (segment train only)
# Classification
dropout: 0.0  # (float) use dropout regularization (classify train only)

# Val/Test settings ----------------------------------------------------------------------------------------------------
val: True  # (bool) validate/test during training
split: val  # (str) dataset split to use for validation, i.e. 'val', 'test' or 'train'
save_json: False  # (bool) save results to JSON file
save_hybrid: False  # (bool) save hybrid version of labels (labels + additional predictions)
conf:  # (float, optional) object confidence threshold for detection (default 0.25 predict, 0.001 val)
iou: 0.7  # (float) intersection over union (IoU) threshold for NMS
max_det: 300  # (int) maximum number of detections per image
half: False  # (bool) use half precision (FP16)
dnn: False  # (bool) use OpenCV DNN for ONNX inference
plots: True  # (bool) save plots during train/val

# Prediction settings --------------------------------------------------------------------------------------------------
source:  # (str, optional) source directory for images or videos
show: False  # (bool) show results if possible
save_txt: False  # (bool) save results as .txt file
save_conf: False  # (bool) save results with confidence scores
save_crop: False  # (bool) save cropped images with results
show_labels: True  # (bool) show object labels in plots
show_conf: True  # (bool) show object confidence scores in plots
vid_stride: 1  # (int) video frame-rate stride
stream_buffer: False  # (bool) buffer all streaming frames (True) or return the most recent frame (False)
line_width:   # (int, optional) line width of the bounding boxes, auto if missing
visualize: False  # (bool) visualize model features
augment: False  # (bool) apply image augmentation to prediction sources
agnostic_nms: False  # (bool) class-agnostic NMS
classes:  # (int | list[int], optional) filter results by class, i.e. classes=0, or classes=[0,2,3]
retina_masks: False  # (bool) use high-resolution segmentation masks
boxes: True  # (bool) Show boxes in segmentation predictions

# Export settings ------------------------------------------------------------------------------------------------------
format: torchscript  # (str) format to export to, choices at https://docs.ultralytics.com/modes/export/#export-formats
keras: False  # (bool) use Kera=s
optimize: False  # (bool) TorchScript: optimize for mobile
int8: False  # (bool) CoreML/TF INT8 quantization
dynamic: False  # (bool) ONNX/TF/TensorRT: dynamic axes
simplify: False  # (bool) ONNX: simplify model
opset:  # (int, optional) ONNX: opset version
workspace: 4  # (int) TensorRT: workspace size (GB)
nms: False  # (bool) CoreML: add NMS

# Hyperparameters ------------------------------------------------------------------------------------------------------
lr0: 0.01  # (float) initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
lrf: 0.01  # (float) final learning rate (lr0 * lrf)
momentum: 0.937  # (float) SGD momentum/Adam beta1
weight_decay: 0.0005  # (float) optimizer weight decay 5e-4
warmup_epochs: 3.0  # (float) warmup epochs (fractions ok)
warmup_momentum: 0.8  # (float) warmup initial momentum
warmup_bias_lr: 0.1  # (float) warmup initial bias lr
box: 7.5  # (float) box loss gain
cls: 0.5  # (float) cls loss gain (scale with pixels)
dfl: 1.5  # (float) dfl loss gain
pose: 12.0  # (float) pose loss gain
kobj: 1.0  # (float) keypoint obj loss gain
label_smoothing: 0.0  # (float) label smoothing (fraction)
nbs: 64  # (int) nominal batch size
hsv_h: 0.015  # (float) image HSV-Hue augmentation (fraction)
hsv_s: 0.7  # (float) image HSV-Saturation augmentation (fraction)
hsv_v: 0.4  # (float) image HSV-Value augmentation (fraction)
degrees: 0.0  # (float) image rotation (+/- deg)
translate: 0.1  # (float) image translation (+/- fraction)
scale: 0.5  # (float) image scale (+/- gain)
shear: 0.0  # (float) image shear (+/- deg)
perspective: 0.0  # (float) image perspective (+/- fraction), range 0-0.001
flipud: 0.0  # (float) image flip up-down (probability)
fliplr: 0.5  # (float) image flip left-right (probability)
mosaic: 1.0  # (float) image mosaic (probability)
mixup: 0.0  # (float) image mixup (probability)
copy_paste: 0.0  # (float) segment copy-paste (probability)

# Custom config.yaml ---------------------------------------------------------------------------------------------------
cfg:  # (str, optional) for overriding defaults.yaml

# Tracker settings ------------------------------------------------------------------------------------------------------
tracker: botsort.yaml  # (str) tracker type, choices=[botsort.yaml, bytetrack.yaml]

```





## [推理](https://blog.csdn.net/qq_37553692/article/details/130910432)

[YOLOv8预测参数详解](https://blog.csdn.net/qq_37553692/article/details/130910432)

在项目的根目录创建detect.py，下载官方的预置权重。可以不使用自己训练过后的权重。

```python
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('./weights/yolov8m.pt')

# 统计每帧中的对象
def tongjiFrame(pred_boxes,names):
    objList = []
    obj = {}
    for d in reversed(pred_boxes):
        c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
        name = ('' if id is None else f'id:{id} ') + names[c]
        if name not in obj:
            obj[name] = 0
        obj[name] += 1
        objList.append({'name': name, 'poss': f'{conf:.2f}'})
    # print('你好：', obj)
    return obj, objList

# 推理图片
# img = cv2.imread('./dataset/person.png')
# results = model(img)
# tongjiFrame(results[0].boxes, results[0].names)
# annotated_frame = results[0].plot()
# cv2.imshow("YOLOv8 Inference", annotated_frame)
# cv2.waitKey(0)

# 推理视频
video_path = "./datasets/transport.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        if cv2.waitKey(5) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()


cv2.destroyAllWindows()
```

# 数据集

1. [人群计数、行人检测等开源数据集资源汇总](https://zhuanlan.zhihu.com/p/578090436)
2. 

```yaml
settings_version: 0.0.4
datasets_dir: /home/buntu/gitRepository/datasets
weights_dir: /home/buntu/gitRepository/ultralytics/weights
runs_dir: /home/buntu/gitRepository/ultralytics/runs
uuid: 8850a426fc05d358779fbe19a5891f5956b00a9b1ad62814a559f1f99c0cd012
sync: true
api_key: ''
clearml: true
comet: true
dvc: true
hub: true
mlflow: true
neptune: true
raytune: true
tensorboard: true
wandb: true
```

