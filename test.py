import easydict
from multiprocessing import Process
import yaml
from pathlib import Path

from yolov5.train_dt import *
from EfficientObjectDetection.train_800 import *

EfficientOD_opt = easydict.EasyDict({
    "gpu_id": '1',
    "lr": 1e-3,
    "load": "EfficientObjectDetection/cv/tmp/ckpt_E_0_R_2.94E-01",
    "cv_dir": "cv/test/",
    "batch_size": 1,
    "img_size": 480,
    "epoch_step": 10000,
    "max_epochs": 300,
    "num_workers": 0,
    "test_epoch": 5,
    "parallel": False,
    "alpha": 0.8,
    "beta": 0.1,
    "sigma": 0.5
})

fine_opt_tr = easydict.EasyDict({
    "cfg": "/home/kang/SAR_OD/EfficientOD_WACV/yolov5/models/yolov5x_custom.yaml",
    "data": "/home/kang/SAR_OD/EfficientOD_WACV/yolov5/data/HRSID_800_od.yaml",
    "hyp": '',
    "epochs": 1,
    "batch_size": 32,
    "img_size": [480, 480],
    "rect": False,
    "resume": False,
    "nosave": False,
    "notest": True,
    "noautoanchor": True,
    "evolve": False,
    "bucket": '',
    "cache_images": False,
    "weights": " ",
    "name": "yolov5x_800_480_200epoch",
    "device": '1',
    "multi_scale": False,
    "single_cls": True,
    "sync_bn": False,
    "local_rank": -1
})

fine_opt_eval = easydict.EasyDict({
    "data": "/home/kang/SAR_OD/EfficientOD_WACV/yolov5/data/HRSID_800_rl.yaml",
    "batch_size": 1,
    "conf_thres": 0.001,
    "iou_thres": 0.6  # for NMS
})

coarse_opt_tr = easydict.EasyDict({
    "cfg": "/home/kang/SAR_OD/EfficientOD_WACV/yolov5/models/yolov5x_custom.yaml",
    "data": "/home/kang/SAR_OD/EfficientOD_WACV/yolov5/data/HRSID_800_od.yaml",
    "hyp": '',
    "epochs": 1,
    "batch_size": 32,
    "img_size": [96, 96],
    "rect": False,
    "resume": False,
    "nosave": False,
    "notest": True,
    "noautoanchor": True,
    "evolve": False,
    "bucket": '',
    "cache_images": False,
    "weights": " ",
    "name": "yolov5x_800_96_200epoch",
    "device": '1',
    "multi_scale": False,
    "single_cls": True,
    "sync_bn": False,
    "local_rank": -1
})

coarse_opt_eval = easydict.EasyDict({
    "data": "/home/kang/SAR_OD/EfficientOD_WACV/yolov5/data/HRSID_800_rl.yaml",
    "batch_size": 1,
    "conf_thres": 0.001,
    "iou_thres": 0.6  # for NMS
})

rl_agent = EfficientOD(EfficientOD_opt)

fine_detector = yolov5(fine_opt_tr, fine_opt_eval)
coarse_detector = yolov5(coarse_opt_tr, coarse_opt_eval)

fine_detector.main(0)
coarse_detector.main(0)

# policy, path
returns = rl_agent.test()

coarse_weight = '/home/kang/SAR_OD/EfficientOD_WACV/runs/exp1_yolov5x_800_96_200epoch/weights/last_yolov5x_800_96_200epoch.pt'
fine_weight = '/home/kang/SAR_OD/EfficientOD_WACV/runs/exp0_yolov5x_800_480_200epoch/weights/last_yolov5x_800_480_200epoch.pt'

for (policy, path) in returns:
    for ind, i in enumerate(policy):
        path = path[0].replace('test', 'rl_ver/test').replace('.jpg', '_' + str(ind) + '.jpg')
        if i == 0:
            c_result = coarse_detector.test(coarse_weight, path, '1')
            print(c_result)
        if i == 1:
            f_result = fine_detector.test(fine_weight, path, '1')


