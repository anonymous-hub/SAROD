import easydict
from multiprocessing import Process
import yaml
from pathlib import Path

from yolov5.train_dt import *
from EfficientObjectDetection.train_new_reward import *

fine_opt_tr = easydict.EasyDict({
    "cfg": "yolov5/models/yolov5x_custom.yaml",
    "data": "yolov5/data/HRSID_800_od.yaml",
    "hyp": '',
    "epochs": 200,
    "batch_size": 32,
    "img_size": [480, 480],
    "rect": False,
    "resume": False,
    "nosave": False,
    "notest": True,
    "noautoanchor": True,
    "evolve": False,
    "bucket": '',
    "cache_images": True,
    "weights": " ",
    "name": "yolov5x_800_480_200epoch",
    "device": '0',
    "multi_scale": False,
    "single_cls": True,
    "sync_bn": False,
    "local_rank": -1
})


fine_opt_eval = easydict.EasyDict({
    "data": "yolov5/data/HRSID_800_rl.yaml",
    "batch_size": 1,
    "conf_thres": 0.001,
    "iou_thres": 0.6  # for NMS
})

coarse_opt_tr = easydict.EasyDict({
    "cfg": "yolov5/models/yolov5x_custom.yaml",
    "data": "yolov5/data/HRSID_800_od.yaml",
    "hyp": '',
    "epochs": 200,
    "batch_size": 32,
    "img_size": [96, 96],
    "rect": False,
    "resume": False,
    "nosave": False,
    "notest": True,
    "noautoanchor": True,
    "evolve": False,
    "bucket": '',
    "cache_images": True,
    "weights": " ",
    "name": "yolov5x_800_96_200epoch",
    "device": '0',
    "multi_scale": False,
    "single_cls": True,
    "sync_bn": False,
    "local_rank": -1
})


coarse_opt_eval = easydict.EasyDict({
    "data": "yolov5/data/HRSID_800_rl.yaml",
    "batch_size": 1,
    "conf_thres": 0.001,
    "iou_thres": 0.6  # for NMS
})

EfficientOD_opt = easydict.EasyDict({
    "gpu_id": '0',
    "lr": 1e-3,
    "load": None,
    "cv_dir": "cv/0/",
    "batch_size": 1,
    "step_batch_size": 100,
    "img_size": 480,
    "epoch_step": 20,
    "max_epochs": 200,
    "num_workers": 0,
    "test_epoch": 5,
    "parallel": False,
    "alpha": 0.8,
    "beta": 0.1,
    "sigma": 0.5
})

fine_detector = yolov5(fine_opt_tr, fine_opt_eval)
coarse_detector = yolov5(coarse_opt_tr, coarse_opt_eval)
rl_agent = EfficientOD(EfficientOD_opt)

epochs = 100

fine_detector.main(epochs)
coarse_detector.main(epochs)

for e in range(epochs):
    fine_detector.train(e)
    coarse_detector.train(e)
    fine_eval_results = fine_detector.eval('train')
    coarse_eval_results = coarse_detector.eval('train')
    rl_agent.train(e, fine_eval_results, coarse_eval_results)
    # if e % 1 == 0:
    #     eval_fine = fine_detector.eval('val')
    #     eval_coarse = coarse_detector.eval('val')
    #     rl_agent.eval(e, eval_fine, eval_coarse)
    if e % 10 == 0:
        test_fine = fine_detector.eval('test')
        test_coarse = coarse_detector.eval('test')
        rl_agent.test(e, test_fine, test_coarse)





