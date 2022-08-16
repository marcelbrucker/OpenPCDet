import os

import kitti_common as kitti
from eval import get_official_eval_result, get_coco_eval_result

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]
# det_path = "/path/to/your_result_folder"
det_path = os.path.normpath(os.path.join(__file__, '../../../../../data/kitti/training/label_2_small'))
dt_annos = kitti.get_label_annos(det_path)
# gt_path = "/path/to/your_gt_label_folder"
gt_path = os.path.normpath(os.path.join(__file__, '../../../../../data/kitti/training/label_2_small'))
# gt_split_file = "/path/to/val.txt" # from https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz
gt_split_file = os.path.normpath(os.path.join(__file__, '../../../../../data/kitti/ImageSets/test_small.txt'))
val_image_ids = _read_imageset_file(gt_split_file)
gt_annos = kitti.get_label_annos(gt_path, val_image_ids)
print(get_official_eval_result(gt_annos, dt_annos, 0)) # 6s in my computer
# print(get_coco_eval_result(gt_annos, dt_annos, 0)) # 18s in my computer