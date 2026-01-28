import torch
import argparse
from alisuretool.Tools import Tools
from net import *
from util.util_tools import MyCommon, AverageMeter, Logger
from dataset.dataset_tools import FSSDataset, Evaluator
import yaml
import os
import subprocess
# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



def visualize_query_support(
        support_imgs, support_masks,
        query_img, query_masks, pred_mask,
        query_name=None, class_id=None,
        save_dir="./vis_results"
    ):

    import os
    import numpy as np
    from PIL import Image

    os.makedirs(save_dir, exist_ok=True)

    # ======================
    # mask overlay function
    # ======================
    def apply_mask(img, mask, color=(255,0,0), alpha=0.5):
        """img: HWC uint8, mask: HW uint8"""
        if img.ndim != 3:
            raise ValueError(f"img should be HWC, got {img.shape}")
        if mask.ndim != 2:
            raise ValueError(f"mask should be HW, got {mask.shape}")

        img = img.astype(np.float32) / 255.0
        mask = mask.astype(np.uint8)
        color = np.array(color) / 255.0

        overlay = img.copy()
        for c in range(3):
            overlay[:,:,c] = np.where(
                mask == 1,
                img[:,:,c] * (1 - alpha) + alpha * color[c],
                img[:,:,c]
            )
        return (overlay * 255).astype(np.uint8)

    # ======================
    #   support image proc
    # ======================
    support_imgs_np = []
    for s in support_imgs:
        arr = s[:3].detach().cpu().numpy()  # 3,H,W
        arr = np.transpose(arr, (1,2,0))    # H,W,C
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-5)
        arr = (arr * 255).astype(np.uint8)
        support_imgs_np.append(arr)

    support_masks_np = [(m.detach().cpu().numpy() > 0.5).astype(np.uint8)
                        for m in support_masks]

    support_overlay = [
        Image.fromarray(apply_mask(img, m, color=(0,0,255)))
        for img, m in zip(support_imgs_np, support_masks_np)
    ]

    # ======================
    #   query image proc
    # ======================
    query_np = query_img[:3].detach().cpu().numpy()  # 3,H,W
    query_np = np.transpose(query_np, (1,2,0))        # H,W,C
    query_np = (query_np - query_np.min()) / (query_np.max() - query_np.min() + 1e-5)
    query_np = (query_np * 255).astype(np.uint8)

    # ======================
    #   query GT mask
    # ======================
    raw = query_masks.detach().cpu().numpy()

    # ==== 自动修复 mask 形状 ====
    # Case 1: 一维向量 (H*W,)
    if raw.ndim == 1:
        H, W = query_np.shape[:2]
        if raw.size == H * W:
            raw = raw.reshape(H, W)
        else:
            raise ValueError(f"Cannot reshape query mask of shape {raw.shape} into image size {(H,W)}")

    # Case 2: (1,H,W)
    elif raw.ndim == 3 and raw.shape[0] == 1:
        raw = raw[0]

    # Case 3: (H,W,1)
    elif raw.ndim == 3 and raw.shape[2] == 1:
        raw = raw[:,:,0]

    # 最终保证是 (H,W)
    elif raw.ndim != 2:
        raise ValueError(f"query mask wrong shape: {raw.shape}")

    gt_mask = (raw > 0.5).astype(np.uint8)
    query_masked_gt = Image.fromarray(apply_mask(query_np, gt_mask, color=(255,0,0)))

    # ======================
    #   predicted mask
    # ======================
    pred_mask_np = pred_mask.detach().cpu().squeeze().numpy()
    if pred_mask_np.ndim != 2:
        raise ValueError(f"pred_mask wrong shape: {pred_mask_np.shape}")

    pred_mask_np = (pred_mask_np > 0.5).astype(np.uint8)
    query_pred_overlay = Image.fromarray(apply_mask(query_np, pred_mask_np, color=(255,0,0)))

    query_original = Image.fromarray(query_np)

    # ======================
    #   combine
    # ======================
    all_imgs = support_overlay + [query_pred_overlay, query_original, query_masked_gt]

    widths, heights = zip(*(im.size for im in all_imgs))
    total_width = sum(widths)
    max_height = max(heights)

    canvas = Image.new("RGB", (total_width, max_height))
    x = 0
    for im in all_imgs:
        canvas.paste(im, (x, 0))
        x += im.size[0]

    # ======================
    # save file
    # ======================
    if query_name is not None:
        fname = os.path.splitext(os.path.basename(str(query_name)))[0]
    elif class_id is not None:
        fname = f"class_{class_id}"
    else:
        fname = "sample"

    save_path = os.path.join(save_dir, f"{fname}.png")
    canvas.save(save_path)
    return save_path

def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    args = argparse.Namespace(**cfg)
    yaml_base = os.path.splitext(os.path.basename(yaml_path))[0]
    args.logpath = yaml_base
    return args

def save_pred_masks(pred, save_dir="./pred_masks"):
    """
    pred: torch.Tensor, shape (B,H,W) or (B,num_classes,H,W)
    """
    os.makedirs(save_dir, exist_ok=True)

    if pred.dim() == 4:  # (B, num_classes, H, W)
        pred = torch.argmax(pred, dim=1)  # -> (B,H,W)

    pred = pred.cpu().numpy().astype(np.uint8)

    for i in range(pred.shape[0]):
        mask = pred[i]  # (H,W)
        img = Image.fromarray(mask)  # 保存成灰度图，每个像素=类别id
        img.save(os.path.join(save_dir, f"pred_{i}.png"))
        
class Runner(object):

    def __init__(self, args):
        self.args = args
        self.device = MyCommon.gpu_setup(use_gpu=self.args.use_gpu, gpu_id=args.gpuid)

        net_cls = globals()[self.args.net_name]
        self.model = net_cls(args).to(self.device)
        self.model.eval()
        weights = torch.load(args.load, map_location=None if self.args.use_gpu else torch.device('cpu'))
        weights = {one.replace("module.", ""): weights[one] for one in weights.keys()}
        weights = {one.replace("hpn_learner.", "mgcd."): weights[one] for one in weights.keys()}
        self.model.load_state_dict(weights)

        FSSDataset.initialize(img_size=args.img_size, datapath=args.datapath)
        self.dataloader_val = FSSDataset.build_dataloader(
            args.benchmark, args.bsz, args.nworker, args.fold, 'val', args.shot,
            use_mask=args.mask, mask_num=args.mask_num)
        Logger.log_params(self.model)
        
    @torch.no_grad()
    def test(self):
        dataloader = self.dataloader_val
        average_meter = AverageMeter(dataloader.dataset, device=self.device)

        for idx, batch in enumerate(dataloader):
            batch = MyCommon.to_cuda(batch, device=self.device)
            pred = self.model.predict_nshot(batch)

            area_inter, area_union = Evaluator.classify_prediction(pred, batch)
            average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
            average_meter.write_process(idx, len(dataloader), 0, write_batch_idx=5)
        miou, fb_iou = average_meter.compute_iou()
        return miou, fb_iou
 

    @torch.no_grad()
    def test_class(self, target_classes=None, fold=0, dataset='lcma'):
        MyCommon.fix_randseed(0)
        dataloader = self.dataloader_val
        average_meter = AverageMeter(dataloader.dataset, device=self.device)
        print("visualizing classes:", target_classes)

        for idx, batch in enumerate(dataloader):
            batch = MyCommon.to_cuda(batch, device=self.device)
            pred = self.model.predict_nshot(batch)
    
            if target_classes is not None:
                for i in range(pred.size(0)):
                    query_name = batch['query_name'][i] if 'query_name' in batch else None
                    class_id = batch['class_id'][i].item()
                    visualize_query_support(
                        batch['support_imgs'][i], 
                        batch['support_labels'][i], 
                        batch['query_img'][i], 
                        batch['query_label'][i],
                        pred[i], 
                        query_name=query_name, 
                        class_id=class_id,
                        save_dir=f"./vis_results_{dataset}_fold{fold}/class_{class_id}"
                    )

            # 计算 IoU
            area_inter, area_union = Evaluator.classify_prediction(pred, batch)
            average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
            average_meter.write_process(idx, len(dataloader), 0, write_batch_idx=10)

        miou, fb_iou, iou, class_ids = average_meter.compute_iou_class()
        return miou, fb_iou, iou, class_ids



if __name__ == '__main__':
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="YAML 配置文件路径")
    parser.add_argument("weights", type=str, help="模型权重路径")
    parser.add_argument("shot", type=int, nargs="?", default=None, help="few-shot 数量（可选）")
    parser.add_argument("--class", type=int, nargs="+", dest="vis_classes", default=None,
                        help="指定需要可视化的类别ID (例如: --class 7 8)")
    args_cli = parser.parse_args()

    # 加载 yaml
    args = load_yaml_config(args_cli.config)
    args.load = args_cli.weights  # 覆盖权重路径

    # shot 参数优先级: CLI > yaml
    if args_cli.shot is not None:
        args.shot = args_cli.shot

    # net_name 默认设置
    if not hasattr(args, 'net_name'):
        Tools.print("Warning: net_name not specified in config! Using default APSENetwork.")
        args.net_name = 'APSENetwork'

    MyCommon.fix_randseed(0)
    Logger.initialize(args, training=False)
    runner = Runner(args=args)
    Logger.log_params(runner.model)
    # Tools.print("Model Architcture:\n")
    # Tools.print(str(runner.model))

    # 调用 test_class（始终算 IoU），是否可视化由 --class 控制
    miou, fb_iou, iou, class_ids = runner.test_class(target_classes=args_cli.vis_classes,fold=args.fold,dataset=args.benchmark)

    print("mIoU (mean):", miou.item() if torch.is_tensor(miou) else miou)
    print("FB-IoU:", fb_iou.item() if torch.is_tensor(fb_iou) else fb_iou)

    # 输出每类指标
    Tools.print("Per-class IoU:")
    id2name = runner.dataloader_val.dataset.id2name
    results = {
        "mIoU": miou.item() if torch.is_tensor(miou) else miou,
        "FB-IoU": fb_iou.item() if torch.is_tensor(fb_iou) else fb_iou,
        "per_class": {}
    }
    for class_id, class_iou in zip(class_ids.tolist(), iou):
        class_name = id2name.get(class_id, f"Class_{class_id}")
        results["per_class"][f"{class_name}_id{class_id}"] = class_iou.item()

        print(f"{class_name} (id {class_id}): {class_iou.item():.4f}")


    import json, os
    save_dir = os.path.dirname(args.load)  # pt 文件所在目录
    base_name = os.path.splitext(os.path.basename(args.load))[0]
    # 根据 shot 参数命名（如 modelname_1shot_eval.json）
    save_name = f"{base_name}_{args.shot}shot_eval.json"
    save_path = os.path.join(save_dir, save_name)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    Tools.print(f"📂 results saved to {save_path}")
