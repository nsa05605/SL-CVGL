import os
import time

import json
import math
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.font_manager as font_manager

from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from PIL import Image

from datetime import datetime, timedelta
from argparse import ArgumentParser

from collections import defaultdict
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToPILImage
import torch.nn.functional as F

import src
import src.data.collate_funcs
from src.utils import MetricDict
from src.data.dataloader import nuScenesMaps
import src.model.network as networks

mpl.rcParams["font.family"] = "serif"
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + "/fonts/ttf/cmr10.ttf")
mpl.rcParams["font.serif"] = cmfont.get_name()
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.grid"] = True


def train(args, dataloader, model, optimizer, epoch):
    print("\n==> Training on {} minibatches".format(len(dataloader)))
    model.train()  # 모델을 학습 모드로 설정
    # MetricDict 객체 초기화
    epoch_loss = MetricDict()
    epoch_loss_per_class = MetricDict()
    batch_acc_loss = MetricDict()
    epoch_iou = MetricDict()
    t = time.time()
    num_classes = len(args.pred_classes_nusc)

    # 데이터 로더로부터 미니배치를 가져옴
    for i, ((image, calib, grid2d), (cls_map, vis_mask)) in enumerate(dataloader):

        # Move tensors to GPU
        image, calib, cls_map, vis_mask, grid2d = (
            image.cuda(),
            calib.cuda(),
            cls_map.cuda(),
            vis_mask.cuda(),
            grid2d.cuda(),
        )

        # Run network forwards
        pred_ms = model(image, calib, grid2d)

        # Convert ground truths to binary mask
        # class map과 visual mask를 이진 mask로 변환
        gt_s1 = (cls_map > 0).float()
        visibility_mask_s1 = (vis_mask > 0).float()

        # Downsample to match model outputs
        # 이진화된 mask를 다운샘플링(출략 크기 맞추기)
        map_sizes = [pred.shape[-2:] for pred in pred_ms]  # 모델의 각 출력에 대해 다운샘플링한다.
        gt_ms = src.utils.downsample_gt(gt_s1, map_sizes)
        vis_ms = src.utils.downsample_gt(visibility_mask_s1, map_sizes)

        # Compute losses for backprop
        # 모델의 예측값인 pred_ms와 그에 상응하는 지상 이미지 gt_ms간의 손실 계산
        loss, loss_dict = compute_loss(pred_ms, gt_ms, args.loss, args)

        # Calculate gradients
        loss.backward()

        # Compute IoU
        # IoU: 모델의 예측과 실제 지상 이미지 간의 겹침 정도를 나타내는 지표
        # compute_multiscale_iou(): 여러 스케일에서 IoU를 계산하고 각 샘플 및 클래스에 대한 IoU 값 반환하는 함수
        # 모델의 예측이 얼마나 정확하게 gt 이미지와 겹치는지 평가하는 데 사용
        iou_per_sample, iou_dict = src.utils.compute_multiscale_iou(
            pred_ms, gt_ms, vis_ms, num_classes
        )

        # Compute per class loss for eval
        # 각 클래스에 대한 손실을 계산
        # compute_multiscale_loss_per_class(): 다중 스케일에서 예측된 결과와 gt 이미지를 기반으로 각 클래스에 대한 손실을 계산하고 이를 사전에 정의된 클래스에 따라 분류하는 함수
        per_class_loss_dict = src.utils.compute_multiscale_loss_per_class(
            pred_ms, gt_ms,
        )

        # 손실값이 NaN이 되면(모델의 가중치가 발산할 경우) RuntimeError 발생
        if float(loss) != float(loss):
            raise RuntimeError("Loss diverged :(")

        # 누적
        epoch_loss += loss_dict
        epoch_loss_per_class += per_class_loss_dict
        batch_acc_loss += loss_dict
        epoch_iou += iou_dict

        # 각 배치가 accumulation_steps 배치마다 그래디언트 업데이트를 수행하고 그동안 누적된 그래디언트 초기화
        if (i + 1) % args.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

            # Print summary
            batch_time = (time.time() - t) / (1 if i == 0 else args.print_iter)  # 현재 배치의 실행 시간
            eta = ((args.epochs - epoch + 1) * len(dataloader) - i) * batch_time  # 예상 남은 학습 시간

            s = "[Epoch: {} {:4d}/{:4d}] batch_time: {:.2f}s eta: {:s} loss: ".format(
                epoch, i, len(dataloader), batch_time, str(timedelta(seconds=int(eta)))
            )
            # loss
            for k, v in batch_acc_loss.mean.items():
                s += "{}: {:.2e} ".format(k, v)

            with open(os.path.join(args.savedir, args.name, "output.txt"), "a") as fp:
                fp.write(s)
            print(s)
            t = time.time()

            batch_acc_loss = MetricDict()

    # Calculate per class IoUs over set
    # 각 클래스에 대한 평균 IoU 계산
    scales = [pred.shape[-1] for pred in pred_ms]
    ms_cumsum_iou_per_class = torch.stack(
        [epoch_iou["s{}_iou_per_class".format(scale)] for scale in scales]
    )
    ms_count_per_class = torch.stack(
        [epoch_iou["s{}_class_count".format(scale)] for scale in scales]
    )
    ms_ious_per_class = (
        (ms_cumsum_iou_per_class / (ms_count_per_class + 1)).cpu().numpy()
    )
    ms_mean_iou = ms_ious_per_class.mean(axis=1)

    # Calculate per class loss over set
    # 각 클래스에 대한 평균 손실 계산
    ms_cumsum_loss_per_class = torch.stack(
        [epoch_loss_per_class["s{}_loss_per_class".format(scale)] for scale in scales]
    )
    ms_loss_per_class = (
        (ms_cumsum_loss_per_class / (ms_count_per_class + 1)).cpu().numpy()
    )
    total_loss = ms_loss_per_class.mean(axis=1).sum()

    # Print epoch summary and save results
    print("==> Training epoch complete")
    for key, value in epoch_loss.mean.items():
        print("{:8s}: {:.4e}".format(key, value))  # 각 손실 지표의 이름과 해당 에폭동안의 평균 값 포함

    with open(os.path.join(args.savedir, args.name, "train_loss.txt"), "a") as f:
        f.write("\n")
        f.write(
            "{},".format(epoch)
            + "{},".format(float(total_loss))
            + "".join("{},".format(v) for v in ms_mean_iou)
        )
    with open(os.path.join(args.savedir, args.name, "train_ious.txt"), "a") as f:
        f.write("\n")
        f.write(
            "Epoch: {}, \n".format(epoch)
            + "Total Loss: {}, \n".format(float(total_loss))
            + "".join(
                "s{}_ious_per_class: {}, \n".format(s, v)
                for s, v in zip(scales, ms_ious_per_class)
            )
            + "".join(
                "s{}_loss_per_class: {}, \n".format(s, v)
                for s, v in zip(scales, ms_loss_per_class)
            )
        )


# 검증 단계에서 모델을 평가하는 함수
def validate(args, dataloader, model, epoch):
    print("\n==> Validating on {} minibatches\n".format(len(dataloader)))
    model.eval()  # 모델을 평가 모드로 설정
    epoch_loss = MetricDict()
    epoch_iou = MetricDict()
    epoch_loss_per_class = MetricDict()
    num_classes = len(args.pred_classes_nusc)
    times = []

    for i, ((image, calib, grid2d), (cls_map, vis_mask)) in enumerate(dataloader):
        # Move tensors to GPU
        image, calib, cls_map, vis_mask, grid2d = (
            image.cuda(),
            calib.cuda(),
            cls_map.cuda(),
            vis_mask.cuda(),
            grid2d.cuda(),
        )

        with torch.no_grad():
            # Run network forwards
            pred_ms = model(image, calib, grid2d)  # 모델의 예측값

            # Upsample largest prediction to 200x200
            # 모델이 생성한 가장 큰 스케일(pred_ms[0])의 예측을 200x200크기로 업샘플링
            # 검증 단계에서는 모델이 다양한 스케일에서 추출한 특징을 하나의 통합된 결과로 가져와서 성능을 평가하므로 업샘플링한다.
            pred_200x200 = F.interpolate(
                pred_ms[0], size=(200, 200), mode="bilinear"
            )
            # pred_200x200 = (pred_200x200 > 0).float()
            pred_ms = [pred_200x200, *pred_ms]  # 모델의 예측값 리스트 맨 앞에 업샘플링된 예측값을 추가

            # Get required gt output sizes
            # pred_ms 리스트에 있는 각 예측 텐서의 크기(높이, 너비)를 gt로 가져옴
            map_sizes = [pred.shape[-2:] for pred in pred_ms]

            # Convert ground truth to binary mask
            # gt를 이진 mask로 변환
            gt_s1 = (cls_map > 0).float()
            vis_mask_s1 = (vis_mask > 0.5).float()

            # Downsample to match model outputs
            # 모델의 출력에 맞게 gt 이미지를 다운 샘플링
            gt_ms = src.utils.downsample_gt(gt_s1, map_sizes)
            vis_ms = src.utils.downsample_gt(vis_mask_s1, map_sizes)

            # Compute IoU
            # IoU 평가지표 계산
            iou_per_sample, iou_dict = src.utils.compute_multiscale_iou(
                pred_ms, gt_ms, vis_ms, num_classes
            )
            # Compute per class loss for eval
            # 평가를 위한 클래스 별 loss 계산
            per_class_loss_dict = src.utils.compute_multiscale_loss_per_class(
                pred_ms, gt_ms,
            )

            epoch_iou += iou_dict
            epoch_loss_per_class += per_class_loss_dict

            # Visualize predictions
            # if epoch % args.val_interval * 4 == 0 and i % 50 == 0:
            #     vis_img = ToPILImage()(image[0].detach().cpu())
            #     pred_vis = pred_ms[1].detach().cpu()
            #     label_vis = gt_ms[1]
            #
            #     # Visualize scores
            #     vis_fig = visualize_score(
            #         pred_vis[0],
            #         label_vis[0],
            #         grid2d[0],
            #         vis_img,
            #         iou_per_sample[0],
            #         num_classes,
            #     )
            #     plt.savefig(
            #         os.path.join(
            #             args.savedir,
            #             args.name,
            #             "val_output_epoch{}_iter{}.png".format(epoch, i),
            #         )
            #     )

    print("\n==> Validation epoch complete")

    # Calculate per class IoUs over set
    # 각 클래스에 대한 평균 IoU 계산
    scales = [pred.shape[-1] for pred in pred_ms]

    ms_cumsum_iou_per_class = torch.stack(
        [epoch_iou["s{}_iou_per_class".format(scale)] for scale in scales]
    )
    ms_count_per_class = torch.stack(
        [epoch_iou["s{}_class_count".format(scale)] for scale in scales]
    )

    ms_ious_per_class = (
        (ms_cumsum_iou_per_class / (ms_count_per_class + 1e-6)).cpu().numpy()
    )
    ms_mean_iou = ms_ious_per_class.mean(axis=1)

    # Calculate per class loss over set
    # 각 클래스에 대한 평균 손실 계산
    ms_cumsum_loss_per_class = torch.stack(
        [epoch_loss_per_class["s{}_loss_per_class".format(scale)] for scale in scales]
    )
    ms_loss_per_class = (
        (ms_cumsum_loss_per_class / (ms_count_per_class + 1)).cpu().numpy()
    )
    total_loss = ms_loss_per_class.mean(axis=1).sum()

    with open(os.path.join(args.savedir, args.name, "val_loss.txt"), "a") as f:
        f.write("\n")
        f.write(
            "{},".format(epoch)
            + "{},".format(float(total_loss))
            + "".join("{},".format(v) for v in ms_mean_iou)
        )

    with open(os.path.join(args.savedir, args.name, "val_ious.txt"), "a") as f:
        f.write("\n")
        f.write(
            "Epoch: {},\n".format(epoch)
            + "Total Loss: {},\n".format(float(total_loss))
            + "".join(
                "s{}_ious_per_class: {}, \n".format(s, v)
                for s, v in zip(scales, ms_ious_per_class)
            )
            + "".join(
                "s{}_loss_per_class: {}, \n".format(s, v)
                for s, v in zip(scales, ms_loss_per_class)
            )
        )


# 모델의 예측값(pred)와 실제 레이블(label)을 사용하여 손실을 계산하는 함수
def compute_loss(preds, labels, loss_name, args):
    scale_idxs = torch.arange(len(preds)).int()  # 예측값이 다양한 스케일에서 계산되었음을 나타내는 인덱스
    # -> 이 인덱스를 사용하여 다양한 스케일에서의 손실을 계산함

    # Dice loss across classes at multiple scales
    # 다중 스케일에서 여러 클래스에 대한 Dice Loss를 계산
    # Dice Loss: 두 세트(pred, label)의 집합 간의 유사성을 측정하는 지표
    ms_loss = torch.stack(  # stack으로 다중 스케일에 대한 전체 손실을 얻음
        [
            src.model.loss.__dict__[loss_name](pred, label, idx_scale, args)
            for pred, label, idx_scale in zip(preds, labels, scale_idxs)
        ]
    )

    if "90" not in args.model_name:
        total_loss = torch.sum(ms_loss[3:]) + torch.mean(ms_loss[:3])
    else:
        total_loss = torch.sum(ms_loss)

    # Store losses in dict
    total_loss_dict = {
        "loss": float(total_loss),
    }

    return total_loss, total_loss_dict


# 시각화 함수
def visualize_score(scores, heatmaps, grid, image, iou, num_classes):
    # Condese scores and ground truths to single map
    class_idx = torch.arange(len(scores)) + 1
    logits = scores.clone().cpu() * class_idx.view(-1, 1, 1)
    logits, _ = logits.max(dim=0)
    scores = (scores.detach().clone().cpu() > 0.5).float() * class_idx.view(-1, 1, 1)
    scores, _ = scores.max(dim=0)
    heatmaps = (heatmaps.detach().clone().cpu() > 0.5).float() * class_idx.view(
        -1, 1, 1
    )
    heatmaps, _ = heatmaps.max(dim=0)

    # Visualize score
    fig = plt.figure(num="score", figsize=(8, 6))
    fig.clear()

    gs = mpl.gridspec.GridSpec(2, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1:, 1])
    ax4 = fig.add_subplot(gs[1:, 2])

    image = ax1.imshow(image)
    ax1.grid(which="both")
    src.visualization.encoded.vis_score_raw(logits, grid, cmap="magma", ax=ax2)
    src.vis_score(scores, grid, cmap="magma", ax=ax3, num_classes=num_classes)
    src.vis_score(heatmaps, grid, cmap="magma", ax=ax4, num_classes=num_classes)

    grid = grid.cpu().detach().numpy()
    yrange = np.arange(grid[:, 0].max(), step=5)
    xrange = np.arange(start=grid[0, :].min(), stop=grid[0, :].max(), step=5)
    ymin, ymax = 0, grid[:, 0].max()
    xmin, xmax = grid[0, :].min(), grid[0, :].max()

    ax2.vlines(xrange, ymin, ymax, color="white", linewidth=0.5)
    ax2.hlines(yrange, xmin, xmax, color="white", linewidth=0.5)
    ax3.vlines(xrange, ymin, ymax, color="white", linewidth=0.5)
    ax3.hlines(yrange, xmin, xmax, color="white", linewidth=0.5)
    ax4.vlines(xrange, ymin, ymax, color="white", linewidth=0.5)
    ax4.hlines(yrange, xmin, xmax, color="white", linewidth=0.5)

    ax1.set_title("Input image", size=11)
    ax2.set_title("Model output logits", size=11)
    ax3.set_title("Model prediction = logits" + r"$ > 0.5$", size=11)
    ax4.set_title("Ground truth", size=11)

    # plt.suptitle(
    #     "IoU : {:.2f}".format(iou), size=14,
    # )

    gs.tight_layout(fig)
    gs.update(top=0.9)

    return fig


def parse_args():
    parser = ArgumentParser()

    # ----------------------------- Data options ---------------------------- #
    parser.add_argument(
        "--root",
        type=str,
        default="nuscenes_data",
        help="root directory of the dataset",
    )
    parser.add_argument(  # NuScenes 데이터셋의 버전 선택
        "--nusc-version", type=str, default="v1.0-trainval", help="nuscenes version",
    )
    parser.add_argument(  # 센서의 높이를 고려한 gt 지도 선택..?
        "--occ-gt",
        type=str,
        default="200down100up",
        help="occluded (occ) or unoccluded(unocc) ground truth maps",
    )
    parser.add_argument(  # gt 버전 선택
        "--gt-version",
        type=str,
        default="semantic_maps_new_200x200",  # -> 그래서 validation 부분에서 200x200로 업샘플링
        help="ground truth name",
    )
    parser.add_argument(  # train set에 사용할 데이터 분할 지정
        "--train-split", type=str, default="train_mini", help="ground truth name",
    )
    parser.add_argument(  # val set에 사용할 데이터 분할 지정
        "--val-split", type=str, default="val_mini", help="ground truth name",
    )
    parser.add_argument(  # 전체 데이터셋의 백분율
        "--data-size",
        type=float,
        default=0.2,  # 전체 데이터셋의 20%만 사용하여 학습
        help="percentage of dataset to train on",
    )
    parser.add_argument(  # NuScenes 데이터셋에서 로드할 클래스 정의
        "--load-classes-nusc",
        type=str,
        nargs=14,
        default=[
            "drivable_area",
            "ped_crossing",
            "walkway",
            "carpark_area",
            "road_segment",
            "lane",
            "bus",
            "bicycle",
            "car",
            "construction_vehicle",
            "motorcycle",
            "trailer",
            "truck",
            "pedestrian",
            "trafficcone",
            "barrier",
        ],
        help="Classes to load for NuScenes",
    )
    parser.add_argument(  # NuScenes 데이터셋에서 예측할 클래스 지정
        "--pred-classes-nusc",
        type=str,
        nargs=12,
        default=[
            "drivable_area",
            "ped_crossing",
            "walkway",
            "carpark_area",
            "bus",
            "bicycle",
            "car",
            "construction_vehicle",
            "motorcycle",
            "trailer",
            "truck",
            "pedestrian",
            "trafficcone",
            "barrier",
        ],
        help="Classes to predict for NuScenes",
    )
    parser.add_argument(  # Lidar 데이터에서 mask 지정 --> Lidar data 사용하는 부분 논문에서 찾아보기
        "--lidar-ray-mask",
        type=str,
        default="dense",  # 밀도 높은 Lidar ray를 사용하여 가시성을 나타냄
        help="sparse or dense lidar ray visibility mask",
    )
    parser.add_argument(  # validation grid의 너비와 깊이를 미터 단위로 설정
        "--grid-size",  # -> 평가 중에 사용되는 공간적인 영역
        type=float,
        nargs=2,
        default=(50.0, 50.0),  # validation grid의 기본 크기는 폭 50, 깊이 50
        help="width and depth of validation grid, in meters",
    )
    parser.add_argument(  # BEV 맵을 예측할 깊이 간격을 지정(z)
        "--z-intervals",  # -> 각 깊이 레베렝서 주변 환경을 예측하고 BEV 맵을 생성한다..?
        type=float,
        nargs="+",
        default=[1.0, 9.0, 21.0, 39.0, 51.0],
        help="depths at which to predict BEV maps",
    )
    parser.add_argument(  # grid 좌표계에 적용되는 무작위 노이즈 크기를 설정
        "--grid-jitter",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="magn. of random noise applied to grid coords",
    )
    parser.add_argument(  # training 중 적용되는 무작위로 이미지 자르기 설정
        "--aug-image-size",  # -> 모델이 다양한 이미지에서 효과적으로 학습되도록 돕는다.
        type=int,
        nargs="+",
        default=[1280, 720],
        help="size of random image crops during training",
    )
    parser.add_argument(  # 이미지 크기 조정
        "--desired-image-size",  # -> 네트워크의 입력 크기를 표준화하기 위해 사용
        type=int,
        nargs="+",
        default=[1600, 900],
        help="size images are padded to before passing to network",
    )
    parser.add_argument(  # grid의 수평 평면에서의 수직 offset을 나타냄
        "--yoffset",  # -> 카메라 축에서 얼마나 떨어져 있는지
        type=float,  # -> 즉, 네트워크가 입력 이미지에서 추출한 특징을 해당 위치에 투영하여 사용하는 방식에 영향을 줌
        default=1.74,  # 그리드는 카메라 축으로부터 1.74 단위 떨어진 위치에 배치됨
        help="vertical offset of the grid from the camera axis",
    )

    # -------------------------- Model options -------------------------- #
    parser.add_argument(  # 모델 이름
        "--model-name",
        type=str,
        default="PyrOccTranDetr_S_0904_old_rep100x100_out100x100",
        help="Model to train",
    )
    parser.add_argument(  # 그리드 셀 사이즈(m)
        "-r",
        "--grid-res",
        type=float,
        default=0.5,
        help="size of grid cells, in meters",
    )
    parser.add_argument(  # ResNet frontend 아키텍처 이름
        "--frontend",  # -> 이미지의 저수준 특징을 추출하는 데 사용
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet34", "resnet50"],
        help="name of frontend ResNet architecture",
    )
    parser.add_argument(  # ResNet frontend를 사전 훈련된 가중치로 초기화할 지 여부
        "--pretrained",
        type=bool,
        default=True,
        help="choose pretrained frontend ResNet",
    )
    parser.add_argument(  # BEV 추정 모델을 사전 훈련된 가중치로 초기화할 지 여부
        "--pretrained-bem",
        type=bool,
        default=False,
        help="choose pretrained BEV estimation model",
    )
    parser.add_argument(  # 미리 훈련된 모델을 로드할 때 사용할 모델 이름 지정
        "--pretrained-model",  # -> 지정된 이름은 특정 디렉토리에서 해당 모델의 가중치를 찾을 때 사용됨
        type=str,
        default="iccv_segdet_pyrocctrandetr_s_0904_100x100_200down100up_dice_adam_lr5e5_di3_1600x900",
        help="name of pretrained model to load",
    )
    parser.add_argument(  # 체크포인트의 이름
        "--load-ckpt",
        type=str,
        default="checkpoint-0020.pth.gz",
        help="name of checkpoint to load",
    )
    parser.add_argument(  # 미리 훈련된 모델의 가중치를 로드할 때 무시할 모듈을 지정
        "--ignore", type=str, default=["nothing"], help="pretrained modules to ignore",
    )
    parser.add_argument(
        "--ignore-reload",
        type=str,
        default=["nothing"],  # 특정 모듈을 무시하고 싶을 때 해당 모듈의 이름을 추가
        help="pretrained modules to ignore",
    )
    parser.add_argument(  # focal length의 길이
        "--focal-length", type=float, default=1266.417, help="focal length",
    )
    parser.add_argument(  # ResNet frontend의 각 레이어의 크기를 설정
        "--scales",
        type=float,
        nargs=4,
        default=[8.0, 16.0, 32.0, 64.0],
        help="resnet frontend scale factor",
    )
    parser.add_argument(  # ResNet frontend의 특징 맵을 자르는 높이 설정
        "--cropped-height",
        type=float,
        nargs=4,
        default=[20.0, 20.0, 20.0, 20.0],
        help="resnet feature maps cropped height",
    )
    parser.add_argument(  # 모든 깊이 간격에 대한 world space에서의 최대 y차원 설정
        "--y-crop",  # -> BEV를 생성하는 동안 처리되는 세로 차원의 최대 크기
        type=float,
        nargs=4,
        default=[15, 15.0, 15.0, 15.0],
        help="Max y-dimension in world space for all depth intervals",
    )
    parser.add_argument(  # topdown 네트워크에 대한 입력 정규화 방법 지정
        "--dla-norm",
        type=str,
        default="GroupNorm",  # 배치 정규화랑 비슷함
        help="Normalisation for inputs to topdown network",
    )
    parser.add_argument(  # BEVT 모델의 linear 레이어에 배치 정규화, ReLU, Dropout을 추가할 지 여부를 지정
        "--bevt-linear-additions",  # ----> BEVT 모델 논문에서 알아보기
        type=str2bool,
        default=False,
        help="BatchNorm, ReLU and Dropout addition to linear layer in BEVT",
    )
    parser.add_argument(  # BEVT 모델의 conv에 대해 배치 정규화, ReLU, Dropout을 추가할 지 여부를 지정
        "--bevt-conv-additions",
        type=str2bool,
        default=False,
        help="BatchNorm, ReLU and Dropout addition to conv layer in BEVT",
    )
    parser.add_argument(  # DLA 모델의 첫 번째 conv layer에서의 채널 수 설정
        "--dla-l1-nchannels",  # -> 네트워크의 초기 학습 단계에서 입력 데이터를 처리하는 데 사용되는 채널의 수 설정
        type=int,  # ----> DLA 모델 논문에서 알아보기
        default=64,
        help="vertical offset of the grid from the camera axis",
    )
    parser.add_argument(  # encoder 수 지정(2)
        "--n-enc-layers",
        type=int,
        default=2,
        help="number of transfomer encoder layers",
    )
    parser.add_argument(  # deconder 수 지정(2)
        "--n-dec-layers",
        type=int,
        default=2,
        help="number of transformer decoder layers",
    )

    # ---------------------------- Loss options ---------------------------- #
    parser.add_argument(
        "--loss", type=str, default="dice_loss_mean", help="Loss function",
    )
    parser.add_argument(
        "--exp-cf",
        type=float,
        default=0.0,
        help="Exponential for class frequency in weighted dice loss",
    )
    parser.add_argument(
        "--exp-os",
        type=float,
        default=0.2,
        help="Exponential for object size in weighted dice loss",
    )

    # ------------------------ Optimization options ----------------------- #
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer")
    parser.add_argument("-l", "--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum for SGD")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument(
        "--lr-decay",
        type=float,
        default=0.99,
        help="factor to decay learning rate by every epoch",
    )

    # ------------------------- Training options ------------------------- #
    parser.add_argument(
        "-e", "--epochs", type=int, default=600, help="number of epochs to train for"
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=1, help="mini-batch size for training"
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=5,
        help="Gradient accumulation over number of batches",
    )

    # ------------------------ Experiment options ----------------------- #
    parser.add_argument(
        "--name", type=str,
        default="tiim_220613",
        help="name of experiment",
    )
    parser.add_argument(
        "-s",
        "--savedir",
        type=str,
        default="experiments",
        help="directory to save experiments to",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=int,
        nargs="*",
        default=[0],
        help="ids of gpus to train on. Leave empty to use cpu",
    )
    parser.add_argument(
        "--num-gpu", type=int, default=1, help="number of gpus",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=4,
        help="number of worker threads to use for data loading",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=1,
        help="number of epochs between validation runs",
    )
    parser.add_argument(
        "--print-iter",
        type=int,
        default=5,
        help="print loss summary every N iterations",
    )
    parser.add_argument(
        "--vis-iter",
        type=int,
        default=20,
        help="display visualizations every N iterations",
    )
    return parser.parse_args()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def _make_experiment(args):
    print("\n" + "#" * 80)
    print(datetime.now().strftime("%A %-d %B %Y %H:%M"))  # 현재 날짜와 시간
    print(  # 실험 이름과 디렉토리 정보
        "Creating experiment '{}' in directory:\n  {}".format(args.name, args.savedir)
    )
    print("#" * 80)
    print("\nConfig:")  # 제목
    for key in sorted(args.__dict__):
        print("  {:12s} {}".format(key + ":", args.__dict__[key]))
    print("#" * 80)

    # Create a new directory for the experiment
    # 실험 디렉토리 생성
    savedir = os.path.join(args.savedir, args.name)
    os.makedirs(savedir, exist_ok=True)  # 이미 디렉토리가 존재하면 pass

    # # Create tensorboard summary writer
    summary = SummaryWriter(savedir)

    # # Save configuration to file
    # 실험 설정을 txt파일로 저장
    with open(os.path.join(savedir, "config.txt"), "w") as fp:
        json.dump(args.__dict__, fp)

    # # Write config as a text summary
    # summary.add_text(
    #     "config",
    #     "\n".join("{:12s} {}".format(k, v) for k, v in sorted(args.__dict__.items())),
    # )
    # summary.file_writer.flush()

    return None


def save_checkpoint(args, epoch, model, optimizer, scheduler):
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    ckpt_file = os.path.join(
        args.savedir, args.name, "checkpoint-{:04d}.pth.gz".format(epoch)
    )
    print("==> Saving checkpoint '{}'".format(ckpt_file))
    torch.save(ckpt, ckpt_file)


def main():
    # Parse command line arguments
    args = parse_args()
    args.root = os.path.join(os.getcwd(), args.root)  # 전체 경로
    print(args.root)
    args.savedir = os.path.join(os.getcwd(), args.savedir)  # savedir 전체 경로
    print(args.savedir)

    # Build depth intervals along Z axis and reverse
    # z축에 따라 깊이 간격 구축, 그 간격을 역으로 만들어 grid_size에 할당
    z_range = args.z_intervals
    args.grid_size = (z_range[-1] - z_range[0], z_range[-1] - z_range[0])

    # Calculate cropped heights of feature maps
    # feature map의 잘린 높이 계산
    h_cropped = src.utils.calc_cropped_heights(
        args.focal_length, np.array(args.y_crop), z_range, args.scales
    )
    args.cropped_height = [h for h in h_cropped]

    num_gpus = torch.cuda.device_count()
    args.num_gpu = num_gpus

    ### Create experiment ###
    # 훈련 및 검증 데이터셋을 로드
    summary = _make_experiment(args)  # 실험을 생성하는 함수 -> 실험 정보 출력

    # train data 로드
    print("loading train data")
    # Create datasets
    train_data = nuScenesMaps(
        root=args.root,  # dataset의 root directory
        split=args.train_split,  # train dataset 분할
        grid_size=args.grid_size,  # grid 크기
        grid_res=args.grid_res,  # 해상도
        classes=args.load_classes_nusc,  # class
        dataset_size=args.data_size,  # dataset 크기
        desired_image_size=args.desired_image_size,  # 원하는 이미지 크기
        mini=True,  # mini dataset 사용 여부
        gt_out_size=(100, 100),  # 출력 크기
    )

    # val data 로드
    print("loading val data")
    val_data = nuScenesMaps(
        root=args.root,
        split=args.val_split,
        grid_size=args.grid_size,
        grid_res=args.grid_res,
        classes=args.load_classes_nusc,
        dataset_size=args.data_size,
        desired_image_size=args.desired_image_size,
        mini=True,
        gt_out_size=(200, 200),
    )

    # Create dataloaders
    # train dataloader
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,  # 병렬 작업 수 설정
        collate_fn=src.data.collate_funcs.collate_nusc_s,  # 미니 배치로 묶을 때 사용할 함수
        drop_last=True,
    )

    # val dataloader
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=src.data.collate_funcs.collate_nusc_s,
        drop_last=True,
    )

    # Build model
    model = networks.__dict__[args.model_name](
        num_classes=len(args.pred_classes_nusc),  # 모델의 출력 클래스 수 설정: 모델이 예측해야하는 클래스 목록의 수 만큼
        frontend=args.frontend,
        grid_res=args.grid_res,  # 모델이 사용하는 그리드의 해상도 설정
        pretrained=args.pretrained,  # 사전 훈련된 가중치 사용 여부
        img_dims=args.desired_image_size,  # 입력 이미지의 크기 설정
        z_range=z_range,  # 카메라 시점에서의 깊이(Z축) 범위 설정
        h_cropped=args.cropped_height,  # 이미지 특징 맵의 높이 설정
        dla_norm=args.dla_norm,  # 정규화 방법 설정
        additions_BEVT_linear=args.bevt_linear_additions,
        additions_BEVT_conv=args.bevt_conv_additions,
        dla_l1_n_channels=args.dla_l1_nchannels,  # L1 level에서 사용할 채널 수
        n_enc_layers=args.n_enc_layers,  # encoder layer 수
        n_dec_layers=args.n_dec_layers,  # decoder layer 수
    )

    # 사전 훈련된 모델의 가중치를 현재 모델에 불러오기
    if args.pretrained_bem:
        pretrained_model_dir = os.path.join(args.savedir, args.pretrained_model)
        # pretrained_ckpt_fn = sorted(
        #     [
        #         f
        #         for f in os.listdir(pretrained_model_dir)
        #         if os.path.isfile(os.path.join(pretrained_model_dir, f))
        #         and ".pth.gz" in f
        #     ]
        # )
        pretrained_pth = os.path.join(pretrained_model_dir, args.load_ckpt)
        pretrained_dict = torch.load(pretrained_pth)["model"]
        mod_dict = OrderedDict()

        # # Remove "module" from name
        for k, v in pretrained_dict.items():
            if any(module in k for module in args.ignore):
                continue
            else:
                name = k[7:]
                mod_dict[name] = v

        model.load_state_dict(mod_dict, strict=False)
        print("loaded pretrained model")

    device = torch.device("cuda")
    model = nn.DataParallel(model)  # 모델을 병렬로 처리할 수 있도록 하는 모듈
    model.to(device)

    # Setup optimizer
    # Optimizer 설정
    if args.optimizer == "adam":  # adam 으로 설정할 경우
        optimizer = optim.Adam(model.parameters(), args.lr, )
    else:
        optimizer = optim.__dict__[args.optimizer](
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    # learning rate scheduler 설정 -> 학습률을 동적으로 조정 가능
    # ExponentialLR: 각 epoch이 끝날 때마다 현재 학습률을 비율(lr_decay)로 감소
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)

    # Check if saved model checkpoint exists
    # 학습을 재개할 떄 저장된 모델 체크포인트 파일이 있는지 확인
    model_dir = os.path.join(args.savedir, args.name)
    checkpt_fn = sorted(
        [
            f
            for f in os.listdir(model_dir)
            if os.path.isfile(os.path.join(model_dir, f)) and ".pth.gz" in f
        ]
    )
    # 체크포인트가 있다면 로드하여 이전의 학습 상태 복원
    if len(checkpt_fn) != 0:
        model_pth = os.path.join(model_dir, checkpt_fn[-1])
        ckpt = torch.load(model_pth)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["scheduler"])
        epoch_ckpt = ckpt["epoch"]
        print("starting training from {}".format(checkpt_fn[-1]))
    else:
        epoch_ckpt = 1
        pass

    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # 학습 루프
    for epoch in range(epoch_ckpt, args.epochs + 1):  # 저장된 체크포인트부터 지정된 epoch까지 학습

        print("\n=== Beginning epoch {} of {} ===".format(epoch, args.epochs))

        # # Train model
        train(args, train_loader, model, optimizer, epoch)

        # Run validation every N epochs
        # cal_interval 마다 검증 실행
        if epoch % args.val_interval == 0:
            # Save model checkpoint
            # save_checkpoint(args, epoch, model, optimizer, scheduler)
            validate(args, val_loader, model, epoch)

        # Update and log learning rate
        scheduler.step()


if __name__ == "__main__":
    main()
