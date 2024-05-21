# Ultralytics YOLO 🚀, AGPL-3.0 license
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from .metrics import bbox_iou
from .tal import bbox2dist


class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                    .mean(1)
                    .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    '''
    target_scores：shape=[3,8400,1]
    '''

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""

        '''
        就是把预测框中正样本的target_scores挑出来。
        weight：shape=[101,1]。
        '''
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:  # True
            '''
            输入：
            anchor_points：shape=[8400,2]的Tensor，是在3张特征图上的anchor点xy。
            target_bboxes：shape=[3,8400,4]的Tensor，里面是每个预测框对应的真实框，xyxy的、未归一化的、3张特征图上的。
            输出：
            target_ltrb：shape=[3,8400,4]，3张特征图上的、未归一化的、ltrb。里面是每个预测框对应的真实框。
            '''
            #
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            '''
            预测框与真实框的，ltrb的不同。
            输入：
            target_ltrb[fg_mask]：shape=[101,4]。
            pred_dist[fg_mask]：shape=[101,64]。pred_dist[fg_mask].view(-1, self.reg_max + 1)：shape=[404,16]
            输出：
            loss_dfl：shape=[101,1]。就是每对(预测框,真实框)得到一个ltrb的损失数字。
            '''
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        # tl（整数）《——wr（小数）——》target（小数）《——wl（小数）——》tr（整数）

        '''
        target：
            target_ltrb[fg_mask]：shape=[101,4]。
        pred_dist：
            pred_dist[fg_mask]：shape=[101,64]。pred_dist[fg_mask].view(-1, self.reg_max + 1)：shape=[404,16]


        '''
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
                F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl  # shape=[101,4]
                + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr  # shape=[101,4]
        ).mean(-1, keepdim=True)  # shape=[101,1]


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    '''
    N个真实物体，N个预测物体，一一对应的。
    计算每对(真实物体，预测物体)的kpt_iou，1-kpt_iou得到每对(真实物体，预测物体)的关键点损失，N对物体的关键点损失相加。
    (1 - torch.exp(-e))是因为，类似1-kpt_iou，就是kpt_iou越小，损失越大。最后返回的是损失，不是kpt_iou。

    pred_kpts：shape=[101,1,3]，浮点数。
    gt_kpts：shape=[101,1,3]，浮点数。
    kpt_mask：shape=[101,1]，True/False。
    area：shape=[101,1]，浮点数。
    '''
    '''
    kpt_mask.shape[1]：1。
    (torch.sum(kpt_mask != 0, dim=1) + 1e-9)：shape=101的Tensor。
    kpt_loss_factor：shape=101的Tensor，这里都是1。
    e：shape=[101,1]的Tensor。
    (1 - torch.exp(-e))：shape=[101,1]的Tensor。


    '''

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor, target_kpts=None):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        out_kpts = None
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
            out_kpts = torch.zeros(batch_size, 0, target_kpts.shape[1], target_kpts.shape[2], device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            out_kpts = torch.zeros(batch_size, counts.max(), target_kpts.shape[1], target_kpts.shape[2],
                                   device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
                    out_kpts[j, :n] = target_kpts[matches]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
            out_kpts[..., 0:2] = out_kpts[..., 0:2].mul_(scale_tensor[0:2])

        return out, out_kpts

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(
            gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
            self,
            fg_mask: torch.Tensor,
            masks: torch.Tensor,
            target_gt_idx: torch.Tensor,
            target_bboxes: torch.Tensor,
            batch_idx: torch.Tensor,
            proto: torch.Tensor,
            pred_masks: torch.Tensor,
            imgsz: torch.Tensor,
            overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        # sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.from_numpy(
            np.array([1.0]) / 10.0).to(self.device)
        self.keypoint_loss = KeypointLoss(sigmas=self.sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets, gt_keypoints = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]],
                                                # target_kpts=torch.squeeze(batch['keypoints'][:,:,0:2], dim=1).to(self.device).float().clone())
                                                target_kpts=batch['keypoints'].to(self.device).float().clone())
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        '''
        走到utils/loss.py的183行bbox_decode()方法。
        传参：anchor_points：shape=[8400,2]，是8400个anchor点，在3张特征图上的坐标；
        pred_dist：shape=[3,8400,64]，未解码的预测框。
        if self.use_dfl:，为True。
        b, a, c = pred_dist.shape  # batch, anchors, channels，得到，b=3，a=8400，c=64。
        187行pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))，
        其中c//4=64//4=16；
        pred_dist.view(b, a, 4, c // 4)，得到shape=[3,8400,4,16]的Tensor；
        pred_dist.view(b, a, 4, c // 4).softmax(3)，其中.softmax(3)的3是dim=3，
        .softmax()函数，"它应用于沿 dim 的所有切片，并将重新缩放它们，使元素位于[0, 1]范围内且总和为 1。"，
        得到shape=[3,8400,4,16]的Tensor。
        pred_dist.dtype，是torch.float16。
        self.proj是tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14., 15.], device='cuda:0')。
        self.proj.type(pred_dist.dtype)得到tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14., 15.], device='cuda:0', dtype=torch.float16)。
        .matmul()函数是矩阵乘法。
        得到pred_dist={Tensor:3}，shape=[3,8400,4]。
        '''
        # pred_bboxes：shape=[3,8400,4]，是没有归一化的，在3张特征图上的。
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # pred_kpts：shape=[3,8400,1,2]，是没有归一化的，在3张特征图上的。
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        # pd_kkpts = torch.squeeze(pred_kpts, dim=2)[..., 0:2] # (3,8400,2)
        pd_kkpts = pred_kpts.to(self.device).float().clone()  # 必须clone()，不然计算完pd_kkpts，会使得pred_kpts的值也改变。
        # pd_kkpts[..., 0:2] = (pd_kkpts[..., 0:2].detach() * stride_tensor).type(gt_keypoints.dtype) # (3,8400,1,2)
        # ndim = self.kpt_shape[1]
        # pd_kkpts[:, 0::ndim] = pd_kkpts[:, 0::ndim] * stride_tensor
        # pd_kkpts[:, 1::ndim] = pd_kkpts[:, 1::ndim] * stride_tensor

        '''
        stride_tensor：shape=[8400,1]
        pd_kkpts：shape=[3,8400,1,3]
        pd_kkpts[..., 0]：shape=[3,8400,1]
        pd_kkpts[..., 1]：shape=[3,8400,1]
        '''
        pd_kkpts[..., 0] = (pd_kkpts[..., 0].detach() * stride_tensor).type(gt_keypoints.dtype)
        pd_kkpts[..., 1] = (pd_kkpts[..., 1].detach() * stride_tensor).type(gt_keypoints.dtype)

        '''
        正样本满足以下条件：
        1、anchor点在这个真值框中，
        2、anchor点生成的预测框与这个真值框，的CIoU×score，在所有【在这个真值框内的anchor点生成的预测框中】，是前topk(10)大。
        3、如果一个预测框anchor box被分配到多个真值框gts, CIoU最高的那一个真值框将被选择。

        target_labels：shape=[3,8400]，里面的数都是0。
        target_bboxes：shape=[3,8400,4]，是没有归一化的，在原图上的[640,640]，xyxy的。
        target_scores：shape=[3,8400,1]，正样本的预测框，都会有个值，这个值由如下2值得到：预测框与其对应真值框的CIoU(主要)、预测框的分数(很少)。非正样本的预测框位置填0。
        fg_mask：shape=[3,8400]，里面0/1，1表示这个预测框有对应到一个真值框，0表示这个预测框没有对应到一个真值框。
        target_gt_idx：shape=[3,8400]，表示这个预测框对应到0-7这8个真值框的哪一个。
        '''
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),  # pred_scores：shape=[3,8400,1]
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            # pred_bboxes：shape=[3,8400,4]，是没有归一化的，在3张特征图上的，xyxy的。* stride_tensor，映射回640×640的图上，没有归一化。
            anchor_points * stride_tensor,  # shape=[8400,2]，把3张特征图上得到的8400个anchor点，映射到640×640的原图上去。
            gt_labels,  # shape=[3,8,1]，都是0。
            gt_bboxes,  # shape=[3,8,4]，是没有归一化的，在原图上的[640,640]，xyxy的。
            mask_gt,  # shape=[3,8,1]。
            gt_kkpts=gt_keypoints,  # gt_keypoints：shape=[3,8,2]，是没有归一化的，在原图上的[640,640]。
            # pd_kkpts=(pd_kkpts.detach() * stride_tensor).type(gt_keypoints.dtype),
            pd_kkpts=pd_kkpts,
            # pd_kkpts：shape=[3,8400,2]，是没有归一化的，在3张特征图上的。* stride_tensor，映射回640×640的图上，没有归一化。
            sigma=self.sigmas
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            # pred_bboxes、target_bboxes，都是3张特征图上的，用来计算CIoU。
            loss[0], loss[4] = self.bbox_loss(
                pred_distri,  # pred_distri：shape=[3,8400,64]。解码前的box。
                pred_bboxes,  # pred_bboxes：{Tensor:3}，shape=[3,8400,4]，是预测的、没有归一化的、xyxy的、框是在3张特征图上的 bbox。
                anchor_points,  # shape=[8400,2]，3张特征图上的8400个anchor点。
                target_bboxes,
                # target_bboxes：shape=[3,8400,4]的Tensor，是3张图，每张图8400个预测框，每个预测框对应到的真值框的bbox；bbox是，没有归一化的、是xyxy的、框是在3张特征图上的。
                target_scores,
                # target_scores：shape=[3,8400,1]，正样本的预测框，都会有个值，这个值由如下2值得到：预测框与其对应真值框的CIoU(主要)、预测框的分数(很少)。非正样本的预测框位置填0。
                target_scores_sum,
                fg_mask,  # fg_mask：shape=[3,8400]，里面0/1，1表示这个预测框有对应到一个真值框，0表示这个预测框没有对应到一个真值框。
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            '''
            fg_mask：shape=[3,8400]，里面0/1，1表示这个预测框有对应到一个真值框，0表示这个预测框没有对应到一个真值框。
            target_gt_idx：shape=[3,8400]，表示这个预测框对应到0-7这8个真值框的哪一个。
            keypoints
            batch_idx
            stride_tensor
            target_bboxes：shape=[3,8400,4]的Tensor，是3张图，每张图8400个预测框，每个预测框对应到的真值框的bbox；bbox是，没有归一化的、是xyxy的、框是在3张特征图上的。
            pred_kpts：shape=[3,8400,1,2]，是没有归一化的，在3张特征图上的。
            '''
            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        # loss[0] *= 0  # box gain
        # loss[1] *= self.hyp.pose  # pose gain
        # loss[2] *= 0  # kobj gain
        # loss[3] *= 0  # cls gain
        # loss[4] *= 0  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    '''
    https://github.com/ultralytics/ultralytics/issues/8443
    *2是为了把xy坐标范围从[-0.5,0.5]转成[-1.0,1.0]
    '''

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
            self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            (tuple): Returns a tuple containing:
                - kpts_loss (torch.Tensor): The keypoints loss.
                - kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        '''
        selected_keypoints：shape=[3,8400,1,3]，当前预测关键点对应到的真值关键点的数据xyv，xyv都是在3个特征图上。
        pred_kpts：shape=[3,8400,1,2]，是没有归一化的，在3张特征图上的。
        target_bboxes：shape=[3,8400,4]的Tensor，是3张图，每张图8400个预测框，每个预测框对应到的真值框的bbox；bbox是，没有归一化的、是xyxy的、框是在3张特征图上的。
        '''
        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            # gt_kpt：shape=[101,1,3]
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            # kpt_mask：如果是xyv，那么v=0为False，v=1或2都是True；如果是xy，那么所有点都认为是True。
            # 因为gt_kpt：shape=[101,1,3]，所以kpt_mask：if或else都得到shape=[101,1]的Tensor。
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            # pred_kpt、gt_kpt、area 都是3张特征图上的。用来计算kpt_iou。
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = torch.nn.functional.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


class MultiTaskLoss(v8DetectionLoss):

    def __init__(self, model):  # model must be de-paralleled
        super().__init__(model)
        self.pose_loss = v8PoseLoss(model)
        self.seg_loss = v8SegmentationLoss(model)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        # box_loss, pose_loss, kobj_loss, seg_loss, cls_loss, dfl_loss
        loss = torch.zeros(6, device=self.device)
        feats, pred_kpts, pred_masks, proto = preds if len(preds) == 4 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        batch_idx = batch['batch_idx'].view(-1, 1)
        targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)


        targets, gt_keypoints = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]],
                                                # target_kpts=torch.squeeze(batch['keypoints'][:,:,0:2], dim=1).to(self.device).float().clone())
                                                target_kpts=batch['keypoints'].to(self.device).float().clone())
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        '''
        走到utils/loss.py的183行bbox_decode()方法。
        传参：anchor_points：shape=[8400,2]，是8400个anchor点，在3张特征图上的坐标；
        pred_dist：shape=[3,8400,64]，未解码的预测框。
        if self.use_dfl:，为True。
        b, a, c = pred_dist.shape  # batch, anchors, channels，得到，b=3，a=8400，c=64。
        187行pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))，
        其中c//4=64//4=16；
        pred_dist.view(b, a, 4, c // 4)，得到shape=[3,8400,4,16]的Tensor；
        pred_dist.view(b, a, 4, c // 4).softmax(3)，其中.softmax(3)的3是dim=3，
        .softmax()函数，"它应用于沿 dim 的所有切片，并将重新缩放它们，使元素位于[0, 1]范围内且总和为 1。"，
        得到shape=[3,8400,4,16]的Tensor。
        pred_dist.dtype，是torch.float16。
        self.proj是tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14., 15.], device='cuda:0')。
        self.proj.type(pred_dist.dtype)得到tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14., 15.], device='cuda:0', dtype=torch.float16)。
        .matmul()函数是矩阵乘法。
        得到pred_dist={Tensor:3}，shape=[3,8400,4]。
        '''
        # pred_bboxes：shape=[3,8400,4]，是没有归一化的，在3张特征图上的。
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # pred_kpts：shape=[3,8400,1,2]，是没有归一化的，在3张特征图上的。
        pred_kpts = self.pose_loss.kpts_decode(anchor_points,
                                               pred_kpts.view(batch_size, -1,
                                                              *self.pose_loss.kpt_shape))  # (b, h*w, 17, 3)

        # pd_kkpts = torch.squeeze(pred_kpts, dim=2)[..., 0:2] # (3,8400,2)
        pd_kkpts = pred_kpts.to(self.device).float().clone()  # 必须clone()，不然计算完pd_kkpts，会使得pred_kpts的值也改变。
        # pd_kkpts[..., 0:2] = (pd_kkpts[..., 0:2].detach() * stride_tensor).type(gt_keypoints.dtype) # (3,8400,1,2)
        # ndim = self.kpt_shape[1]
        # pd_kkpts[:, 0::ndim] = pd_kkpts[:, 0::ndim] * stride_tensor
        # pd_kkpts[:, 1::ndim] = pd_kkpts[:, 1::ndim] * stride_tensor

        '''
        stride_tensor：shape=[8400,1]
        pd_kkpts：shape=[3,8400,1,3]
        pd_kkpts[..., 0]：shape=[3,8400,1]
        pd_kkpts[..., 1]：shape=[3,8400,1]
        '''
        pd_kkpts[..., 0] = (pd_kkpts[..., 0].detach() * stride_tensor).type(gt_keypoints.dtype)
        pd_kkpts[..., 1] = (pd_kkpts[..., 1].detach() * stride_tensor).type(gt_keypoints.dtype)

        '''
        正样本满足以下条件：
        1、anchor点在这个真值框中，
        2、anchor点生成的预测框与这个真值框，的CIoU×score，在所有【在这个真值框内的anchor点生成的预测框中】，是前topk(10)大。
        3、如果一个预测框anchor box被分配到多个真值框gts, CIoU最高的那一个真值框将被选择。

        target_labels：shape=[3,8400]，里面的数都是0。
        target_bboxes：shape=[3,8400,4]，是没有归一化的，在原图上的[640,640]，xyxy的。
        target_scores：shape=[3,8400,1]，正样本的预测框，都会有个值，这个值由如下2值得到：预测框与其对应真值框的CIoU(主要)、预测框的分数(很少)。非正样本的预测框位置填0。
        fg_mask：shape=[3,8400]，里面0/1，1表示这个预测框有对应到一个真值框，0表示这个预测框没有对应到一个真值框。
        target_gt_idx：shape=[3,8400]，表示这个预测框对应到0-7这8个真值框的哪一个。
        '''
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),  # pred_scores：shape=[3,8400,1]
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            # pred_bboxes：shape=[3,8400,4]，是没有归一化的，在3张特征图上的，xyxy的。* stride_tensor，映射回640×640的图上，没有归一化。
            anchor_points * stride_tensor,  # shape=[8400,2]，把3张特征图上得到的8400个anchor点，映射到640×640的原图上去。
            gt_labels,  # shape=[3,8,1]，都是0。
            gt_bboxes,  # shape=[3,8,4]，是没有归一化的，在原图上的[640,640]，xyxy的。
            mask_gt,  # shape=[3,8,1]。
            gt_kkpts=gt_keypoints,  # gt_keypoints：shape=[3,8,2]，是没有归一化的，在原图上的[640,640]。
            # pd_kkpts=(pd_kkpts.detach() * stride_tensor).type(gt_keypoints.dtype),
            pd_kkpts=pd_kkpts,
            # pd_kkpts：shape=[3,8400,2]，是没有归一化的，在3张特征图上的。* stride_tensor，映射回640×640的图上，没有归一化。
            sigma=self.pose_loss.sigmas,
            stride_tensor=stride_tensor,
            reg_max=self.reg_max,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        loss[4] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.any():
            target_strided_bboxes = target_bboxes / stride_tensor

            # bbox regression loss
            loss[0], loss[5] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_strided_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

            # keypoints loss
            keypoints = batch['keypoints'].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]
            loss[1], loss[2] = self.pose_loss.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch_idx,
                stride_tensor,
                target_strided_bboxes,
                pred_kpts,
            )

            # segmentation loss
            masks = batch['masks'].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode='nearest')[0]

            loss[3] = self.seg_loss.calculate_segmentation_loss(fg_mask, masks, target_gt_idx, target_bboxes, batch_idx,
                                                                proto, pred_masks, imgsz, self.seg_loss.overlap)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.box  # seg gain
        loss[4] *= self.hyp.cls  # cls gain
        loss[5] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()
