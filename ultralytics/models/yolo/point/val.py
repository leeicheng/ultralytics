# Ultralytics PointDetectionValidator
import torch

from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import PointDetMetrics, point_oks
from ultralytics.utils.plotting import plot_images_with_points


class PointDetectionValidator(BaseValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = "point"
        self.metrics = PointDetMetrics(save_dir=self.save_dir)
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.niou = self.iouv.numel()
        self.sigmas = None

    def preprocess(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)
        return batch

    def init_metrics(self, model):
        # Initialize evaluation metrics for point validation
        # Set names and class count explicitly (BaseValidator.init_metrics is a no-op)
        self.names = getattr(model, 'names', {})
        self.nc = len(self.names)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.seen = 0
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
        # Sigmas for OKS (constant per class by default)
        self.sigmas = torch.ones(self.nc, device=self.device) * 0.072
    def get_desc(self):
        # Display as OKS metrics columns
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "OKS(P", "R", "OKS50", "OKS50-95)")

    @staticmethod
    def point_non_max_suppression(prediction, conf_thres=0.25, radius=10.0, max_det=300):
        output = [torch.zeros((0, 4), device=prediction.device)] * prediction.shape[0]
        for i, x in enumerate(prediction):
            conf, class_id = x[:, 2:].max(1, keepdim=True)
            x = torch.cat((x[:, :2], conf, class_id.float()), 1)
            x = x[conf.view(-1) > conf_thres]
            if not x.shape[0]:
                continue
            unique_classes = x[:, 3].unique()
            final_preds_per_image = []
            for c in unique_classes:
                class_mask = x[:, 3] == c
                class_preds = x[class_mask]
                _, sort_idx = class_preds[:, 2].sort(descending=True)
                class_preds = class_preds[sort_idx]
                preds_to_keep = []
                while class_preds.shape[0]:
                    best_pred = class_preds[0]
                    preds_to_keep.append(best_pred)
                    if class_preds.shape[0] == 1:
                        break
                    distances = torch.cdist(best_pred[:2].unsqueeze(0), class_preds[1:, :2]).squeeze(0)
                    keep_mask = distances >= radius
                    class_preds = class_preds[1:][keep_mask]
                if preds_to_keep:
                    final_preds_per_image.append(torch.stack(preds_to_keep))
            if final_preds_per_image:
                final_preds_per_image = torch.cat(final_preds_per_image, 0)
                _, sort_idx = final_preds_per_image[:, 2].sort(descending=True)
                final_preds_per_image = final_preds_per_image[sort_idx]
                output[i] = final_preds_per_image[:max_det]
        return output

    def postprocess(self, preds):
        # Debug logging for preds structure
        try:
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
        except Exception as e:
            LOGGER.warning(f"[PointValidator] postprocess debug log failed: {e}")

        radius = getattr(self.args, "radius", 10.0)
        return self.point_non_max_suppression(
            preds, conf_thres=self.args.conf, radius=radius, max_det=self.args.max_det
        )

    def _prepare_batch(self, si, batch):
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bboxes = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            points = bboxes[:, :2] * torch.tensor(imgsz, device=self.device)[[1, 0]]
            ops.scale_coords(imgsz, points, ori_shape, ratio_pad=ratio_pad)
            areas = (bboxes[:, 2] * imgsz[1]) * (bboxes[:, 3] * imgsz[0])
        else:
            points = bboxes.new_zeros((0, 2))
            areas = bboxes.new_zeros((0,))
        return {
            "cls": cls,
            "points": points,
            "areas": areas,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad,
        }

    def _prepare_pred(self, pred, pbatch):
        predn = pred.clone()
        ops.scale_coords(pbatch["imgsz"], predn[:, :2], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])
        return predn

    def update_metrics(self, preds, batch):
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, points, areas = pbatch.pop("cls"), pbatch.pop("points"), pbatch.pop("areas")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in ("tp",):
                        self.stats[k].append(stat[k])
                continue

            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 2]
            stat["pred_cls"] = predn[:, 3]

            if nl:
                stat["tp"] = self._process_batch(predn, points, cls, areas)
            self.stats["tp"].append(stat["tp"])
            self.stats["conf"].append(stat["conf"])
            self.stats["pred_cls"].append(stat["pred_cls"])
            if nl:
                self.stats["target_cls"].append(cls)
                self.stats["target_img"].append(stat["target_img"])

    def _process_batch(self, detections, gt_points, gt_cls, gt_areas):
        oks = point_oks(gt_points, detections[:, :2], gt_areas, self.sigmas)
        return self.match_predictions(detections[:, 3], gt_cls, oks)

    def plot_val_samples(self, batch, ni):
        # Plot GT points (center of bbox) on the validation batch mosaic
        bboxes = batch["bboxes"]
        points = bboxes[:, :2]
        plot_images_with_points(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            points,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        # Convert preds (list of per-image tensors [x,y,conf,cls]) to batched arrays for plotting
        if not len(preds):
            return
        pts, cls, conf, bidx = [], [], [], []
        for i, p in enumerate(preds):
            if p is None or len(p) == 0:
                continue
            pts.append(p[:, :2])
            conf.append(p[:, 2])
            cls.append(p[:, 3])
            bidx.append(torch.full((p.shape[0],), i, device=p.device, dtype=torch.long))
        if not pts:
            return
        pts = torch.cat(pts, 0)
        cls = torch.cat(cls, 0)
        conf = torch.cat(conf, 0)
        bidx = torch.cat(bidx, 0)
        H, W = batch["img"].shape[2:]
        norm = torch.tensor([W, H], device=pts.device)
        plot_images_with_points(
            batch["img"],
            bidx,
            cls,
            pts / norm,
            confs=conf,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )
