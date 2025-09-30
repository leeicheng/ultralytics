"""Point detection predictor for YOLO point-based models."""

import torch

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class PointDetectionPredictor(BasePredictor):
    """Predictor for point-detection models that outputs (x, y, conf, cls) per detection."""

    @staticmethod
    def _point_nms_per_image(x: torch.Tensor, conf_thres: float, radius: float, max_det: int, class_filter=None):
        """Greedy radius-based NMS on points. x shape: (A, 2+nc). Returns (N,4) [x,y,conf,cls]."""
        if x.numel() == 0:
            return x.new_zeros((0, 4))
        conf, class_id = x[:, 2:].max(1)
        keep = conf > conf_thres
        if class_filter is not None:
            keep &= torch.isin(class_id, class_filter.to(class_id.device))
        x = torch.stack((x[:, 0], x[:, 1], conf, class_id.float()), dim=1)[keep]
        if not x.shape[0]:
            return x
        out = []
        for c in x[:, 3].unique():
            m = x[:, 3] == c
            pts = x[m]
            pts = pts[pts[:, 2].argsort(descending=True)]
            keep_pts = []
            while pts.shape[0]:
                best = pts[0]
                keep_pts.append(best)
                if pts.shape[0] == 1:
                    break
                d = torch.cdist(best[:2].unsqueeze(0), pts[1:, :2]).squeeze(0)
                pts = pts[1:][d >= radius]
            out.append(torch.stack(keep_pts))
        x = torch.cat(out, 0)
        x = x[x[:, 2].argsort(descending=True)]
        return x[:max_det]

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Post-process raw model outputs (points+cls) into scaled points with per-class radius-NMS."""
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        bs = preds.shape[0]
        results = []
        class_filter = None
        if self.args.classes is not None:
            class_filter = torch.as_tensor(self.args.classes, dtype=torch.long, device=preds.device)

        for i in range(bs):
            p = preds[i]  # (A, 2+nc)
            # point NMS per image
            det = self._point_nms_per_image(
                p, conf_thres=self.args.conf, radius=getattr(self.args, "radius", 10.0), max_det=self.args.max_det,
                class_filter=class_filter,
            )  # (N, 4) [x,y,conf,cls]

            # scale points back to original image size
            if det.shape[0]:
                ops.scale_coords(img.shape[2:], det[:, :2], orig_imgs[i].shape)

            results.append(self.construct_result(det, img, orig_imgs[i], self.batch[0][i]))
        return results

    def construct_result(self, pred, img, orig_img, img_path):
        """Build Results using keypoints field with K=1 per detection: (x, y, conf)."""
        if pred is None or pred.numel() == 0:
            return Results(orig_img, path=img_path, names=self.model.names)
        # Prepare keypoints tensor (N, 1, 3)
        kpts = torch.zeros((pred.shape[0], 1, 3), device=pred.device, dtype=pred.dtype)
        kpts[:, 0, 0:2] = pred[:, 0:2]
        kpts[:, 0, 2] = pred[:, 2].clamp_min(0.51)
        r = Results(orig_img, path=img_path, names=self.model.names, keypoints=kpts)
        # Attach class and confidence for downstream usage
        r.point_cls = pred[:, 3].clone()
        r.point_conf = pred[:, 2].clone()
        return r


    def write_results(self, i, p, im, s):
        """Write results, extended to save points JSON if requested."""
        string = ""
        if len(im.shape) == 3:
            im = im[None]
        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:
            string += f"{i}: "
            frame = self.dataset.count
        else:
            import re
            match = re.search(r"frame (\d+)/", s[i])
            frame = int(match[1]) if match else None

        self.txt_path = self.save_dir / "labels" / (p.stem + ("" if self.dataset.mode == "image" else f"_{frame}"))
        string += "{:g}x{:g} ".format(*im.shape[2:])
        result = self.results[i]
        result.save_dir = self.save_dir.__str__()
        string += f"{result.verbose()}{result.speed['inference']:.1f}ms"

        if self.args.save or self.args.show:
            self.plotted_img = result.plot(
                line_width=self.args.line_width,
                boxes=self.args.show_boxes,
                conf=self.args.show_conf,
                labels=self.args.show_labels,
                im_gpu=None if self.args.retina_masks else im[i],
            )

        if self.args.save_txt:
            result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
        if getattr(self.args, "save_json", False):
            from pathlib import Path as _Path
            json_file = _Path(f"{self.txt_path}.json")
            json_file.parent.mkdir(parents=True, exist_ok=True)
            with open(json_file, "w", encoding="utf-8") as f:
                f.write(result.to_json(normalize=True))
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem)
        if self.args.show:
            self.show(str(p))
        if self.args.save:
            self.save_predicted_images(str(self.save_dir / p.name), frame)

        return string
