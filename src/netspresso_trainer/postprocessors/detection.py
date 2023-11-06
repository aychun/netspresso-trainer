import torch
import torchvision

from ..models.utils import ModelOutput


class DetectionPostprocessor:
    def __init__(self):
        pass

    def __call__(self, outputs: ModelOutput, original_shape, num_classes, conf_thresh=0.7, nms_thre=0.45, class_agnostic=False):
        pred = outputs['pred']
        dtype = pred[0].type()
        stage_strides= [original_shape[-1] // o.shape[-1] for o in pred]
        
        pred = self.decode_outputs(pred, dtype=dtype, stage_strides=stage_strides)
        pred = self.postprocess(pred, num_classes=num_classes, conf_thre=conf_thresh, nms_thre=nms_thre, class_agnostic=class_agnostic)
        return pred

    def decode_outputs(self, outputs, dtype, stage_strides):
        hw = [x.shape[-2:] for x in outputs]
        # [batch, n_anchors_all, num_classes + 5]
        outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
        outputs[..., 4:] = outputs[..., 4:].sigmoid()

        grids = []
        strides = []
        for (hsize, wsize), stride in zip(hw, stage_strides):
            yv, xv = torch.meshgrid(torch.arange(hsize), torch.arange(wsize), indexing='ij')
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs = torch.cat([
            (outputs[..., 0:2] + grids) * strides,
            torch.exp(outputs[..., 2:4]) * strides,
            outputs[..., 4:]
        ], dim=-1)
        return outputs

    def postprocess(self, prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [torch.zeros(0, 7).to(prediction.device) for i in range(len(prediction))]
        for i, image_pred in enumerate(prediction):

            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            if not detections.size(0):
                continue

            if class_agnostic:
                nms_out_index = torchvision.ops.nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    nms_thre,
                )
            else:
                nms_out_index = torchvision.ops.batched_nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    detections[:, 6],
                    nms_thre,
                )

            detections = detections[nms_out_index]
            output[i] = torch.cat((output[i], detections))

        return output
