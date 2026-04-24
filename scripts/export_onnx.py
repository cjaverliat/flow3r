import logging
import torch
import torch.onnx

from flow3r.models.flow3r import Flow3r
from flow3r.models.onnx_compat import onnx_export_mode
import os


class Wrapper(torch.nn.Module):
    """
    Wrapper class for DINOV2 model.
    """

    def __init__(self, model: Flow3r):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor):
        ff = self.model.forward(images)
        return ff


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logging.info("Loading model...")
    # Use CPU: CUDA custom ops (cuRoPE2D, FlashAttention backends) can't be
    # serialised to ONNX. The model runs on CPU during export tracing.
    device = torch.device("cpu")
    flow3r = Flow3r.from_pretrained("Clara211111/flow3r", model_kwargs=dict(for_onnx=True)).to(device).eval()

    img_w, img_h = 280, 280
    patch_size = 14

    # B=1, N=2 (num frames), C=3, H, W.
    # H and W must be multiples of 14 (DINOv2 patch size).
    B, N, C, H, W = 1, 2, 3, img_h, img_w  # 280 x 280
    dummy_imgs = torch.randn((B, N, C, H, W), dtype=torch.float32, device=device)
    # One pair per batch: (image 0, image 1). Triggers the DPT flow head.
    dummy_pair_indices = torch.tensor([[[0, 1]]], dtype=torch.long, device=device)

    model = Wrapper(flow3r)

    batch = torch.export.Dim("batch", min=1)
    num_frames = torch.export.Dim("num_frames", min=1)
    height = torch.export.Dim("height", min=140, max=1400)  # Must be multiple of 14
    width = torch.export.Dim("width", min=140, max=1400)  # Must be multiple of 14

    logging.info("Exporting ONNX model...")
    os.makedirs("./outputs", exist_ok=True)
    with onnx_export_mode():
        torch.onnx.export(
            model,
            dummy_imgs.reshape(B, N, C, H, W),
            "./outputs/flow3r.onnx",
            input_names=["images"], output_names=["output"],
            opset_version=18,
            dynamo=True,
            dynamic_shapes={
                "images": {
                    0: batch,
                    1: num_frames,
                    3: height,
                    4: width
                },
            },
            verify=True,
        )
    logging.info("Exported ONNX model.")
