import logging
import os.path

import onnx
import onnxsim
import torch.onnx

from flow3r.models.flow3r import Flow3r


class Wrapper(torch.nn.Module):
    """
    Wrapper class for Flow3r model.
    """

    def __init__(self, model: Flow3r):
        super().__init__()
        self.model = model

    def forward(self, imgs: torch.Tensor, pair_indices: torch.Tensor | None = None):
        return self.model.forward(imgs, pair_indices=pair_indices)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logging.info("Loading model...")

    patch_size = 14
    max_img_w, max_img_h = 672, 672

    assert max_img_w % patch_size == 0
    assert max_img_h % patch_size == 0
    max_pos = max(max_img_w, max_img_h) // patch_size

    device = torch.device("cpu")
    model_kwargs = dict(for_onnx=True, max_pos=max_pos)
    flow3r = Flow3r.from_pretrained("Clara211111/flow3r", model_kwargs=model_kwargs)
    flow3r = flow3r.to(device).eval()

    # B=1, N=2 (num frames), C=3, H, W.
    # H and W must be multiples of 14 (DINOv2 patch size).
    # Use non-max dummy values so the exporter sees variation from the bounds.
    B, N, C, H, W = 2, 2, 3, 448, 448
    dummy_imgs = torch.randn((B, N, C, H, W), dtype=torch.float32, device=device)

    model = Wrapper(flow3r)

    b_sym = torch.export.Dim("B", min=1, max=16)
    n_sym = torch.export.Dim("N", min=1, max=16)
    # Express H and W as multiples of patch_size so the exporter knows the
    # divisibility constraint and won't specialise on the dummy value.
    patch_h_sym = torch.export.Dim("patch_H", min=1, max=max_img_h // patch_size)
    patch_w_sym = torch.export.Dim("patch_W", min=1, max=max_img_w // patch_size)
    h_sym = patch_size * patch_h_sym
    w_sym = patch_size * patch_w_sym

    output_path = "./outputs/flow3r.onnx"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Note that pair_indices stays None so the flow_head will not be exported.
    torch.onnx.export(
        model,
        (dummy_imgs,),
        output_path,
        input_names=["imgs"],
        output_names=["output"],
        opset_version=18,
        dynamo=True,
        dynamic_shapes={
            "imgs": {0: b_sym, 1: n_sym, 3: h_sym, 4: w_sym},
        },
        # verify=True,
    )
    logging.info(f"Exported ONNX model to {os.path.abspath(output_path)}.")

    # onnxsim requires the full proto in memory (SerializeToString) which fails
    # for models >2 GB. If the export produced an external-data sidecar file,
    # the model is too large to simplify this way — skip gracefully.
    external_data_path = output_path + ".data"
    if os.path.exists(external_data_path):
        logging.warning(
            "Model uses external data (>2 GB); skipping onnxsim simplification."
        )
    else:
        model = onnx.load(output_path)
        model, success = onnxsim.simplify(
            model,
            overwrite_input_shapes={"imgs": [B, N, C, H, W]}
        )
        if success:
            onnx.save(model, output_path)
            logging.info(
                f"Successfully simplified the model and overwrote {os.path.abspath(output_path)}.")
        else:
            logging.warning("onnxsim reported failure — keeping the unsimplified model.")