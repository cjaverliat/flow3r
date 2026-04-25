import os
import cv2
import torch
import numpy as np
import gradio as gr
import sys
import shutil
from datetime import datetime
import glob
import gc
import time
import trimesh
import matplotlib
import onnxruntime as ort

from flow3r.utils.basic import load_images_as_tensor
from flow3r.utils.geometry import depth_edge

from scipy.spatial.transform import Rotation

ONNX_MODEL_PATH = os.environ.get("FLOW3R_ONNX_PATH", "./outputs/flow3r.onnx")
TRT_CACHE_PATH = os.environ.get("FLOW3R_TRT_CACHE", "./outputs/trt_cache")

# TensorRT optimization profile — (B=1 fixed, N dynamic, C=3, H dynamic, W dynamic)
_TRT_B = 1
_TRT_N_MIN, _TRT_N_OPT, _TRT_N_MAX = 1, 2, 8
_TRT_H_MIN, _TRT_H_OPT, _TRT_H_MAX = 1, 336, 672
_TRT_W_MIN, _TRT_W_OPT, _TRT_W_MAX = 1, 336, 672


def _shape_str(b, n, h, w):
    return f"imgs:{b}x{n}x3x{h}x{w}"


print("Initializing ONNX Runtime session...")

_available = ort.get_available_providers()

if "TensorrtExecutionProvider" in _available:
    os.makedirs(TRT_CACHE_PATH, exist_ok=True)
    _trt_options = {
        "trt_fp16_enable": True,
        "trt_profile_min_shapes": _shape_str(_TRT_B, _TRT_N_MIN, _TRT_H_MIN, _TRT_W_MIN),
        "trt_profile_opt_shapes": _shape_str(_TRT_B, _TRT_N_OPT, _TRT_H_OPT, _TRT_W_OPT),
        "trt_profile_max_shapes": _shape_str(_TRT_B, _TRT_N_MAX, _TRT_H_MAX, _TRT_W_MAX),
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": TRT_CACHE_PATH,
    }
    providers = [
        ("TensorrtExecutionProvider", _trt_options),
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    print("Using TensorRT execution provider (FP16 enabled)")
elif "CUDAExecutionProvider" in _available:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    print("TensorRT not available, using CUDA execution provider")
else:
    providers = ["CPUExecutionProvider"]
    print("Using CPU execution provider")

session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)

_output_names = [o.name for o in session.get_outputs()]
print(f"ONNX model loaded. Outputs: {_output_names}")

# Determine fixed spatial dimensions from the model input shape (indices 3=H, 4=W).
# If the exporter baked in fixed sizes, we must resize inputs to match.
_input_shape = session.get_inputs()[0].shape  # [B, N, C, H, W]
_expected_h = _input_shape[3] if isinstance(_input_shape[3], int) else None
_expected_w = _input_shape[4] if isinstance(_input_shape[4], int) else None
print(f"ONNX expected input spatial size: H={_expected_h}, W={_expected_w} (None = dynamic)")

# Map output names to expected keys. The dynamo exporter preserves dict keys;
# fall back to positional mapping if they differ.
_EXPECTED_KEYS = ["points", "local_points", "conf", "camera_poses", "flow"]


def _map_outputs(raw_outputs: list) -> dict:
    if all(k in _output_names for k in _EXPECTED_KEYS):
        return {k: raw_outputs[_output_names.index(k)] for k in _EXPECTED_KEYS}
    return dict(zip(_EXPECTED_KEYS, raw_outputs))


# -------------------------------------------------------------------------
# Utils (unchanged from original)
# -------------------------------------------------------------------------
def predictions_to_glb(predictions, conf_thres=50.0, filter_by_frames="all", show_cam=True) -> trimesh.Scene:
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    if conf_thres is None:
        conf_thres = 10

    print("Building GLB scene")
    selected_frame_idx = None
    if filter_by_frames != "all" and filter_by_frames != "All":
        try:
            selected_frame_idx = int(filter_by_frames.split(":")[0])
        except (ValueError, IndexError):
            pass

    pred_world_points = predictions["points"]
    pred_world_points_conf = predictions.get("conf", np.ones_like(pred_world_points[..., 0]))
    images = predictions["images"]
    camera_poses = predictions["camera_poses"]

    if selected_frame_idx is not None:
        pred_world_points = pred_world_points[selected_frame_idx][None]
        pred_world_points_conf = pred_world_points_conf[selected_frame_idx][None]
        images = images[selected_frame_idx][None]
        camera_poses = camera_poses[selected_frame_idx][None]

    vertices_3d = pred_world_points.reshape(-1, 3)
    if images.ndim == 4 and images.shape[1] == 3:  # NCHW
        colors_rgb = np.transpose(images, (0, 2, 3, 1))
    else:
        colors_rgb = images
    colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)

    conf = pred_world_points_conf.reshape(-1)
    if conf_thres == 0.0:
        conf_threshold = 0.0
    else:
        conf_threshold = conf_thres / 100

    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)
    vertices_3d = vertices_3d[conf_mask]
    colors_rgb = colors_rgb[conf_mask]

    if vertices_3d is None or np.asarray(vertices_3d).size == 0:
        vertices_3d = np.array([[1, 0, 0]])
        colors_rgb = np.array([[255, 255, 255]])
        scene_scale = 1
    else:
        lower_percentile = np.percentile(vertices_3d, 5, axis=0)
        upper_percentile = np.percentile(vertices_3d, 95, axis=0)
        scene_scale = np.linalg.norm(upper_percentile - lower_percentile)

    colormap = matplotlib.colormaps.get_cmap("gist_rainbow")
    scene_3d = trimesh.Scene()
    point_cloud_data = trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)
    scene_3d.add_geometry(point_cloud_data)

    num_cameras = len(camera_poses)
    if show_cam:
        for i in range(num_cameras):
            camera_to_world = camera_poses[i]
            rgba_color = colormap(i / num_cameras)
            current_color = tuple(int(255 * x) for x in rgba_color[:3])
            integrate_camera_into_scene(scene_3d, camera_to_world, current_color, 1.)

    align_rotation = np.eye(4)
    align_rotation[:3, :3] = Rotation.from_euler("y", 100, degrees=True).as_matrix()
    align_rotation[:3, :3] = align_rotation[:3, :3] @ Rotation.from_euler("x", 155, degrees=True).as_matrix()
    scene_3d.apply_transform(align_rotation)

    print("GLB Scene built")
    return scene_3d


def get_opengl_conversion_matrix() -> np.ndarray:
    matrix = np.identity(4)
    matrix[1, 1] = -1
    matrix[2, 2] = -1
    return matrix


def integrate_camera_into_scene(scene: trimesh.Scene, transform: np.ndarray, face_colors: tuple, scene_scale: float):
    cam_width = scene_scale * 0.05
    cam_height = scene_scale * 0.1
    rot_45_degree = np.eye(4)
    rot_45_degree[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    rot_45_degree[2, 3] = -cam_height
    opengl_transform = get_opengl_conversion_matrix()
    complete_transform = transform @ opengl_transform @ rot_45_degree
    camera_cone_shape = trimesh.creation.cone(cam_width, cam_height, sections=4)
    slight_rotation = np.eye(4)
    slight_rotation[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()
    vertices_combined = np.concatenate([
        camera_cone_shape.vertices,
        0.95 * camera_cone_shape.vertices,
        transform_points(slight_rotation, camera_cone_shape.vertices),
    ])
    vertices_transformed = transform_points(complete_transform, vertices_combined)
    mesh_faces = compute_camera_faces(camera_cone_shape)
    camera_mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=mesh_faces)
    camera_mesh.visual.face_colors[:, :3] = face_colors
    scene.add_geometry(camera_mesh)


def transform_points(transformation: np.ndarray, points: np.ndarray, dim: int = None) -> np.ndarray:
    points = np.asarray(points)
    initial_shape = points.shape[:-1]
    dim = dim or points.shape[-1]
    transformation = transformation.swapaxes(-1, -2)
    points = points @ transformation[..., :-1, :] + transformation[..., -1:, :]
    return points[..., :dim].reshape(*initial_shape, dim)


def compute_camera_faces(cone_shape: trimesh.Trimesh) -> np.ndarray:
    faces_list = []
    num_vertices_cone = len(cone_shape.vertices)
    for face in cone_shape.faces:
        if 0 in face:
            continue
        v1, v2, v3 = face
        v1_offset, v2_offset, v3_offset = face + num_vertices_cone
        v1_offset_2, v2_offset_2, v3_offset_2 = face + 2 * num_vertices_cone
        faces_list.extend([
            (v1, v2, v2_offset), (v1, v1_offset, v3), (v3_offset, v2, v3),
            (v1, v2, v2_offset_2), (v1, v1_offset_2, v3), (v3_offset_2, v2, v3),
        ])
    faces_list += [(v3, v2, v1) for v1, v2, v3 in faces_list]
    return np.array(faces_list)


# -------------------------------------------------------------------------
# 1) Core model inference (ONNX)
# -------------------------------------------------------------------------
def run_model(target_dir, sess) -> dict:
    print(f"Processing images from {target_dir}")

    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    imgs = load_images_as_tensor(os.path.join(target_dir, "images"), interval=1)  # (N, 3, H, W) float32 in [0,1]

    if _expected_h is not None and _expected_w is not None:
        imgs = torch.nn.functional.interpolate(
            imgs, size=(_expected_h, _expected_w), mode="bilinear", align_corners=False
        )

    # ONNX Runtime expects (B, N, C, H, W) float32 numpy array
    imgs_np = imgs.numpy()[np.newaxis].astype(np.float32)  # (1, N, 3, H, W)

    print("Running ONNX inference...")
    raw_outputs = sess.run(None, {"imgs": imgs_np})

    predictions = _map_outputs(raw_outputs)

    # Convert to torch for depth_edge, then back to numpy
    local_points_t = torch.from_numpy(predictions["local_points"])  # (1, N, H, W, 3)
    conf_t = torch.from_numpy(predictions["conf"])  # (1, N, H, W, 1)

    conf_t = torch.sigmoid(conf_t)
    edge = depth_edge(local_points_t[..., 2], rtol=0.03)
    conf_t[edge] = 0.0

    predictions["conf"] = conf_t.numpy()
    predictions["images"] = imgs.numpy()[np.newaxis].transpose(0, 1, 3, 4, 2)  # (1, N, H, W, 3)
    del predictions["local_points"]
    del predictions["flow"]

    # Remove batch dimension
    for key in list(predictions.keys()):
        predictions[key] = predictions[key].squeeze(0)

    return predictions


# -------------------------------------------------------------------------
# 2) Handle uploaded video/images --> produce target_dir + images
# -------------------------------------------------------------------------
def handle_uploads(input_video, input_images):
    start_time = time.time()
    gc.collect()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"input_images_{timestamp}"
    target_dir_images = os.path.join(target_dir, "images")

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    os.makedirs(target_dir_images)

    image_paths = []

    if input_images is not None:
        for file_data in input_images:
            if isinstance(file_data, dict) and "name" in file_data:
                file_path = file_data["name"]
            else:
                file_path = file_data
            dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)

    if input_video is not None:
        if isinstance(input_video, dict) and "name" in input_video:
            video_path = input_video["name"]
        else:
            video_path = input_video

        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 1)

        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            count += 1
            if count % frame_interval == 0:
                image_path = os.path.join(target_dir_images, f"{video_frame_num:06}.png")
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                video_frame_num += 1

    image_paths = sorted(image_paths)
    end_time = time.time()
    print(f"Files copied to {target_dir_images}; took {end_time - start_time:.3f} seconds")
    return target_dir, image_paths


# -------------------------------------------------------------------------
# 3) Update gallery on upload
# -------------------------------------------------------------------------
def update_gallery_on_upload(input_video, input_images):
    if not input_video and not input_images:
        return None, None, None, None
    target_dir, image_paths = handle_uploads(input_video, input_images)
    return None, target_dir, image_paths, "Upload complete. Click 'Reconstruct' to begin 3D processing."


# -------------------------------------------------------------------------
# 4) Reconstruction
# -------------------------------------------------------------------------
def gradio_demo(target_dir, conf_thres=3.0, frame_filter="All", show_cam=True):
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, "No valid target directory found. Please upload first.", None, None

    start_time = time.time()
    gc.collect()

    target_dir_images = os.path.join(target_dir, "images")
    all_files = sorted(os.listdir(target_dir_images)) if os.path.isdir(target_dir_images) else []
    all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]
    frame_filter_choices = ["All"] + all_files

    print("Running run_model...")
    predictions = run_model(target_dir, session)

    prediction_save_path = os.path.join(target_dir, "predictions.npz")
    np.savez(prediction_save_path, **predictions)

    if frame_filter is None:
        frame_filter = "All"

    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_cam{show_cam}.glb",
    )

    glbscene = predictions_to_glb(predictions, conf_thres=conf_thres, filter_by_frames=frame_filter, show_cam=show_cam)
    glbscene.export(file_obj=glbfile)

    del predictions
    gc.collect()

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds (including IO)")
    log_msg = f"Reconstruction Success ({len(all_files)} frames). Waiting for visualization."

    return glbfile, log_msg, gr.Dropdown(choices=frame_filter_choices, value=frame_filter, interactive=True)


# -------------------------------------------------------------------------
# 5) Helper functions
# -------------------------------------------------------------------------
def clear_fields():
    return None


def update_log():
    return "Loading and Reconstructing..."


def update_visualization(target_dir, conf_thres, frame_filter, show_cam, is_example):
    if is_example == "True":
        return None, "No reconstruction available. Please click the Reconstruct button first."

    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        return None, "No reconstruction available. Please click the Reconstruct button first."

    predictions_path = os.path.join(target_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        return None, f"No reconstruction available at {predictions_path}. Please run 'Reconstruct' first."

    key_list = ["images", "points", "conf", "camera_poses"]
    loaded = np.load(predictions_path)
    predictions = {key: np.array(loaded[key]) for key in key_list}

    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_cam{show_cam}.glb",
    )

    if not os.path.exists(glbfile):
        glbscene = predictions_to_glb(predictions, conf_thres=conf_thres, filter_by_frames=frame_filter,
                                      show_cam=show_cam)
        glbscene.export(file_obj=glbfile)

    return glbfile, "Updating Visualization"


# -------------------------------------------------------------------------
# 6) Build Gradio UI
# -------------------------------------------------------------------------
theme = gr.themes.Ocean()
theme.set(
    checkbox_label_background_fill_selected="*button_primary_background_fill",
    checkbox_label_text_color_selected="*button_primary_text_color",
)

with gr.Blocks() as demo:
    is_example = gr.Textbox(label="is_example", visible=False, value="None")
    num_images = gr.Textbox(label="num_images", visible=False, value="None")

    gr.HTML(
        """
    <h1>Flow3r: Factored Flow Prediction for Visual Geometry Learning</h1>
    <p>
    <a href="https://github.com/Kidrauh/flow3r">GitHub Repository</a> |
    <a href="https://flow3r-project.github.io/">Project Page</a>
    </p>

    <div style="font-size: 16px; line-height: 1.5;">
    <p>Upload a video or a set of images to create a 3D reconstruction of a scene or object. Flow3r takes these images and generates a 3D point cloud, along with estimated camera poses.</p>
    </div>
    """
    )

    target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")

    with gr.Row():
        with gr.Column(scale=2):
            input_video = gr.Video(label="Upload Video", interactive=True)
            input_images = gr.File(file_count="multiple", label="Upload Images", interactive=True)

            image_gallery = gr.Gallery(
                label="Preview",
                columns=4,
                height="300px",
                object_fit="contain",
                preview=True,
            )

        with gr.Column(scale=4):
            with gr.Column():
                gr.Markdown("**3D Reconstruction (Point Cloud and Camera Poses)**")
                log_output = gr.Markdown(
                    "Please upload a video or images, then click Reconstruct.", elem_classes=["custom-log"]
                )
                reconstruction_output = gr.Model3D(height=520, zoom_speed=0.5, pan_speed=0.5)

            with gr.Row():
                submit_btn = gr.Button("Reconstruct", scale=1, variant="primary")
                clear_btn = gr.ClearButton(
                    [input_video, input_images, reconstruction_output, log_output, target_dir_output, image_gallery],
                    scale=1,
                )

            with gr.Row():
                conf_thres = gr.Slider(minimum=0, maximum=100, value=0, step=0.1, label="Confidence Threshold (%)")
                frame_filter = gr.Dropdown(choices=["All"], value="All", label="Show Points from Frame")
                with gr.Column():
                    show_cam = gr.Checkbox(label="Show Camera", value=True)

    submit_btn.click(fn=clear_fields, inputs=[], outputs=[reconstruction_output]).then(
        fn=update_log, inputs=[], outputs=[log_output]
    ).then(
        fn=gradio_demo,
        inputs=[target_dir_output, conf_thres, frame_filter, show_cam],
        outputs=[reconstruction_output, log_output, frame_filter],
    ).then(
        fn=lambda: "False", inputs=[], outputs=[is_example]
    )

    conf_thres.change(
        update_visualization,
        [target_dir_output, conf_thres, frame_filter, show_cam, is_example],
        [reconstruction_output, log_output],
    )
    frame_filter.change(
        update_visualization,
        [target_dir_output, conf_thres, frame_filter, show_cam, is_example],
        [reconstruction_output, log_output],
    )
    show_cam.change(
        update_visualization,
        [target_dir_output, conf_thres, frame_filter, show_cam, is_example],
        [reconstruction_output, log_output],
    )

    input_video.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images],
        outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
    )
    input_images.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images],
        outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
    )

    demo.queue(max_size=20).launch(show_error=True, share=True,
                                   theme=theme,
                                   css="""
    .custom-log * {
        font-style: italic;
        font-size: 22px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        font-weight: bold !important;
        color: transparent !important;
        text-align: center !important;
    }

    .example-log * {
        font-style: italic;
        font-size: 16px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent !important;
    }

    #my_radio .wrap {
        display: flex;
        flex-wrap: nowrap;
        justify-content: center;
        align-items: center;
    }

    #my_radio .wrap label {
        display: flex;
        width: 50%;
        justify-content: center;
        align-items: center;
        margin: 0;
        padding: 10px 0;
        box-sizing: border-box;
    }
    """)
