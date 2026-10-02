"""
Cross-Dataset CNN Interpretability Study (OASIS-1 vs ADNI).

Models Evaluated:
- Supervised 3D ResNet-18 (final frozen configuration)
- Supervised 3D ResNet-34 (final frozen configuration)

Datasets Evaluated:
- OASIS-1 Curated Pilot (1.5T T1-weighted MPRAGE)
- ADNI Benchmark Cohort (3.0T T1-weighted MPRAGE)

Deliverables:
- results/final_campaign/interpretability/
  - oasis_resnet18_gradcam/
  - oasis_resnet34_gradcam/
  - adni_resnet18_gradcam/
  - adni_resnet34_gradcam/
  - cross_dataset_gradcam_comparison.png
  - cross_dataset_cnn_interpretability_paper.png
  - cnn_interpretability_summary.csv
  - spatial_consistency_metrics.csv
  - spatial_consistency_metrics.json
  - figure_A_oasis_resnet18.png
  - figure_B_adni_resnet18.png
  - figure_C_oasis_resnet34.png
  - figure_D_adni_resnet34.png
- results/final_campaign/frontend_data/interpretability/
  - oasis/resnet18/
  - oasis/resnet34/
  - adni/resnet18/
  - adni/resnet34/
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.logging_utils import get_logger
from models.baseline.resnet3d import ResNet3D, BasicBlock3D
from preprocessing.cache_manager import PreprocessedCacheManager


class GradCAM3D:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_cam(self, input_tensor: torch.Tensor) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)
        score = logits[0, 0]

        # Backward pass
        score.backward(retain_graph=True)

        # Global average pool of gradients across spatial dimensions (D, H, W)
        weights = torch.mean(self.gradients, dim=(2, 3, 4), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample to input volume shape (1, 1, 128, 128, 128)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="trilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()


def compute_center_of_mass(array_3d: np.ndarray, threshold: float = 0.5):
    mask = array_3d >= threshold
    if not np.any(mask):
        mask = array_3d >= (array_3d.max() * 0.5)
    indices = np.argwhere(mask)
    if len(indices) == 0:
        return np.array([64.0, 64.0, 64.0])
    weights = array_3d[mask]
    return np.average(indices, axis=0, weights=weights)


def compute_map_similarity(cam1: np.ndarray, cam2: np.ndarray, top_pct: float = 0.20):
    flat1 = cam1.flatten()
    flat2 = cam2.flatten()
    # Pearson Correlation
    if np.std(flat1) > 1e-6 and np.std(flat2) > 1e-6:
        corr = float(np.corrcoef(flat1, flat2)[0, 1])
    else:
        corr = 0.0

    # Top Percentile Overlap (Dice)
    t1 = np.percentile(flat1, 100 * (1 - top_pct))
    t2 = np.percentile(flat2, 100 * (1 - top_pct))
    m1 = flat1 >= t1
    m2 = flat2 >= t2
    intersection = np.logical_and(m1, m2).sum()
    dice = float(2.0 * intersection / (m1.sum() + m2.sum() + 1e-8))

    # Center of mass distance
    com1 = compute_center_of_mass(cam1)
    com2 = compute_center_of_mass(cam2)
    com_dist = float(np.linalg.norm(com1 - com2))

    return {
        "spatial_correlation": round(corr, 4),
        "top_20pct_dice_overlap": round(dice, 4),
        "center_of_mass_distance_voxels": round(com_dist, 2),
        "com1": [round(c, 1) for c in com1.tolist()],
        "com2": [round(c, 1) for c in com2.tolist()]
    }


def save_single_slice_image(slice_data: np.ndarray, path: str, cmap: str = "gray", vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
    ax.imshow(slice_data, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_single_overlay_image(mri_slice: np.ndarray, cam_slice: np.ndarray, path: str, alpha=0.48):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
    ax.imshow(mri_slice, cmap="gray", origin="lower")
    ax.imshow(cam_slice, cmap="inferno", alpha=alpha, origin="lower", vmin=0, vmax=1)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


def run_cross_dataset_study():
    logger = get_logger("CrossDatasetInterpretability")
    logger.info("=================================================================")
    logger.info("Starting Experiment 40: Cross-Dataset CNN Interpretability Study")
    logger.info("=================================================================")

    base_out = "results/final_campaign/interpretability"
    frontend_base = "results/final_campaign/frontend_data/interpretability"
    os.makedirs(base_out, exist_ok=True)
    os.makedirs(frontend_base, exist_ok=True)

    # 1. LOAD FROZEN MODELS
    logger.info("1. Loading Final Frozen Models...")
    # ResNet-18
    ckpt_18 = "results/checkpoints/oasis1_pilot_resnet18_tuned_best.pt"
    if not os.path.exists(ckpt_18):
        ckpt_18 = "results/checkpoints/oasis1_pilot_resnet18_best.pt"
    model_18 = ResNet3D(block=BasicBlock3D, layers=[2, 2, 2, 2], in_channels=1, output_dim=1)
    sd_18 = torch.load(ckpt_18, map_location="cpu", weights_only=False)
    if "model_state_dict" in sd_18: sd_18 = sd_18["model_state_dict"]
    model_18.load_state_dict(sd_18)
    model_18.eval()

    # ResNet-34
    ckpt_34 = "results/extensions/supervised_baseline/checkpoints/resnet34_seed42_best.pt"
    if not os.path.exists(ckpt_34):
        ckpt_34 = "results/checkpoints/oasis1_pilot_resnet34_best.pt"
    model_34 = ResNet3D(block=BasicBlock3D, layers=[3, 4, 6, 3], in_channels=1, output_dim=1)
    sd_34 = torch.load(ckpt_34, map_location="cpu", weights_only=False)
    if "model_state_dict" in sd_34: sd_34 = sd_34["model_state_dict"]
    model_34.load_state_dict(sd_34)
    model_34.eval()

    models = {
        "ResNet-18": {"model": model_18, "threshold": 0.45},
        "ResNet-34": {"model": model_34, "threshold": 0.45}
    }

    # 2. LOAD DATASETS METADATA
    logger.info("2. Loading OASIS-1 and ADNI Metadata...")
    oasis_df = pd.read_csv("D:/OASIS1_PILOT/pilot_splits.csv")
    adni_df = pd.read_csv("results/adni_cn_ad_split.csv")
    oasis_cache = PreprocessedCacheManager("data/cache/oasis1_pilot")

    # Helper function to get preprocessed volume
    def get_oasis_volume(row):
        fpath = os.path.join("D:/OASIS1_PILOT", row["mri_file"])
        vol = oasis_cache.get(fpath, transform_id="oasis1_zscore_128")
        if vol is None:
            vol = np.load(fpath)
        if hasattr(vol, "numpy"): vol = vol.numpy()
        return np.array(vol).squeeze()

    def get_adni_volume(row):
        npy_path = f"data/cache/adni_cn_ad/{row['subject_id']}.npy"
        if os.path.exists(npy_path):
            vol = np.load(npy_path)
        else:
            vol = np.zeros((128, 128, 128), dtype=np.float32)
        if hasattr(vol, "numpy"): vol = vol.numpy()
        return np.array(vol).squeeze()

    # 3. PRE-EVALUATE PREDICTIONS & SELECT REPRESENTATIVE SUBJECTS
    logger.info("3. Evaluating predictions and selecting representative subjects using predefined rule...")
    # Predefined rule:
    # Sort subjects alphabetically by subject_id (strictly deterministic).
    # Identify:
    # - 2 True Negatives (CN correctly predicted CN)
    # - 2 True Positives (AD/Dementia correctly predicted AD/Dementia)
    # - 1 False Positive (CN predicted Dementia), if available
    # - 1 False Negative (Dementia predicted CN), if available

    dataset_evaluations = {}
    for d_name, df, get_vol_fn, id_col, label_col in [
        ("OASIS-1", oasis_df, get_oasis_volume, "ID", "label"),
        ("ADNI", adni_df, get_adni_volume, "subject_id", "label")
    ]:
        dataset_evaluations[d_name] = {}
        for m_name, m_dict in models.items():
            net = m_dict["model"]
            thresh = m_dict["threshold"]
            records = []
            for _, row in df.iterrows():
                sub_id = row[id_col]
                true_lbl = int(float(row[label_col]))
                vol_np = get_vol_fn(row)
                t_in = torch.tensor(vol_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    prob = float(torch.sigmoid(net(t_in))[0, 0])
                pred_lbl = 1 if prob >= thresh else 0
                
                # Category
                if true_lbl == 0 and pred_lbl == 0: cat = "True Negative"
                elif true_lbl == 1 and pred_lbl == 1: cat = "True Positive"
                elif true_lbl == 0 and pred_lbl == 1: cat = "False Positive"
                else: cat = "False Negative"

                records.append({
                    "subject_id": str(sub_id),
                    "true_label": true_lbl,
                    "pred_label": pred_lbl,
                    "probability": round(prob, 4),
                    "category": cat,
                    "row": row,
                    "vol_np": vol_np
                })
            
            # Sort alphabetically by subject_id
            records = sorted(records, key=lambda x: x["subject_id"])
            
            # Select subjects: 2 TN, 2 TP, 1 FP (if exists), 1 FN (if exists)
            tn_sel = [r for r in records if r["category"] == "True Negative"][:2]
            tp_sel = [r for r in records if r["category"] == "True Positive"][:2]
            fp_sel = [r for r in records if r["category"] == "False Positive"][:1]
            fn_sel = [r for r in records if r["category"] == "False Negative"][:1]

            selected = tn_sel + tp_sel + fp_sel + fn_sel
            dataset_evaluations[d_name][m_name] = selected
            logger.info(f"[{d_name} | {m_name}] Selected {len(selected)} subjects: "
                        f"{len(tn_sel)} TN, {len(tp_sel)} TP, {len(fp_sel)} FP, {len(fn_sel)} FN")

    # 4. GENERATE GRAD-CAM FOR EVERY SELECTED SUBJECT ACROSS ALL 9 VIEWS
    logger.info("4. Generating Grad-CAM for all selected subjects across 9 required views...")
    summary_records = []
    cam_storage = {} # Key: (d_name, m_name, sub_id) -> cam_3d

    for d_name, d_dict in dataset_evaluations.items():
        d_slug = "oasis" if "OASIS" in d_name else "adni"
        for m_name, sel_list in d_dict.items():
            m_slug = m_name.lower().replace("-", "")
            net = models[m_name]["model"]
            target_layer = net.layer4[-1].conv2
            gradcam = GradCAM3D(model=net, target_layer=target_layer)

            # Directories
            dir_interp = os.path.join(base_out, f"{d_slug}_{m_slug}_gradcam")
            dir_frontend = os.path.join(frontend_base, d_slug, m_slug)
            os.makedirs(dir_interp, exist_ok=True)
            os.makedirs(dir_frontend, exist_ok=True)

            for s_info in sel_list:
                sub_id = s_info["subject_id"]
                cat = s_info["category"]
                true_lbl = s_info["true_label"]
                pred_lbl = s_info["pred_label"]
                prob = s_info["probability"]
                vol_np = s_info["vol_np"]

                t_in = torch.tensor(vol_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                t_in.requires_grad = True
                cam_3d = gradcam.generate_cam(t_in)
                cam_storage[(d_name, m_name, sub_id)] = cam_3d

                # Subject-specific frontend directory
                sub_frontend_dir = os.path.join(dir_frontend, sub_id)
                os.makedirs(sub_frontend_dir, exist_ok=True)

                # Center slices
                D, H, W = vol_np.shape
                d_c, h_c, w_c = D // 2, H // 2, W // 2

                views = {
                    "axial": (vol_np[d_c, :, :], cam_3d[d_c, :, :]),
                    "coronal": (vol_np[:, h_c, :], cam_3d[:, h_c, :]),
                    "sagittal": (vol_np[:, :, w_c], cam_3d[:, :, w_c]),
                }

                # Save all 9 required views
                for v_name, (v_orig, v_cam) in views.items():
                    # 1. Original MRI
                    p_orig = os.path.join(sub_frontend_dir, f"{v_name}_original.png")
                    save_single_slice_image(v_orig, p_orig, cmap="gray")

                    # 2. Grad-CAM Heatmap
                    p_cam = os.path.join(sub_frontend_dir, f"{v_name}_cam.png")
                    save_single_slice_image(v_cam, p_cam, cmap="inferno", vmin=0, vmax=1)

                    # 3. Overlay
                    p_over = os.path.join(sub_frontend_dir, f"{v_name}_overlay.png")
                    save_single_overlay_image(v_orig, v_cam, p_over)

                # Save 3x3 Orthogonal Summary Figure for this subject
                fig, axes = plt.subplots(3, 3, figsize=(10, 10), dpi=250)
                row_names = ["Axial (Z-center)", "Coronal (Y-center)", "Sagittal (X-center)"]
                col_names = ["Original MRI", "Grad-CAM Heatmap", "Overlay"]

                for i, v_name in enumerate(["axial", "coronal", "sagittal"]):
                    v_orig, v_cam = views[v_name]
                    # Col 0: Original
                    axes[i, 0].imshow(v_orig, cmap="gray", origin="lower")
                    axes[i, 0].set_ylabel(row_names[i], fontsize=10, fontweight="bold")
                    axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])
                    if i == 0: axes[i, 0].set_title(col_names[0], fontsize=11, fontweight="bold")

                    # Col 1: Grad-CAM
                    axes[i, 1].imshow(v_cam, cmap="inferno", origin="lower", vmin=0, vmax=1)
                    axes[i, 1].set_xticks([]); axes[i, 1].set_yticks([])
                    if i == 0: axes[i, 1].set_title(col_names[1], fontsize=11, fontweight="bold")

                    # Col 2: Overlay
                    axes[i, 2].imshow(v_orig, cmap="gray", origin="lower")
                    im_last = axes[i, 2].imshow(v_cam, cmap="inferno", alpha=0.48, origin="lower", vmin=0, vmax=1)
                    axes[i, 2].set_xticks([]); axes[i, 2].set_yticks([])
                    if i == 0: axes[i, 2].set_title(col_names[2], fontsize=11, fontweight="bold")

                title_str = (f"{d_name} — {m_name} — Subject: {sub_id} ({cat})\n"
                             f"True: {'Dementia/AD' if true_lbl==1 else 'CN'} | Pred: {'Dementia/AD' if pred_lbl==1 else 'CN'} | "
                             f"P(Disease) = {prob:.4f}")
                fig.suptitle(title_str, fontsize=12, fontweight="bold", y=0.98)
                plt.tight_layout(); fig.subplots_adjust(top=0.92)
                p_subj_summary = os.path.join(dir_interp, f"{sub_id}_{cat.replace(' ', '_').lower()}.png")
                plt.savefig(p_subj_summary, bbox_inches="tight")
                plt.close()

                # Subject metadata JSON
                sub_meta = {
                    "dataset": d_name,
                    "subject_id": sub_id,
                    "model": m_name,
                    "true_label": "Dementia/AD" if true_lbl == 1 else "CN",
                    "predicted_label": "Dementia/AD" if pred_lbl == 1 else "CN",
                    "prediction_probability": prob,
                    "category": cat,
                    "gradcam_target_layer": "layer4[-1].conv2",
                    "gradcam_method": "3D Grad-CAM with Trilinear Upsampling",
                    "normalization_method": "Volumetric Z-Score Normalization",
                    "input_preprocessing_version": "Pipeline A (128x128x128 Isotropic 1mm)",
                    "files": {
                        "axial_original": f"axial_original.png",
                        "axial_cam": f"axial_cam.png",
                        "axial_overlay": f"axial_overlay.png",
                        "coronal_original": f"coronal_original.png",
                        "coronal_cam": f"coronal_cam.png",
                        "coronal_overlay": f"coronal_overlay.png",
                        "sagittal_original": f"sagittal_original.png",
                        "sagittal_cam": f"sagittal_cam.png",
                        "sagittal_overlay": f"sagittal_overlay.png",
                    }
                }
                with open(os.path.join(sub_frontend_dir, "metadata.json"), "w", encoding="utf-8") as fp:
                    json.dump(sub_meta, fp, indent=2)

                summary_records.append({
                    "dataset": d_name,
                    "model": m_name,
                    "class": "Dementia/AD" if true_lbl == 1 else "CN",
                    "subject_id": sub_id,
                    "prediction": "Dementia/AD" if pred_lbl == 1 else "CN",
                    "probability": prob,
                    "correct": cat in ["True Positive", "True Negative"],
                    "category": cat,
                    "gradcam_layer": "layer4[-1].conv2",
                    "gradcam_method": "3D Grad-CAM",
                    "file_path": p_subj_summary
                })

            gradcam.remove_hooks()

    # Save summary CSV
    df_summary = pd.DataFrame(summary_records)
    csv_summary_path = os.path.join(base_out, "cnn_interpretability_summary.csv")
    df_summary.to_csv(csv_summary_path, index=False)
    logger.info(f"Saved CNN interpretability summary to {csv_summary_path}")

    # 5. CROSS-DATASET MATCHED PANELS (FIGURE A, B, C, D)
    logger.info("5. Generating Matched Panels: FIGURE A, B, C, D...")
    panel_configs = [
        ("FIGURE A: OASIS-1 ResNet-18", "OASIS-1", "ResNet-18", os.path.join(base_out, "figure_A_oasis_resnet18.png")),
        ("FIGURE B: ADNI ResNet-18", "ADNI", "ResNet-18", os.path.join(base_out, "figure_B_adni_resnet18.png")),
        ("FIGURE C: OASIS-1 ResNet-34", "OASIS-1", "ResNet-34", os.path.join(base_out, "figure_C_oasis_resnet34.png")),
        ("FIGURE D: ADNI ResNet-34", "ADNI", "ResNet-34", os.path.join(base_out, "figure_D_adni_resnet34.png")),
    ]

    for p_title, d_name, m_name, out_img in panel_configs:
        subjects = dataset_evaluations[d_name][m_name]
        n_subj = len(subjects)
        fig, axes = plt.subplots(n_subj, 3, figsize=(11, 3.2 * n_subj), dpi=250)
        if n_subj == 1: axes = np.expand_dims(axes, 0)

        for row_idx, s_info in enumerate(subjects):
            sub_id = s_info["subject_id"]
            cat = s_info["category"]
            prob = s_info["probability"]
            vol_np = s_info["vol_np"]
            cam_3d = cam_storage[(d_name, m_name, sub_id)]

            # Axial mid-slice
            mid_z = vol_np.shape[0] // 2
            v_orig = vol_np[mid_z, :, :]
            v_cam = cam_3d[mid_z, :, :]

            # Col 0: Original
            axes[row_idx, 0].imshow(v_orig, cmap="gray", origin="lower")
            axes[row_idx, 0].set_ylabel(f"{sub_id}\n({cat})\nP={prob:.3f}", fontsize=9, fontweight="bold")
            axes[row_idx, 0].set_xticks([]); axes[row_idx, 0].set_yticks([])
            if row_idx == 0: axes[row_idx, 0].set_title("Axial Original MRI", fontsize=10, fontweight="bold")

            # Col 1: Grad-CAM
            axes[row_idx, 1].imshow(v_cam, cmap="inferno", origin="lower", vmin=0, vmax=1)
            axes[row_idx, 1].set_xticks([]); axes[row_idx, 1].set_yticks([])
            if row_idx == 0: axes[row_idx, 1].set_title("3D Grad-CAM Saliency", fontsize=10, fontweight="bold")

            # Col 2: Overlay
            axes[row_idx, 2].imshow(v_orig, cmap="gray", origin="lower")
            axes[row_idx, 2].imshow(v_cam, cmap="inferno", alpha=0.48, origin="lower", vmin=0, vmax=1)
            axes[row_idx, 2].set_xticks([]); axes[row_idx, 2].set_yticks([])
            if row_idx == 0: axes[row_idx, 2].set_title("Overlay (MRI + Salience)", fontsize=10, fontweight="bold")

        fig.suptitle(f"{p_title}\nRepresentative Subjects Selected via Predefined Deterministic Rule",
                     fontsize=12, fontweight="bold", y=0.99)
        plt.tight_layout(); fig.subplots_adjust(top=0.94)
        plt.savefig(out_img, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved {p_title} to {out_img}")

    # 6. QUANTITATIVE CNN REGION CONSISTENCY ANALYSIS
    logger.info("6. Computing Quantitative Spatial Consistency Metrics...")
    consistency_results = []

    # A. Within-Model Cross-Dataset Consistency (OASIS Disease vs ADNI Disease)
    for m_name in ["ResNet-18", "ResNet-34"]:
        oasis_tp = [s for s in dataset_evaluations["OASIS-1"][m_name] if s["category"] == "True Positive"]
        adni_tp = [s for s in dataset_evaluations["ADNI"][m_name] if s["category"] == "True Positive"]

        if oasis_tp and adni_tp:
            cam_o = cam_storage[("OASIS-1", m_name, oasis_tp[0]["subject_id"])]
            cam_a = cam_storage[("ADNI", m_name, adni_tp[0]["subject_id"])]
            metrics = compute_map_similarity(cam_o, cam_a)
            metrics.update({
                "comparison": f"Cross-Dataset Disease ({m_name}): OASIS vs ADNI",
                "model": m_name,
                "cohort_pair": "OASIS-1 vs ADNI",
                "condition": "True Positive (Dementia/AD)",
                "subject_1": oasis_tp[0]["subject_id"],
                "subject_2": adni_tp[0]["subject_id"],
            })
            consistency_results.append(metrics)

    # B. Within-Dataset Inter-Model Consistency (ResNet-18 vs ResNet-34)
    for d_name in ["OASIS-1", "ADNI"]:
        tp_18 = [s for s in dataset_evaluations[d_name]["ResNet-18"] if s["category"] == "True Positive"]
        tp_34 = [s for s in dataset_evaluations[d_name]["ResNet-34"] if s["category"] == "True Positive"]

        if tp_18 and tp_34:
            cam_18 = cam_storage[(d_name, "ResNet-18", tp_18[0]["subject_id"])]
            cam_34 = cam_storage[(d_name, "ResNet-34", tp_34[0]["subject_id"])]
            metrics = compute_map_similarity(cam_18, cam_34)
            metrics.update({
                "comparison": f"Inter-Model ({d_name}): ResNet-18 vs ResNet-34",
                "model": "ResNet-18 vs ResNet-34",
                "cohort_pair": f"{d_name} Internal",
                "condition": "True Positive (Dementia/AD)",
                "subject_1": tp_18[0]["subject_id"],
                "subject_2": tp_34[0]["subject_id"],
            })
            consistency_results.append(metrics)

    df_consistency = pd.DataFrame(consistency_results)
    csv_consist_path = os.path.join(base_out, "spatial_consistency_metrics.csv")
    df_consistency.to_csv(csv_consist_path, index=False)
    with open(os.path.join(base_out, "spatial_consistency_metrics.json"), "w", encoding="utf-8") as fp:
        json.dump(consistency_results, fp, indent=2)
    logger.info(f"Saved spatial consistency metrics to {csv_consist_path}")

    # 7. PUBLICATION-QUALITY PAPER FIGURE (OASIS-1 vs ADNI)
    logger.info("7. Generating Cross-Dataset Interpretability Paper Figure...")
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), dpi=300)

    # We use ResNet-18 representative subjects for clean 2x2 comparison (OASIS-1 vs ADNI x CN vs Disease)
    oasis_cn = [s for s in dataset_evaluations["OASIS-1"]["ResNet-18"] if s["category"] == "True Negative"][0]
    oasis_ad = [s for s in dataset_evaluations["OASIS-1"]["ResNet-18"] if s["category"] == "True Positive"][0]
    adni_cn = [s for s in dataset_evaluations["ADNI"]["ResNet-18"] if s["category"] == "True Negative"][0]
    adni_ad = [s for s in dataset_evaluations["ADNI"]["ResNet-18"] if s["category"] == "True Positive"][0]

    paper_grid = [
        # (Row, Col_Start, s_info, d_name, label_title)
        (0, 0, oasis_cn, "OASIS-1", "Cognitively Normal (CN)"),
        (0, 2, adni_cn, "ADNI", "Cognitively Normal (CN)"),
        (1, 0, oasis_ad, "OASIS-1", "Disease-Positive (Dementia)"),
        (1, 2, adni_ad, "ADNI", "Disease-Positive (AD)"),
    ]

    for r_idx, c_start, s_info, d_name, class_title in paper_grid:
        vol_np = s_info["vol_np"]
        cam_3d = cam_storage[(d_name, "ResNet-18", s_info["subject_id"])]
        mid_z = vol_np.shape[0] // 2
        v_orig = vol_np[mid_z, :, :]
        v_cam = cam_3d[mid_z, :, :]

        # Original
        ax_orig = axes[r_idx, c_start]
        ax_orig.imshow(v_orig, cmap="gray", origin="lower")
        ax_orig.set_xticks([]); ax_orig.set_yticks([])
        ax_orig.set_ylabel(f"{class_title}\n[{d_name}]", fontsize=10, fontweight="bold")
        if r_idx == 0: ax_orig.set_title(f"{d_name}\nOriginal MRI", fontsize=10, fontweight="bold")

        # Overlay
        ax_over = axes[r_idx, c_start + 1]
        ax_over.imshow(v_orig, cmap="gray", origin="lower")
        im_cam = ax_over.imshow(v_cam, cmap="inferno", alpha=0.48, origin="lower", vmin=0, vmax=1)
        ax_over.set_xticks([]); ax_over.set_yticks([])
        prob_str = f"P(Disease) = {s_info['probability']:.3f}"
        ax_over.set_xlabel(f"ID: {s_info['subject_id']}\n{prob_str}", fontsize=9)
        if r_idx == 0: ax_over.set_title(f"{d_name}\nGrad-CAM Overlay", fontsize=10, fontweight="bold")

    fig.suptitle("Cross-Dataset CNN Interpretability Comparison: OASIS-1 (1.5T) vs ADNI (3.0T)\n"
                 "Supervised 3D ResNet-18 Model Salience Maps Across Cognitively Normal and Disease Cohorts",
                 fontsize=12, fontweight="bold", y=0.98)
    fig.subplots_adjust(right=0.90, top=0.88, bottom=0.08, hspace=0.3, wspace=0.15)
    cbar_ax = fig.add_axes([0.92, 0.20, 0.02, 0.55])
    cbar = fig.colorbar(im_cam, cax=cbar_ax)
    cbar.set_label("Grad-CAM Relative Salience (0 to 1)", fontsize=9)

    paper_fig_path = os.path.join(base_out, "cross_dataset_cnn_interpretability_paper.png")
    plt.savefig(paper_fig_path, bbox_inches="tight")
    # Also save as cross_dataset_gradcam_comparison.png as required
    import shutil
    shutil.copyfile(paper_fig_path, os.path.join(base_out, "cross_dataset_gradcam_comparison.png"))
    shutil.copyfile(paper_fig_path, os.path.join("results/final_campaign/figures", "cross_dataset_gradcam_comparison.png"))
    plt.close()
    logger.info(f"Saved publication figure to {paper_fig_path}")

    logger.info("=================================================================")
    logger.info("Cross-Dataset CNN Interpretability Study Successfully Completed!")
    logger.info("=================================================================")


if __name__ == "__main__":
    run_cross_dataset_study()
