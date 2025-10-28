#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, sys, glob
import numpy as np
import torch

try:
    from src import config as sap_config
    from src.model import Encode2Points
    from src.utils import load_config, load_model_manual, scale2onet, export_mesh, export_pointcloud, is_url, load_url
    from src.dpsr import DPSR
except Exception as e:
    print("Could not import SAP modules. Run from the repo root.")
    raise

try:
    import open3d as o3d
except Exception:
    o3d = None

def read_ply_points(path):
    if o3d is not None:
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points).astype(np.float32)
        return pts
    # minimal ASCII fallback
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            header.append(line)
            if line.strip() == b'end_header':
                break
        data = np.loadtxt(f, dtype=np.float32)
    if data.shape[1] >= 3:
        return data[:, :3].astype(np.float32)
    raise ValueError(f"{path} has no XYZ columns")

def svd_planarity_ratio(pts):
    X = pts - pts.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = S**2
    return float(var[-1] / (var.sum() + 1e-12))

def normalize_safe(pts, mode="auto", target=0.48):
    """
    Map points to a centered cube of [-target, target]^3.
    - mode='auto'  : only normalize if any coord is outside [-0.55, 0.55]
    - mode='always': always normalize
    - mode='never' : never normalize
    Uses bbox center (robust than mean for partial shapes).
    """
    m = pts.min(0); M = pts.max(0)
    already_unit = (m >= -0.55).all() and (M <= 0.55).all()
    if mode == "never" or (mode == "auto" and already_unit):
        return pts.copy().astype(np.float32), np.zeros(3, np.float32), 1.0, "skip"
    c = 0.5 * (m + M)   # bbox center
    pts_c = pts - c
    r = np.abs(pts_c).max()  # infinity-norm radius
    if r < 1e-9: r = 1.0
    s = target / r
    return (pts_c * s).astype(np.float32), c.astype(np.float32), s, "apply"

def main():
    ap = argparse.ArgumentParser("SAP folder inference (safe margin normalization)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--pattern", default="*.ply")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--points", type=int, default=3000)
    ap.add_argument("--norm", choices=["auto","always","never"], default="auto")
    ap.add_argument("--target", type=float, default=0.48, help="Target half-extent inside unit cube; 0.48 leaves margin")
    ap.add_argument("--psr_res", type=int, default=None)
    ap.add_argument("--psr_sigma", type=float, default=None)
    ap.add_argument("--no_cuda", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, 'configs/default.yaml')
    if args.psr_res is not None:
        cfg['generation']['psr_resolution'] = int(args.psr_res)
    if args.psr_sigma is not None:
        cfg['generation']['psr_sigma'] = float(args.psr_sigma)

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")

    # model
    model = Encode2Points(cfg).to(device)
    if is_url(cfg['test']['model_file']):
        state_dict = load_url(cfg['test']['model_file'])
    else:
        out_dir_cfg = cfg['train']['out_dir']
        state_dict = torch.load(os.path.join(out_dir_cfg, 'model_best.pt'), map_location='cpu')
    load_model_manual(state_dict['state_dict'], model)

    generator = sap_config.get_generator(model, cfg, device=device)

    gen_dir = cfg['generation'].get('generation_dir', 'generation')
    out_root = args.out_dir or os.path.join(cfg['train']['out_dir'], gen_dir)
    mesh_dir = os.path.join(out_root, 'meshes')
    in_dir = os.path.join(out_root, 'input')
    os.makedirs(mesh_dir, exist_ok=True); os.makedirs(in_dir, exist_ok=True)

    dpsr = DPSR(res=(cfg['generation']['psr_resolution'],)*3, sig=cfg['generation']['psr_sigma']).to(device)

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        print("No files found.")
        sys.exit(1)

    print(f"Found {len(files)} files. Writing meshes to: {mesh_dir}")
    model.eval()
    with torch.no_grad():
        for idx, fp in enumerate(files):
            name = os.path.splitext(os.path.basename(fp))[0]
            pts = read_ply_points(fp)

            # sample to N
            if pts.shape[0] >= args.points:
                sel = np.random.choice(pts.shape[0], args.points, replace=False)
            else:
                sel = np.random.choice(pts.shape[0], args.points, replace=True)
            pts = pts[sel].astype(np.float32)

            # normalize with margin
            pts_norm, center, scale, tag = normalize_safe(pts, mode=args.norm, target=args.target)

            # stats
            m, M = pts_norm.min(0), pts_norm.max(0)
            print(f"[{name}] norm={tag} range=[{m.min():.3f},{M.max():.3f}] (target=±{args.target})")

            p = torch.from_numpy(pts_norm).float().unsqueeze(0).to(device)
            data = {'inputs': p, 'idx': torch.tensor(idx)}

            out = generator.generate_mesh(data)
            v, f, points, normals = out[:4]

            # save
            export_mesh(os.path.join(mesh_dir, f"{name}.off"), scale2onet(v), f)
            # also save the (normalized) input we fed the model
            export_pointcloud(os.path.join(in_dir, f"{name}.ply"), p[0])

            print(f"[OK] {name} | V={v.shape[0]} F={f.shape[0]}")
    print("Done.")
    
if __name__ == "__main__":
    main()
'''
python infer_folder_v3.py \
  --config configs/learning_based/noise_small/ours_pretrained.yaml \
  --input_dir /root/autodl-tmp/Point-Cloud-Flow-Matching/runs/scissors_hybrid/samples_recon_ep2800 \
  --points 3000 \
  --norm auto \
  --target 0.48 \
  --psr_res 256


python infer_folder_v3.py \
  --config configs/learning_based/outlier/ours_3plane.yaml \
  --input_dir /root/autodl-tmp/Point-Cloud-Flow-Matching/runs/scissors_hybrid/samples_recon_ep2800 \
  --points 3000 \
  --norm auto \
  --target 0.48 \
  --psr_res 256


python infer_folder_v3.py \
  --config configs/optim_based/thingi_noisy.yaml \
  --input_dir /root/autodl-tmp/Point-Cloud-Flow-Matching/runs/scissors_hybrid/samples_recon_ep2800 \
  --points 3000 \
  --norm auto \
  --target 0.48 \
  --psr_res 256


'''