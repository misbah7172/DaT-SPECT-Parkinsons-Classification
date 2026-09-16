"""Build v28: v23 proven weights + v27 balanced CNN weights for diversity.
The hypothesis: adding more diverse CNN models may help slightly."""
import os, json, shutil

v28 = r'E:\DaT\submission_v28'
os.makedirs(os.path.join(v28, 'weights'), exist_ok=True)

# Start with v23's exact code and weights
for f in ['main.py', 'cnn_infer.py', 'sbr_extractor.py', 'atlas_template.npy']:
    src = os.path.join(r'E:\DaT\submission_v23', f)
    dst = os.path.join(v28, f)
    if os.path.exists(src):
        shutil.copy2(src, dst)

# Copy v23 weights
v23_w = r'E:\DaT\submission_v23\weights'
v28_w = os.path.join(v28, 'weights')
for f in os.listdir(v23_w):
    shutil.copy2(os.path.join(v23_w, f), os.path.join(v28_w, f))

# Add v27 balanced full-data models for diversity (4 more seeds)
# These are different from v23's seeds, adding diversity
added = 0
for seed in [42, 777, 2024, 100]:
    src = rf'E:\DaT\submission_v27\weights\deep_r_{seed}_full.pt'
    dst_name = f'deep_r{seed}_bal.pt'
    if os.path.exists(src):
        # Load and re-save with correct arch field
        import torch
        ck = torch.load(src, map_location='cpu', weights_only=False)
        ck['arch'] = 'Net3dR'
        # Rename to avoid collision with v23's r_42 etc
        torch.save(ck, os.path.join(v28_w, dst_name))
        added += 1

print(f"Added {added} balanced full-data models")

# Update ship_final.json: keep v23's blend weights
# v23's w_deep6=0.8912, w_sbr=0.1088, T=0.7731
# With 10 models now (6 v23 + 4 balanced), CNN weight should be even higher
with open(os.path.join(v28_w, 'ship_final.json')) as f:
    ship = json.load(f)

print(f"\nv28 config:")
print(f"  CNN models: 6 (v23) + 4 (balanced) = 10")
print(f"  w_deep: {ship.get('w_deep6', ship.get('w_deep', '?'))}")
print(f"  w_sbr: {ship.get('w_sbr', '?')}")
print(f"  T: {ship.get('T', '?')}")

# List all files
print(f"\nv28 weights:")
for f in sorted(os.listdir(v28_w)):
    sz = os.path.getsize(os.path.join(v28_w, f))
    print(f"  {f}: {sz/1024/1024:.1f}MB")

# Create zip
import zipfile
zip_path = r'E:\DaT\submission_v28.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(v28):
        for f in files:
            fp = os.path.join(root, f)
            arc = os.path.relpath(fp, v28)
            zf.write(fp, arc)
sz = os.path.getsize(zip_path)
print(f"\nCreated {zip_path}: {sz/1024/1024:.1f}MB")
with zipfile.ZipFile(zip_path) as zf:
    print(f"Entries: {len(zf.infolist())}")
