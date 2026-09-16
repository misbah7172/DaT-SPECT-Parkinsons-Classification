"""Build v29: v23 proven code + 3 diverse architectures + optimized T.
Copy v23 code, add varb/se/ms arch support to cnn_infer.py,
update main.py allow list, set T=0.85."""
import os, json, shutil

v29 = r'E:\DaT\submission_v29'
os.makedirs(os.path.join(v29, 'weights'), exist_ok=True)

# Copy v23 code
for f in ['main.py', 'sbr_extractor.py', 'atlas_template.npy']:
    shutil.copy2(os.path.join(r'E:\DaT\submission_v23', f), os.path.join(v29, f))

# Copy v23 SBR weights
v23_w = r'E:\DaT\submission_v23\weights'
v29_w = os.path.join(v29, 'weights')
for f in os.listdir(v23_w):
    if f.startswith('sbr_') or f == 'ship_final.json':
        shutil.copy2(os.path.join(v23_w, f), os.path.join(v29_w, f))

# Copy v23 CNN weights
for f in ['deep_r_42.pt', 'deep_r_777.pt', 'deep_r_2024.pt', 'deep_r_100.pt',
          'deep_big_2025.pt', 'deep_big_1984.pt']:
    shutil.copy2(os.path.join(v23_w, f), os.path.join(v29_w, f))

# Copy diverse CNN weights
diverse_w = r'E:\DaT\submission_v29\weights'
for f in os.listdir(diverse_w):
    if f.startswith('deep_') and f.endswith('.pt') and any(a in f for a in ['varb', 'se', 'ms']):
        pass  # already there from training
# Actually the training saved directly to v29/weights
# Let me check
import glob
diverse_pts = glob.glob(os.path.join(v29_w, 'deep_*.pt'))
print(f"CNN weights in v29:")
for f in sorted(diverse_pts):
    print(f"  {os.path.basename(f)}")

# Update ship_final.json: T=0.85 (optimized from OOF)
with open(os.path.join(v29_w, 'ship_final.json')) as f:
    ship = json.load(f)
ship['T'] = 0.85
with open(os.path.join(v29_w, 'ship_final.json'), 'w') as f:
    json.dump(ship, f)
print(f"\nship_final.json: {json.dumps(ship)}")

print(f"\nv29 weights: {sorted(os.listdir(v29_w))}")
