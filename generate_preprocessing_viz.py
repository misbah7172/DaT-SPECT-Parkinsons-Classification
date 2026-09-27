"""Generate before/after preprocessing visualization."""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

OUTPUT_DIR = "E:/DaT/assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Raw Data Analysis (full dataset) ────────────────────────────────────
import nibabel as nib

path = 'Dataset/DaT_Parkinsons_Challenge_-_niftis.zip'
shapes = Counter()
zooms = Counter()
sizes = []

for f in os.listdir(path):
    try:
        img = nib.load(os.path.join(path, f))
        shape = img.shape[:3]
        shapes[shape] += 1
        z = tuple(round(float(zz), 2) for zz in img.header.get_zooms()[:3])
        zooms[z] += 1
        # Physical size in mm
        phys = tuple(s * float(z) for s, z in zip(shape, img.header.get_zooms()[:3]))
        sizes.append((shape, z, phys))
    except:
        pass

print(f"Total scans analyzed: {len(sizes)}")

# ─── Before Preprocessing ────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 14))

# 1. Shape distribution (top 15)
ax1 = plt.subplot(3, 3, 1)
top_shapes = shapes.most_common(15)
shape_labels = [f"{s[0]}×{s[1]}×{s[2]}" for s, _ in top_shapes]
shape_counts = [c for _, c in top_shapes]
colors = plt.cm.viridis(np.linspace(0, 1, len(shape_labels)))
bars = ax1.barh(range(len(shape_labels)), shape_counts, color=colors, edgecolor='black')
ax1.set_yticks(range(len(shape_labels)))
ax1.set_yticklabels(shape_labels, fontsize=9)
ax1.invert_yaxis()
ax1.set_xlabel('Count')
ax1.set_title('Top 15 Raw Shapes (D×H×W)', fontweight='bold', fontsize=11)
for bar, count in zip(bars, shape_counts):
    ax1.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
             str(count), va='center', fontsize=8)

# 2. Voxel size distribution
ax2 = plt.subplot(3, 3, 2)
voxel_labels = [f"{v[0]}×{v[1]}×{v[2]}" for v, _ in zooms.most_common()]
voxel_counts = [c for _, c in zooms.most_common()]
colors2 = plt.cm.plasma(np.linspace(0, 1, len(voxel_labels)))
wedges, texts, autotexts = ax2.pie(voxel_counts, labels=voxel_labels, autopct='%1.1f%%',
                                    colors=colors2, startangle=90, textprops={'fontsize': 8})
ax2.set_title('Raw Voxel Size Distribution (mm)', fontweight='bold', fontsize=11)

# 3. Physical size (mm) distribution
ax3 = plt.subplot(3, 3, 3)
phys_sizes = [p for _, _, p in sizes]
d_phys = [p[0] for p in phys_sizes]
h_phys = [p[1] for p in phys_sizes]
w_phys = [p[2] for p in phys_sizes]
ax3.scatter(d_phys, h_phys, alpha=0.5, s=10, c='blue', label='Depth×Height')
ax3.scatter(d_phys, w_phys, alpha=0.5, s=10, c='red', label='Depth×Width')
ax3.axhline(y=160, color='green', linestyle='--', label='Target 160mm (80×2.0)')
ax3.axvline(x=160, color='green', linestyle='--')
ax3.set_xlabel('Depth (mm)')
ax3.set_ylabel('Height/Width (mm)')
ax3.set_title('Physical Size Distribution (mm)', fontweight='bold', fontsize=11)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# 4. Slice count distribution (Z dimension)
ax4 = plt.subplot(3, 3, 4)
z_counts = Counter(s[0][2] for s in sizes)
z_slices = sorted(z_counts.keys())
z_freq = [z_counts[z] for z in z_slices]
ax4.bar(z_slices, z_freq, color='steelblue', edgecolor='black', width=2)
ax4.axvline(x=80, color='red', linestyle='--', linewidth=2, label='Target: 80 slices')
ax4.set_xlabel('Number of Slices (Z)')
ax4.set_ylabel('Count')
ax4.set_title('Slice Count Distribution', fontweight='bold', fontsize=11)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

# 5. In-plane resolution (X, Y)
ax5 = plt.subplot(3, 3, 5)
xy_sizes = [(s[0], s[1]) for s, _, _ in sizes]
x_sizes = [x for x, _ in xy_sizes]
y_sizes = [y for _, y in xy_sizes]
ax5.scatter(x_sizes, y_sizes, alpha=0.5, s=10, c='purple')
ax5.plot([64, 256], [64, 256], 'k--', alpha=0.3)
ax5.axvline(x=80, color='red', linestyle='--', label='Target: 80')
ax5.axhline(y=80, color='red', linestyle='--')
ax5.set_xlabel('X Dimension (voxels)')
ax5.set_ylabel('Y Dimension (voxels)')
ax5.set_title('In-Plane Dimensions (X vs Y)', fontweight='bold', fontsize=11)
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# ─── After Preprocessing ─────────────────────────────────────────────────
# All resampled to 2.0mm isotropic, COM-centered on 80³ grid
ax6 = plt.subplot(3, 3, 6)
# Show the transformation
stages = ['Raw\n(Various)', 'Resample\n2.0mm', 'COM-center\n80³ grid', 'Z-score\nNormalize']
stage_desc = [
    f'{len(shapes)} unique shapes\n{len(zooms)} voxel sizes\n1363 scans',
    'All → 2.0mm isotropic\nTrilinear interp.',
    'Fixed 80×80×80\nCOM aligned',
    'Per-volume\nμ=0, σ=1'
]
y_pos = np.arange(len(stages))
ax6.barh(y_pos, [1]*len(stages), height=0.6, color=['#C73E1D', '#F18F01', '#2E86AB', '#6A994E'], edgecolor='black')
ax6.set_yticks(y_pos)
ax6.set_yticklabels(stages, fontsize=10)
for i, desc in enumerate(stage_desc):
    ax6.text(1.05, i, desc, va='center', fontsize=9, ha='left')
ax6.set_xlim(0, 2.5)
ax6.set_xticks([])
ax6.set_title('Preprocessing Pipeline', fontweight='bold', fontsize=11)

# 7. After: Uniform 80³
ax7 = plt.subplot(3, 3, 7)
# All same shape after preprocessing
ax7.bar(['After'], [1363], color='#6A994E', edgecolor='black', width=0.5)
ax7.text(0, 680, '80×80×80', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
ax7.text(0, 200, '2.0mm isotropic', ha='center', va='center', fontsize=11, color='white')
ax7.text(0, -100, '1,362 scans\n(1 failed QC)', ha='center', va='center', fontsize=10, color='white')
ax7.set_ylim(0, 1400)
ax7.set_ylabel('Number of Scans')
ax7.set_title('After Preprocessing: Uniform', fontweight='bold', fontsize=11)

# 8. Physical coverage comparison
ax8 = plt.subplot(3, 3, 8)
# Before: physical size varies
before_d = [p[0] for _, _, p in sizes]
before_h = [p[1] for _, _, p in sizes]
before_w = [p[2] for _, _, p in sizes]
ax8.hist(before_d, bins=20, alpha=0.5, label='Depth (raw)', color='red', density=True)
ax8.hist(before_h, bins=20, alpha=0.5, label='Height (raw)', color='blue', density=True)
ax8.hist(before_w, bins=20, alpha=0.5, label='Width (raw)', color='green', density=True)
ax8.axvline(x=160, color='black', linestyle='--', linewidth=2, label='Target: 160mm (80×2.0)')
ax8.set_xlabel('Physical Size (mm)')
ax8.set_ylabel('Density')
ax8.set_title('Physical Coverage Before vs After', fontweight='bold', fontsize=11)
ax8.legend(fontsize=8)
ax8.grid(True, alpha=0.3)

# 9. Summary statistics table
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')
summary_data = [
    ['Metric', 'Before Preprocessing', 'After Preprocessing'],
    ['Number of Scans', '1,363', '1,362 (1 failed QC)'],
    ['Unique Shapes', str(len(shapes)), '1 (80×80×80)'],
    ['Unique Voxel Sizes', str(len(zooms)), '1 (2.0×2.0×2.0mm)'],
    ['Shape Range (D)', f'{min(s[0][0] for s in sizes)}–{max(s[0][0] for s in sizes)}', '80'],
    ['Shape Range (H)', f'{min(s[0][1] for s in sizes)}–{max(s[0][1] for s in sizes)}', '80'],
    ['Shape Range (W)', f'{min(s[0][2] for s in sizes)}–{max(s[0][2] for s in sizes)}', '80'],
    ['Voxel Size Range', f'{min(min(v) for v in zooms.keys()):.2f}–{max(max(v) for v in zooms.keys()):.2f}mm', '2.0mm isotropic'],
    ['Physical Range (mm)', f'{min(min(p) for p in phys_sizes):.0f}–{max(max(p) for p in phys_sizes):.0f}', '160×160×160mm'],
    ['Normalization', 'None (scanner-dependent)', 'Per-volume z-score'],
]
table = ax9.table(cellText=summary_data, loc='center', cellLoc='left', colWidths=[0.3, 0.35, 0.35])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.8)
for i in range(len(summary_data)):
    for j in range(3):
        cell = table[(i, j)]
        if i == 0:
            cell.set_facecolor('#2E86AB')
            cell.set_text_props(weight='bold', color='white')
        elif i % 2 == 0:
            cell.set_facecolor('#F5F5F5')
ax9.set_title('Before vs After Summary', fontweight='bold', fontsize=11, pad=20)

plt.suptitle('DaT-SPECT Preprocessing: Before → After Analysis\n(1,363 NIfTI scans from 15 scanner sites)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/preprocessing_before_after.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: preprocessing_before_after.png")

# ─── Additional: Detailed voxel size table ───────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Top 20 voxel sizes with details
voxel_details = zooms.most_common(20)
table_data = [['Rank', 'Voxel Size (mm)', 'Count', '% of Total', 'Typical Shape(s)']]
total = sum(zooms.values())
for i, (v, c) in enumerate(voxel_details, 1):
    # Find typical shapes for this voxel size
    typical_shapes = Counter(s for s, z, _ in sizes if tuple(round(float(zz),2) for zz in z) == v)
    top_shape = typical_shapes.most_common(1)[0][0] if typical_shapes else 'N/A'
    table_data.append([
        str(i),
        f'{v[0]}×{v[1]}×{v[2]}',
        str(c),
        f'{c/total*100:.1f}%',
        f'{top_shape[0]}×{top_shape[1]}×{top_shape[2]}'
    ])
table_data.append(['...', 'Other sizes', str(total - sum(c for _, c in voxel_details)), 
                   f'{(total - sum(c for _, c in voxel_details))/total*100:.1f}%', 'Various'])

table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                 colWidths=[0.08, 0.22, 0.12, 0.12, 0.22])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)
for i in range(len(table_data)):
    for j in range(5):
        cell = table[(i, j)]
        if i == 0:
            cell.set_facecolor('#2E86AB')
            cell.set_text_props(weight='bold', color='white')
        elif i % 2 == 0:
            cell.set_facecolor('#F5F5F5')
plt.title('Raw Voxel Size Details (Top 20 of 66 unique)', fontsize=14, fontweight='bold', pad=20)
plt.savefig(f'{OUTPUT_DIR}/voxel_size_details.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: voxel_size_details.png")

print("\nDone! Generated 2 new visualizations in", OUTPUT_DIR)