"""Generate the 4-stage methodological pipeline diagram."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

OUTPUT_DIR = "E:/DaT/Paper/figures"

fig, ax = plt.subplots(figsize=(16, 6))
ax.set_xlim(0, 16)
ax.set_ylim(0, 4)
ax.axis('off')

# Colors for each stage
stage_colors = {
    1: '#2E86AB',  # Blue
    2: '#A23B72',  # Purple
    3: '#F18F01',  # Orange
    4: '#C73E1D',  # Red
}

# Stage definitions
stages = [
    {
        'num': 1,
        'title': 'Stage 1:\nSpatial \& Volumetric\nNormalization',
        'details': [
            'RAS Reorientation',
            '2.0 mm³ Isotropic Resampling',
            'Center-of-Mass Alignment',
            '80³ Bounding Box Crop',
            'Per-Volume Z-Score Norm.'
        ],
        'color': stage_colors[1]
    },
    {
        'num': 2,
        'title': 'Stage 2:\nModel Exploration\n(Dual Track)',
        'details': [
            'Track A: End-to-End 3D Deep Learning',
            '   • 3D ResNet v25 (845K/1.3M)',
            '   • Mixup + Label Smoothing',
            '   • 15-View TTA',
            'Track B: Quantitative ROI Biomarkers',
            '   • 93 SBR/Asymmetry Features',
            '   • XGBoost / LightGBM / LR'
        ],
        'color': stage_colors[2]
    },
    {
        'num': 3,
        'title': 'Stage 3:\nSite-Aware Leakage-Free\nValidation',
        'details': [
            'Stratified Group K-Fold (K=5)',
            'Groups = 15 Acquisition Sites',
            'Stratified by Binary Label',
            'Fold-Isolated Preprocessing',
            'No Site Overlap Train/Val'
        ],
        'color': stage_colors[3]
    },
    {
        'num': 4,
        'title': 'Stage 4:\nPost-Hoc Probability\nCalibration',
        'details': [
            'Platt Scaling on OOF Logits',
            'p_cal = σ(a·logit(p) + b)',
            'a=1.45, b=0.20 (Grid Search)',
            'Minimizes Out-of-Sample LogLoss',
            'ECE: 0.0142 → 0.0061 (-57%)'
        ],
        'color': stage_colors[4]
    }
]

# Draw stages
stage_width = 3.7
stage_height = 3.5
gap = 0.3
start_x = 0.2

for i, stage in enumerate(stages):
    x = start_x + i * (stage_width + gap)
    y = 0.2
    
    # Stage box
    box = FancyBboxPatch(
        (x, y), stage_width, stage_height,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=stage['color'],
        edgecolor='white',
        linewidth=2,
        alpha=0.9
    )
    ax.add_patch(box)
    
    # Stage number circle
    circle = plt.Circle((x + 0.3, y + stage_height - 0.3), 0.25,
                       facecolor='white', edgecolor=stage['color'], linewidth=2)
    ax.add_patch(circle)
    ax.text(x + 0.3, y + stage_height - 0.3, str(stage['num']),
           ha='center', va='center', fontsize=14, fontweight='bold', color=stage['color'])
    
    # Title
    ax.text(x + stage_width/2, y + stage_height - 0.7, stage['title'],
           ha='center', va='top', fontsize=11, fontweight='bold', color='white',
           wrap=True)
    
    # Details
    detail_y = y + stage_height - 1.4
    for detail in stage['details']:
        ax.text(x + 0.3, detail_y, detail,
               ha='left', va='top', fontsize=8.5, color='white',
               fontfamily='monospace')
        detail_y -= 0.35
    
    # Arrow to next stage
    if i < len(stages) - 1:
        arrow_x = x + stage_width + 0.05
        ax.annotate('', xy=(arrow_x, y + stage_height/2), 
                   xytext=(x + stage_width - 0.05, y + stage_height/2),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=2, shrinkA=5, shrinkB=5))

# Main title
ax.text(8, 3.8, 'Four-Stage Methodological Framework for Robust Multi-Center 3D Medical Image Classification',
       ha='center', va='center', fontsize=14, fontweight='bold', color='#333333')

# Input/Output labels
ax.text(0.5, -0.3, 'Raw Multi-Center 3D Volumes\n(Heterogeneous: 66 voxel sizes, 15 sites)',
       ha='center', va='top', fontsize=10, style='italic', color='#666666')
ax.text(15.5, -0.3, 'Calibrated Probabilities\n(AUROC=0.9610, LogLoss=0.2453)',
       ha='center', va='top', fontsize=10, style='italic', color='#666666')

# Data flow arrow at bottom
ax.annotate('', xy=(15, -0.15), xytext=(1.5, -0.15),
           arrowprops=dict(arrowstyle='->', color='#888888', lw=1.5, shrinkA=5, shrinkB=5))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/method_pipeline.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(f'{OUTPUT_DIR}/method_pipeline.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: method_pipeline.pdf and method_pipeline.png")