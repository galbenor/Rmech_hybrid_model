"""
Generate all five figures for:
  "Mechanistic information: a pre-trial quality criterion for adaptive precision medicine"
Target: 300 dpi PDFs, Nature Medicine style (sans-serif, clean, 180 mm wide for main figs)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import norm
from scipy.special import xlogy

np.random.seed(2024)

# ─── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 7,
    'axes.labelsize': 7,
    'axes.titlesize': 7.5,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'legend.fontsize': 6.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 1.2,
    'grid.linewidth': 0.4,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

NM_W = 7.09      # 180 mm in inches
NM_W2 = 3.46     # 88 mm (single column)
THRESHOLD = 1.0
C_FAIL = '#d73027'
C_PASS = '#1a9641'
C_GRAY = '#636363'
C_BLUE = '#2166ac'
C_ORANGE = '#f46d43'

def panel_label(ax, letter, x=-0.18, y=1.08):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='top')

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Framework
# ═══════════════════════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(1, 3, figsize=(NM_W, 2.2))

# ── Panel a: adaptive treatment selection schematic ──────────────────────────
ax = axes1[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.axis('off')

def box(ax, x, y, w, h, text, fc='#deebf7', ec='#2166ac', fontsize=6.5, bold=False):
    rect = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.15',
                          fc=fc, ec=ec, linewidth=0.8)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold' if bold else 'normal',
            wrap=True)

def arrow(ax, x1, y1, x2, y2, color='#555555'):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=0.8))

# Mechanistic model box
box(ax, 0.3, 7.2, 4.0, 1.6, 'Mechanistic\nmodel\n(pre-treatment)', fc='#fff7bc', ec='#d95f02')
# Patient box
box(ax, 0.3, 4.8, 4.0, 1.6, 'Patient\nbiomarkers\n(mutation, LDH, TPS)', fc='#e5f5e0', ec='#31a354')
# Recommendation
box(ax, 5.5, 7.2, 4.0, 1.6, 'Treatment\nrecommendation\n$\\hat{\\pi}$', fc='#deebf7', ec='#2166ac')
# Adaptive algorithm
box(ax, 5.5, 4.5, 4.0, 1.8, 'Adaptive\nalgorithm\n(cycles 1…N)', fc='#f0f0f0', ec='#636363')
# Outcome
box(ax, 5.5, 1.8, 4.0, 1.6, 'Outcome\nobservation\n$r_t$', fc='#fee8c8', ec='#e6550d')
# Optimal treatment
box(ax, 0.3, 1.8, 4.0, 1.6, 'Optimal\ntreatment\n$\\pi^*$ (unknown)', fc='#fde0dd', ec='#de2d26')

arrow(ax, 4.3, 8.0, 5.5, 8.0)
arrow(ax, 4.3, 5.6, 5.5, 5.6)
arrow(ax, 7.5, 7.2, 7.5, 6.3)
arrow(ax, 7.5, 4.5, 7.5, 3.4)
arrow(ax, 5.5, 2.6, 4.3, 2.6)

# R_mech label on arrow
ax.text(4.9, 8.3, '$R_{mech}$', ha='center', fontsize=7, color='#d95f02', style='italic')
ax.text(4.9, 5.9, 'covariates', ha='center', fontsize=6, color='#31a354')

ax.set_title('a', loc='left', fontweight='bold', fontsize=9, pad=2)

# ── Panel b: information flow bar ────────────────────────────────────────────
ax = axes1[1]
H_mu_ex = 1.62
R_ex = 0.72
H_mech = H_mu_ex - R_ex

# Stacked horizontal bar
bar_h = 0.5
y_pos = 0.5
ax.barh(y_pos, H_mu_ex, height=bar_h, color='#f7f7f7', edgecolor='#636363', linewidth=0.8)
ax.barh(y_pos, R_ex, height=bar_h, color=C_ORANGE, edgecolor='none', label=f'$R_{{mech}}$ = {R_ex}')
ax.axvline(1.0, color=C_FAIL, linestyle='--', lw=1.0, label='Threshold 1.0 nat')

# Annotations
ax.text(R_ex/2, y_pos, f'$R_{{mech}}$\n{R_ex} nats\n(model)', ha='center', va='center',
        fontsize=6, color='white', fontweight='bold')
ax.text(R_ex + H_mech/2, y_pos, f'$H_{{mech}}$\n{H_mech:.2f} nats\n(algorithm)', 
        ha='center', va='center', fontsize=6, color='#333333')
ax.text(H_mu_ex + 0.03, y_pos, f'$H(\\mu)$ = {H_mu_ex} nats', 
        va='center', fontsize=6.5)

ax.set_xlabel('Mechanistic information (nats)')
ax.set_xlim(0, 2.2)
ax.set_ylim(0, 1.2)
ax.set_yticks([])
ax.legend(loc='upper right', frameon=False, fontsize=6)
ax.set_title('b', loc='left', fontweight='bold', fontsize=9, pad=2)

# ── Panel c: channel capacity vs ODE bias ────────────────────────────────────
ax = axes1[2]

# C(B_mu) for two H(mu) values
sigma = 0.4; kappa = 1.5; d_F = 3; sigma_F2 = 2*sigma**2*1.62/(kappa**2*d_F)
B_mu = np.linspace(0, 1.2, 200)
C1 = (d_F/2) * np.log(1 + kappa**2 * sigma_F2 / (kappa**2*B_mu**2 + sigma**2))
sigma_F2b = 2*sigma**2*2.0/(kappa**2*d_F)
C2 = (d_F/2) * np.log(1 + kappa**2 * sigma_F2b / (kappa**2*B_mu**2 + sigma**2))

ax.plot(B_mu, C1, color=C_BLUE, lw=1.3, label='$H(\\mu)$ = 1.62 nats (5-FU)')
ax.plot(B_mu, C2, color=C_ORANGE, lw=1.3, label='$H(\\mu)$ = 2.02 nats (insulin)')
ax.axhline(THRESHOLD, color=C_FAIL, linestyle='--', lw=1.0, label='1.0 nat threshold')
ax.fill_between(B_mu, THRESHOLD, np.minimum(C1, 4), where=C1 >= THRESHOLD,
                alpha=0.12, color=C_PASS)
ax.fill_between(B_mu, 0, np.minimum(C1, THRESHOLD), where=C1 < THRESHOLD,
                alpha=0.08, color=C_FAIL)

ax.set_xlabel('ODE bias $B_\\mu$ (model imprecision)')
ax.set_ylabel('Max achievable $R_{mech}$ (nats)')
ax.legend(frameon=False, loc='upper right', fontsize=6)
ax.set_xlim(0, 1.2); ax.set_ylim(0, 2.2)
ax.text(0.65, 1.12, 'Model\naccelerates\nconvergence', fontsize=5.5,
        color=C_PASS, ha='center')
ax.text(0.85, 0.4, 'No meaningful\ndifference', fontsize=5.5,
        color=C_FAIL, ha='center')
ax.set_title('c', loc='left', fontweight='bold', fontsize=9, pad=2)

fig1.tight_layout(w_pad=2.0)
fig1.savefig('/home/claude/fig1_framework.pdf', dpi=300)
print("Fig 1 done")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Audit summary
# ═══════════════════════════════════════════════════════════════════════════════
settings = [
    ('5-FU FOLFOX',        1.62, 0.32, 2, 'Oncology'),
    ('BRAF melanoma',      1.13, 0.76, 3, 'Oncology'),
    ('Pembro selection',   1.28, 0.31, 2, 'Oncology'),
    ('Pembro continuation',0.65, 0.21, 1, 'Oncology'),
    ('Lithium (BD)',        0.45, 0.20, 1, 'Psychiatry'),
    ('Nortriptyline (MDD)', 1.76, 0.62, 4, 'Psychiatry'),
    ('Valproate (epilepsy)',1.67, 0.61, 2, 'Neurology'),
    ('Insulin T1D',        2.02, 0.59, 2, 'Endocrinology'),
    ('Busulfan HSCT\n(prior+TDM)', 1.76, 1.30, 0, 'Oncology'),
]
labels, H_vals, R_vals, modes, fields = zip(*settings)

MODE_COLORS = {1: '#e41a1c', 2: '#ff7f00', 3: '#984ea3', 4: '#377eb8', 0: C_PASS}
mode_labels = {1: 'Mode 1: near-binary', 2: 'Mode 2: model imprecise',
               3: 'Mode 3: dominant biomarker', 4: 'Mode 4: dose range', 0: 'Above threshold'}

fig2 = plt.figure(figsize=(NM_W, 3.0))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.6, 1.0], figure=fig2, wspace=0.45)

ax_a = fig2.add_subplot(gs[0])
ax_b = fig2.add_subplot(gs[1])

y_pos = np.arange(len(labels))
bar_colors = [MODE_COLORS[m] for m in modes]

# H(mu) as outline bars
bars_h = ax_a.barh(y_pos, H_vals, height=0.55, fc='none',
                   edgecolor='#aaaaaa', linewidth=0.8, label='$H(\\mu)$ (prior entropy)')
# R_mech as solid bars
for i, (y, r, c) in enumerate(zip(y_pos, R_vals, bar_colors)):
    ax_a.barh(y, r, height=0.55, color=c, alpha=0.85)

ax_a.axvline(THRESHOLD, color=C_FAIL, linestyle='--', lw=1.0, label='1.0 nat threshold')

# Field separators
field_groups = [0, 4, 6, 7, 8]  # start indices of each field group
for g in field_groups[1:]:
    ax_a.axhline(g - 0.5, color='#dddddd', lw=0.5)

ax_a.set_yticks(y_pos)
ax_a.set_yticklabels(labels, fontsize=6)
ax_a.set_xlabel('Nats')
ax_a.set_xlim(0, 2.4)
ax_a.legend(handles=[
    mpatches.Patch(facecolor='none', edgecolor='#aaaaaa', linewidth=0.8, label='$H(\\mu)$: prior entropy'),
    mpatches.Patch(facecolor='#888', label='$R_{mech}$: model information'),
    plt.Line2D([0],[0], color=C_FAIL, linestyle='--', lw=1.0, label='1.0 nat threshold')
], frameon=False, fontsize=5.5, loc='lower right')

# Field labels on right
field_spans = [('Oncology', 0, 3.5), ('Psychiatry', 4, 5.5),
               ('Neurology', 6, 6), ('Endocrinology', 7, 7)]
for fname, ylo, yhi in field_spans:
    ax_a.text(2.38, (ylo+yhi)/2, fname, va='center', ha='right',
              fontsize=6, color='#444444', rotation=0)

panel_label(ax_a, 'a', x=-0.22)

# ── Panel b: failure mode taxonomy ──────────────────────────────────────────
ax_b.set_xlim(0, 10); ax_b.set_ylim(0, 10); ax_b.axis('off')

mode_data = [
    (1, C_FAIL,    'Mode 1\nNear-binary\ndecision',      'Add treatment arms\nor restrict\nsubpopulation',   8.2),
    (2, C_ORANGE,  'Mode 2\nModel too\nimprecise',        'Enrich ODE with\npharmacogenomic\ncovariates',    5.8),
    (3, '#984ea3', 'Mode 3\nDominant\nbiomarker',         'Analyse\nunsegregated\nsubpopulation',            3.4),
    (4, C_BLUE,    'Mode 4\nDose range\ntoo narrow',      'Extend arm set\nto cover PM/UM\nmetabolisers',    1.0),
]

for mode_num, color, cause_text, rx_text, yc in mode_data:
    # Cause box
    rect = FancyBboxPatch((0.2, yc), 4.0, 2.0, boxstyle='round,pad=0.1',
                          fc=color, ec='none', alpha=0.15)
    ax_b.add_patch(rect)
    rect2 = FancyBboxPatch((0.2, yc), 4.0, 2.0, boxstyle='round,pad=0.1',
                           fc='none', ec=color, linewidth=0.8)
    ax_b.add_patch(rect2)
    ax_b.text(2.2, yc+1.0, cause_text, ha='center', va='center',
              fontsize=6, color=color, fontweight='bold')
    # Arrow
    ax_b.annotate('', xy=(6.0, yc+1.0), xytext=(4.2, yc+1.0),
                  arrowprops=dict(arrowstyle='->', color='#888888', lw=0.7))
    # Prescription box
    rect3 = FancyBboxPatch((6.0, yc), 3.7, 2.0, boxstyle='round,pad=0.1',
                           fc='#f7f7f7', ec='#888888', linewidth=0.6)
    ax_b.add_patch(rect3)
    ax_b.text(7.85, yc+1.0, rx_text, ha='center', va='center',
              fontsize=5.8, color='#333333')

ax_b.text(5.0, 9.5, '← Failure cause       Prescription →', ha='center',
          fontsize=6, color='#555555', style='italic')
panel_label(ax_b, 'b', x=-0.08)

fig2.savefig('/home/claude/fig2_audit.pdf', dpi=300)
print("Fig 2 done")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — 5-FU
# ═══════════════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(1, 3, figsize=(NM_W, 2.2))

# ── Panel a: R_mech vs R² for 5-FU ──────────────────────────────────────────
ax = axes3[0]
sigma_CL = 0.75; H_mu = 1.62; sigma_AUC_frac = 0.08
K = 8; d_F = 3
kappa = 1.8

R2_vals = np.linspace(0.05, 0.97, 200)
Rmech_vals = []
for R2 in R2_vals:
    s_res = np.sqrt(1-R2)*sigma_CL
    s_F2 = 2*sigma_AUC_frac**2 * H_mu / (kappa**2 * d_F)
    cap = (d_F/2) * np.log(1 + kappa**2*s_F2/(kappa**2*s_res**2 + sigma_AUC_frac**2))
    cap = min(cap, H_mu)
    Rmech_vals.append(cap)

ax.plot(R2_vals, Rmech_vals, color=C_BLUE, lw=1.4)
ax.axhline(THRESHOLD, color=C_FAIL, linestyle='--', lw=1.0)
ax.axvline(0.51, color=C_GRAY, linestyle=':', lw=0.9)
ax.axvline(0.75, color=C_PASS, linestyle=':', lw=0.9)

ax.scatter([0.51], [0.37], s=30, color=C_FAIL, zorder=5)
ax.scatter([0.75], [1.10], s=30, color=C_PASS, zorder=5)

ax.annotate('Current\n($R^2=0.51$,\n$R_{mech}=0.37$)', xy=(0.51, 0.37),
            xytext=(0.25, 0.7), fontsize=5.5, color=C_FAIL,
            arrowprops=dict(arrowstyle='->', color=C_FAIL, lw=0.7))
ax.annotate('Enriched\n($R^2=0.75$,\n$R_{mech}\\approx1.1$)', xy=(0.75, 1.10),
            xytext=(0.5, 1.5), fontsize=5.5, color=C_PASS,
            arrowprops=dict(arrowstyle='->', color=C_PASS, lw=0.7))

ax.fill_between(R2_vals, THRESHOLD, Rmech_vals,
                where=np.array(Rmech_vals)>=THRESHOLD, alpha=0.12, color=C_PASS)

ax.set_xlabel('ODE $R^2$ (explained clearance variance)')
ax.set_ylabel('$R_{mech}$ (nats)')
ax.set_xlim(0, 1); ax.set_ylim(0, 2.0)
panel_label(axes3[0], 'a')

# ── Panel b: Observation model benefit ───────────────────────────────────────
ax = axes3[1]
strategies = ['BSA\nfixed', 'Uninformed\nTS binary', 'ODE prior\nTS binary',
              'Gamelin\nrule', 'Pop prior\nTS cont.', 'ODE prior\nTS cont.']
attain_c3 = [20.9, 32.1, 34.8, 42.3, 55.7, 59.7]
colors_b = [C_GRAY, '#9ecae1', '#6baed6', '#fc8d59', '#a1d99b', '#31a354']

bars = ax.bar(range(6), attain_c3, color=colors_b, edgecolor='white', linewidth=0.5)
ax.axhline(THRESHOLD*100/1.62, color='#aaaaaa', linestyle=':', lw=0.7)

# Bracket for "continuous AUC" group
ax.annotate('', xy=(5.4, 57), xytext=(3.6, 57),
            arrowprops=dict(arrowstyle='<->', color='#555555', lw=0.8))
ax.text(4.5, 58, 'Continuous\nAUC', ha='center', fontsize=5.5, color='#555555')

ax.set_xticks(range(6)); ax.set_xticklabels(strategies, fontsize=5.5, rotation=0)
ax.set_ylabel('Cycle-3 therapeutic\nattainment (%)')
ax.set_ylim(0, 72)
ax.set_title('Observation model: 5-FU dosing strategies', fontsize=6.5, pad=3)
panel_label(axes3[1], 'b')

# ── Panel c: Information budget ───────────────────────────────────────────────
ax = axes3[2]
categories = ['Prior\nentropy\n$H(\\mu)$', '$R_{mech}$\n(prior)', '$R_{mech}$\n(+TDM)', 'Residual\n$H_{mech}$']
values =     [1.62,           0.32,         0.97,          0.65]
bar_colors_c = ['#9ecae1', C_ORANGE, C_PASS, '#fc9272']

# Waterfall-style
bar_starts = [0, 0, 0, 0.97]
bar_vals   = [1.62, 0.32, 0.97, 0.65]
bar_cs     = ['#9ecae1', C_ORANGE, C_PASS, '#fc9272']
bar_labels = ['$H(\\mu)=1.62$', '$R_{mech}=0.32$\n(prior alone)',
              '$R_{mech}=0.97$\n(prior+TDM)', '$H_{mech}=0.65$\n(residual)']

for i, (start, val, c, lbl) in enumerate(zip(bar_starts, bar_vals, bar_cs, bar_labels)):
    ax.bar(i, val, bottom=start if i==3 else 0, color=c, edgecolor='white',
           linewidth=0.5, alpha=0.85)
    ax.text(i, (start+val/2 if i==3 else val/2), f'{val:.2f}', ha='center',
            va='center', fontsize=6, fontweight='bold', color='white')

ax.axhline(THRESHOLD, color=C_FAIL, linestyle='--', lw=1.0, label='1.0 nat threshold')
ax.set_xticks(range(4)); ax.set_xticklabels(bar_labels, fontsize=5.5)
ax.set_ylabel('Nats')
ax.set_ylim(0, 2.0)
ax.legend(frameon=False, fontsize=6)
panel_label(axes3[2], 'c')

fig3.tight_layout(w_pad=2.0)
fig3.savefig('/home/claude/fig3_5fu.pdf', dpi=300)
print("Fig 3 done")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — BRAF melanoma
# ═══════════════════════════════════════════════════════════════════════════════
fig4, axes4 = plt.subplots(1, 3, figsize=(NM_W, 2.2))

# ── Panel a: MAPK pathway schematic ──────────────────────────────────────────
ax = axes4[0]; ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis('off')

nodes = [
    (5.0, 9.0, 'BRAF V600E\n(constitutively\nactive)', '#d7191c'),
    (5.0, 6.8, 'pMEK\n(phosphorylated)', '#fdae61'),
    (5.0, 4.6, 'pERK\n(phosphorylated)', '#abd9e9'),
    (5.0, 2.4, 'Tumour cell\nproliferation', '#2c7bb6'),
    (5.0, 0.3, 'Serum LDH\n(measurable)', '#1a9641'),
]
for x, y, text, color in nodes:
    rect = FancyBboxPatch((x-2.0, y-0.7), 4.0, 1.4,
                          boxstyle='round,pad=0.1', fc=color, ec='none', alpha=0.25)
    ax.add_patch(rect)
    rect2 = FancyBboxPatch((x-2.0, y-0.7), 4.0, 1.4,
                           boxstyle='round,pad=0.1', fc='none', ec=color, lw=0.8)
    ax.add_patch(rect2)
    ax.text(x, y, text, ha='center', va='center', fontsize=5.8, color=color,
            fontweight='bold')

for i in range(len(nodes)-1):
    ax.annotate('', xy=(nodes[i+1][0], nodes[i+1][1]+0.7),
                xytext=(nodes[i][0], nodes[i][1]-0.7),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=0.8))

# Drug targets
ax.text(7.5, 8.0, 'BRAF\ninhibitors', fontsize=5.5, color='#d7191c', ha='left')
ax.text(7.5, 5.8, 'MEK\ninhibitors', fontsize=5.5, color='#fdae61', ha='left')
ax.annotate('', xy=(7.0, 8.0), xytext=(7.0, 6.0),
            arrowprops=dict(arrowstyle='-[', color='#aaaaaa', lw=0.8))

panel_label(axes4[0], 'a', x=-0.08)

# ── Panel b: Population restriction ──────────────────────────────────────────
ax = axes4[1]
scenarios = ['All\npatients\n(K=4)', 'BRAF+\nonly\n(K=3)', 'BRAF+ +\nV600 type\n(K=3)']
H_scenarios   = [1.13, 1.01, 1.18]
Rm_scenarios  = [0.76, 0.82, 1.05]

x = np.arange(3); w = 0.32
b1 = ax.bar(x - w/2, H_scenarios, w, color='#9ecae1', edgecolor='white',
            label='$H(\\mu)$', linewidth=0.5)
b2 = ax.bar(x + w/2, Rm_scenarios, w,
            color=[C_FAIL if r<1.0 else C_PASS for r in Rm_scenarios],
            edgecolor='white', label='$R_{mech}$', linewidth=0.5)

ax.axhline(THRESHOLD, color=C_FAIL, linestyle='--', lw=1.0)
for xi, rm in zip(x, Rm_scenarios):
    marker = '✓' if rm >= 1.0 else '✗'
    ax.text(xi+w/2, rm+0.04, marker, ha='center', fontsize=8,
            color=C_PASS if rm >= 1.0 else C_FAIL)

ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=6)
ax.set_ylabel('Nats'); ax.set_ylim(0, 1.4)
ax.legend(frameon=False, fontsize=6)
ax.set_title('Population restriction effect', fontsize=6.5, pad=3)
panel_label(axes4[1], 'b')

# ── Panel c: LDH distributions ───────────────────────────────────────────────
ax = axes4[2]
ldh_x = np.linspace(50, 1200, 300)
ldh_ici   = norm.pdf(ldh_x, 180, 55)
ldh_targ  = norm.pdf(ldh_x, 520, 130)
ldh_sand  = norm.pdf(ldh_x, 310, 90)

ax.fill_between(ldh_x, ldh_ici,  alpha=0.4, color='#1a9641', label='ICI-first optimal')
ax.fill_between(ldh_x, ldh_targ, alpha=0.4, color='#d7191c', label='Targeted-first optimal')
ax.fill_between(ldh_x, ldh_sand, alpha=0.3, color='#fdae61', label='Sandwich strategy')
ax.plot(ldh_x, ldh_ici,  color='#1a9641', lw=1.2)
ax.plot(ldh_x, ldh_targ, color='#d7191c', lw=1.2)
ax.plot(ldh_x, ldh_sand, color='#fdae61', lw=1.2)
ax.axvline(350, color='#555555', linestyle='--', lw=1.0,
           label='Decision boundary\n(350 U/L)')

ax.set_xlabel('Serum LDH (U/L)'); ax.set_ylabel('Probability density')
ax.set_xlim(50, 1000)
ax.legend(frameon=False, fontsize=5.5, loc='upper right')
ax.set_title('LDH mechanistic proxy for treatment selection', fontsize=6.5, pad=3)
panel_label(axes4[2], 'c')

fig4.tight_layout(w_pad=2.0)
fig4.savefig('/home/claude/fig4_braf.pdf', dpi=300)
print("Fig 4 done")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Pembrolizumab
# ═══════════════════════════════════════════════════════════════════════════════
fig5, axes5 = plt.subplots(1, 3, figsize=(NM_W, 2.2))

# ── Panel a: PD-L1 TPS vs response ───────────────────────────────────────────
ax = axes5[0]
np.random.seed(42)
n = 180
tps = np.random.beta(1.2, 2.0, n) * 100
noise = np.random.normal(0, 0.18, n)
response_prob = 0.15 + 0.006*tps + noise
response_prob = np.clip(response_prob, 0.0, 1.0)
responders = np.random.binomial(1, response_prob)

ax.scatter(tps[responders==1], response_prob[responders==1]+np.random.normal(0,0.03,responders.sum()),
           alpha=0.5, s=8, color=C_PASS, label='Responder', zorder=3)
ax.scatter(tps[responders==0], response_prob[responders==0]+np.random.normal(0,0.03,(1-responders).sum()),
           alpha=0.4, s=8, color=C_FAIL, label='Non-responder', zorder=3)

tps_fit = np.linspace(0, 100, 100)
fit_line = 0.15 + 0.006*tps_fit
ax.plot(tps_fit, fit_line, 'k-', lw=1.2, label=f'$R^2\\approx$0.25')

ax.set_xlabel('PD-L1 TPS (%)'); ax.set_ylabel('Predicted response')
ax.legend(frameon=False, fontsize=6, loc='upper left')
ax.set_xlim(0, 100); ax.set_ylim(-0.05, 1.05)
ax.text(50, 0.85, '$R^2 \\approx 0.25$\n$R_{mech} = 0.31$ nats', ha='center',
        fontsize=6.5, color='#444444',
        bbox=dict(boxstyle='round', fc='#ffffcc', ec='#aaaaaa', lw=0.6))
panel_label(axes5[0], 'a')

# ── Panel b: Clearance trajectory ────────────────────────────────────────────
ax = axes5[1]
cycles = np.arange(1, 8)
cl_resp    = 0.190 * np.exp(-0.08 * (cycles-1)) + np.random.normal(0,0.005,7)
cl_nonresp = 0.190 * np.exp( 0.04 * (cycles-1)) + np.random.normal(0,0.007,7)
cl_resp2   = 0.190 * np.exp(-0.10 * (cycles-1)) + np.random.normal(0,0.005,7)
cl_nonresp2= 0.190 * np.exp( 0.02 * (cycles-1)) + np.random.normal(0,0.006,7)

for cl in [cl_resp, cl_resp2]:
    ax.plot(cycles, cl, color=C_PASS, alpha=0.6, lw=1.0)
for cl in [cl_nonresp, cl_nonresp2]:
    ax.plot(cycles, cl, color=C_FAIL, alpha=0.6, lw=1.0)

ax.plot(cycles, 0.190*np.exp(-0.08*(cycles-1)), color=C_PASS, lw=1.5,
        label='Responders (↓CL)', solid_capstyle='round')
ax.plot(cycles, 0.190*np.exp( 0.04*(cycles-1)), color=C_FAIL, lw=1.5,
        label='Non-responders (↑CL)', solid_capstyle='round')
ax.axhline(0.190, color='#aaaaaa', linestyle=':', lw=0.8)

ax.set_xlabel('Treatment cycle'); ax.set_ylabel('Pembrolizumab CL (L/day)')
ax.legend(frameon=False, fontsize=6)
ax.set_xlim(0.5, 7.5); ax.set_ylim(0.10, 0.30)
ax.text(4, 0.28, 'Decreasing CL =\ntumour regression', fontsize=5.5,
        color=C_PASS, ha='center')
panel_label(axes5[1], 'b')

# ── Panel c: K=2 structural limit ────────────────────────────────────────────
ax = axes5[2]
p_vals = np.linspace(0.001, 0.999, 500)
H_binary = -(p_vals*np.log(p_vals) + (1-p_vals)*np.log(1-p_vals))
H_max = np.log(2)

ax.plot(p_vals, H_binary, color=C_BLUE, lw=1.5, label='$H(\\mu)$ for $K=2$')
ax.axhline(THRESHOLD, color=C_FAIL, linestyle='--', lw=1.0, label='1.0 nat threshold')
ax.axhline(H_max, color='#aaaaaa', linestyle=':', lw=0.9,
           label=f'$\\ln 2 \\approx {H_max:.2f}$ nats (max for $K=2$)')
ax.fill_between(p_vals, H_binary, H_max, alpha=0.08, color='#aaaaaa')
ax.fill_between(p_vals, 0, np.minimum(H_binary, H_max), alpha=0.08, color=C_FAIL)

ax.scatter([0.35], [-(0.35*np.log(0.35)+0.65*np.log(0.65))], s=35,
           color=C_FAIL, zorder=5, label='Pembro ($p=0.35$)')
ax.set_xlabel('Responder prevalence $p$')
ax.set_ylabel('$H(\\mu)$ (nats)')
ax.legend(frameon=False, fontsize=5.5, loc='lower center')
ax.set_xlim(0, 1); ax.set_ylim(0, 1.15)
ax.text(0.5, 0.82, '$H(\\mu) < 1.0$ nat\nfor all $K=2$ decisions:\nthreshold structurally\nunreachable',
        ha='center', fontsize=6, color=C_FAIL,
        bbox=dict(boxstyle='round', fc='#fff0f0', ec=C_FAIL, lw=0.5, alpha=0.8))
panel_label(axes5[2], 'c')

fig5.tight_layout(w_pad=2.0)
fig5.savefig('/home/claude/fig5_pembro.pdf', dpi=300)
print("Fig 5 done")

print("\nAll figures saved.")
