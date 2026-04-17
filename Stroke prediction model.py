import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # use 'TkAgg' or remove this line if running interactively
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              precision_score, recall_score,
                              roc_curve, precision_recall_curve,
                              average_precision_score, confusion_matrix,
                              classification_report)
from sklearn.utils.class_weight import compute_class_weight

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = "C:/Users/Jayasooriya/Desktop/DATA7001/Group project/dataset.csv"         # ← change to your CSV path
OUTPUT_DIR  = "E:/DATA7001/"                   # ← folder where PNGs are saved
RANDOM_SEED = 42
TEST_SIZE   = 0.20

# Dark-theme colour palette
DARK_BG  = "#0F1117"
CARD_BG  = "#1A1D2E"
ACCENT1  = "#7C3AED"   # violet
ACCENT2  = "#06B6D4"   # cyan
ACCENT3  = "#10B981"   # emerald
ACCENT4  = "#F59E0B"   # amber
ACCENT5  = "#EF4444"   # red
TEXT_COL = "#E2E8F0"
PALETTE  = [ACCENT1, ACCENT2, ACCENT3, ACCENT4, ACCENT5,
            "#A78BFA", "#34D399", "#FCD34D"]

plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": CARD_BG,
    "axes.edgecolor": "#2D3748", "axes.labelcolor": TEXT_COL,
    "xtick.color": TEXT_COL,    "ytick.color": TEXT_COL,
    "text.color": TEXT_COL,     "grid.color": "#2D3748",
    "grid.linestyle": "--",     "grid.alpha": 0.5,
    "font.size": 11,            "axes.titlesize": 13,
    "axes.titleweight": "bold", "legend.facecolor": CARD_BG,
    "legend.edgecolor": "#2D3748",
})


def save_fig(filename, fig=None, dpi=150):
    """Save figure and close it."""
    f = fig or plt.gcf()
    path = OUTPUT_DIR + filename
    f.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(f)
    print(f"  ✓  Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 – LOAD & CLEAN
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 1 · Loading & Cleaning")
print("═"*60)

df = pd.read_csv(DATA_PATH)
df.drop('id', axis=1, inplace=True)

# Remove the rare 'Other' gender category
df = df[df['gender'] != 'Other'].reset_index(drop=True)

# Impute BMI with per-gender median
df['bmi'] = df.groupby('gender')['bmi'].transform(
    lambda x: x.fillna(x.median()))

# Impute smoking_status with the overall mode
df['smoking_status'] = df['smoking_status'].fillna(
    df['smoking_status'].mode()[0])

print(f"  Rows        : {len(df):,}")
print(f"  Stroke rate : {df['stroke'].mean()*100:.2f}%")
print(f"  Missing     : {df.isnull().sum().sum()}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 – EXPLORATORY DATA ANALYSIS  →  fig1_eda.png
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 2 · EDA Plots")
print("═"*60)

fig = plt.figure(figsize=(18, 14), facecolor=DARK_BG)
fig.suptitle("Stroke Risk Dataset – Exploratory Data Analysis",
             fontsize=17, fontweight='bold', color=TEXT_COL, y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── 2a. Class imbalance ──────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
counts = df['stroke'].value_counts()
bars = ax.bar(['No Stroke', 'Stroke'], counts.values,
              color=[ACCENT3, ACCENT5], edgecolor='white', linewidth=0.6)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f'{val:,}\n({val/len(df)*100:.1f}%)',
            ha='center', va='bottom', fontsize=10)
ax.set_title("Class Distribution")
ax.set_ylabel("Count")
ax.set_ylim(0, counts.max() * 1.2)

# ── 2b. Age distribution ─────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
for val, color, label in [(0, ACCENT2, 'No Stroke'), (1, ACCENT5, 'Stroke')]:
    ax.hist(df[df['stroke'] == val]['age'], bins=30,
            alpha=0.7, color=color, label=label, edgecolor='none')
ax.set_title("Age Distribution by Stroke")
ax.set_xlabel("Age")
ax.legend()

# ── 2c. Glucose distribution ─────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
for val, color, label in [(0, ACCENT2, 'No Stroke'), (1, ACCENT5, 'Stroke')]:
    ax.hist(df[df['stroke'] == val]['avg_glucose_level'], bins=30,
            alpha=0.7, color=color, label=label, edgecolor='none')
ax.set_title("Avg Glucose Level by Stroke")
ax.set_xlabel("Glucose Level")
ax.legend()

# ── 2d. BMI distribution ─────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
for val, color, label in [(0, ACCENT2, 'No Stroke'), (1, ACCENT5, 'Stroke')]:
    ax.hist(df[df['stroke'] == val]['bmi'], bins=30,
            alpha=0.7, color=color, label=label, edgecolor='none')
ax.set_title("BMI Distribution by Stroke")
ax.set_xlabel("BMI")
ax.legend()

# ── 2e. Stroke rate by gender ────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
gdata = df.groupby('gender')['stroke'].mean().reset_index()
bars = ax.bar(gdata['gender'], gdata['stroke'] * 100,
              color=PALETTE[:2], edgecolor='white', linewidth=0.6)
for bar, v in zip(bars, gdata['stroke'] * 100):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.1, f'{v:.2f}%', ha='center')
ax.set_title("Stroke Rate by Gender")
ax.set_ylabel("Stroke Rate (%)")

# ── 2f. Stroke rate by smoking ───────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
sdata = (df.groupby('smoking_status')['stroke'].mean()
           .reset_index().sort_values('stroke'))
bars = ax.barh(sdata['smoking_status'], sdata['stroke'] * 100,
               color=PALETTE[:len(sdata)], edgecolor='white', linewidth=0.6)
for bar, v in zip(bars, sdata['stroke'] * 100):
    ax.text(bar.get_width() + 0.05,
            bar.get_y() + bar.get_height()/2,
            f'{v:.2f}%', va='center', fontsize=9)
ax.set_title("Stroke Rate by Smoking Status")
ax.set_xlabel("Stroke Rate (%)")

# ── 2g. Risk factors ─────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 0])
cats  = ['No Hypertension', 'Hypertension', 'No Heart Dis.', 'Heart Disease']
rates = [
    df[df['hypertension']  == 0]['stroke'].mean() * 100,
    df[df['hypertension']  == 1]['stroke'].mean() * 100,
    df[df['heart_disease'] == 0]['stroke'].mean() * 100,
    df[df['heart_disease'] == 1]['stroke'].mean() * 100,
]
bars = ax.bar(cats, rates,
              color=[ACCENT3, ACCENT5, ACCENT2, ACCENT4],
              edgecolor='white', linewidth=0.6)
for bar, v in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.05, f'{v:.2f}%', ha='center', fontsize=9)
ax.set_title("Risk Factors vs Stroke Rate")
ax.set_ylabel("Stroke Rate (%)")
plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right', fontsize=9)

# ── 2h. Work type ────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 1])
wdata = (df.groupby('work_type')['stroke'].mean()
           .reset_index().sort_values('stroke'))
bars = ax.barh(wdata['work_type'], wdata['stroke'] * 100,
               color=PALETTE[:len(wdata)], edgecolor='white', linewidth=0.6)
for bar, v in zip(bars, wdata['stroke'] * 100):
    ax.text(bar.get_width() + 0.05,
            bar.get_y() + bar.get_height()/2,
            f'{v:.2f}%', va='center', fontsize=9)
ax.set_title("Stroke Rate by Work Type")
ax.set_xlabel("Stroke Rate (%)")

# ── 2i. Correlation heatmap ──────────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 2])
num_df = df[['age', 'hypertension', 'heart_disease',
             'avg_glucose_level', 'bmi', 'stroke']]
corr = num_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=ax,
            cmap=sns.diverging_palette(240, 10, as_cmap=True),
            annot=True, fmt='.2f', linewidths=0.5, linecolor='#2D3748',
            square=True, annot_kws={'size': 8}, cbar_kws={'shrink': 0.7})
ax.set_title("Feature Correlation Heatmap")

save_fig("fig1_eda.png", fig)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 – PREPROCESSING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 3 · Preprocessing & Feature Engineering")
print("═"*60)

df_model = df.copy()

# Label-encode binary/ordinal columns
for col in ['gender', 'ever_married', 'Residence_type']:
    df_model[col] = LabelEncoder().fit_transform(df_model[col])

# One-hot encode multi-category columns
df_model = pd.get_dummies(df_model,
                           columns=['work_type', 'smoking_status'],
                           drop_first=True)

X = df_model.drop('stroke', axis=1)
y = df_model['stroke']

# ── Interaction / risk features ───────────────────────────────────────────────
X['age_glucose'] = X['age'] * X['avg_glucose_level']
X['age_bmi']     = X['age'] * X['bmi']
X['glucose_bmi'] = X['avg_glucose_level'] * X['bmi']
X['risk_score']  = (X['hypertension']
                    + X['heart_disease']
                    + (X['age'] > 60).astype(int)
                    + (X['avg_glucose_level'] > 200).astype(int))

feat_names = list(X.columns)
print(f"  Total features after engineering : {len(feat_names)}")

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)

# ── Scale (required for LR, KNN, SVM) ────────────────────────────────────────
scaler   = StandardScaler()
Xs_train = scaler.fit_transform(X_train)
Xs_test  = scaler.transform(X_test)

# ── Class weights to handle severe imbalance ──────────────────────────────────
cw  = compute_class_weight('balanced',
                            classes=np.array([0, 1]), y=y_train)
cwd = {0: cw[0], 1: cw[1]}
print(f"  Class weights : {cwd}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 – TRAIN MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 4 · Training Models")
print("═"*60)

MODELS = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, class_weight=cwd,
                           C=0.1, random_state=RANDOM_SEED),
        True   # needs scaled data
    ),
    "Decision Tree": (
        DecisionTreeClassifier(max_depth=6, class_weight=cwd,
                               random_state=RANDOM_SEED),
        False
    ),
    "Random Forest": (
        RandomForestClassifier(n_estimators=100, max_depth=8,
                               class_weight=cwd,
                               random_state=RANDOM_SEED, n_jobs=-1),
        False
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                   learning_rate=0.1, subsample=0.8,
                                   random_state=RANDOM_SEED),
        False
    ),
    "KNN": (
        KNeighborsClassifier(n_neighbors=7),
        True
    ),
    "SVM": (
        SVC(probability=True, class_weight=cwd, C=1.0,
            kernel='rbf', random_state=RANDOM_SEED),
        True
    ),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

for name, (model, use_scaled) in MODELS.items():
    Xtr = Xs_train if use_scaled else X_train.values
    Xte = Xs_test  if use_scaled else X_test.values

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]

    cv_scores = cross_val_score(model, Xtr, y_train,
                                cv=cv, scoring='roc_auc', n_jobs=-1)

    results[name] = {
        'model':      model,
        'y_pred':     y_pred,
        'y_prob':     y_prob,
        'accuracy':   accuracy_score(y_test, y_pred),
        'roc_auc':    roc_auc_score(y_test, y_prob),
        'f1':         f1_score(y_test, y_pred),
        'precision':  precision_score(y_test, y_pred, zero_division=0),
        'recall':     recall_score(y_test, y_pred),
        'cv_auc':     cv_scores.mean(),
        'cv_std':     cv_scores.std(),
    }
    r = results[name]
    print(f"  {name:<22}  ACC={r['accuracy']:.4f}  "
          f"AUC={r['roc_auc']:.4f}  F1={r['f1']:.4f}  "
          f"Recall={r['recall']:.4f}")

# ── Soft-voting ensemble of the top-3 models by AUC ──────────────────────────
top3 = sorted(results, key=lambda k: results[k]['roc_auc'], reverse=True)[:3]
print(f"\n  Building Ensemble from : {top3}")

ensemble = VotingClassifier(
    estimators=[(n, results[n]['model']) for n in top3],
    voting='soft',
    weights=[3, 2, 1]
)
ensemble.fit(Xs_train, y_train)          # scaled safe for all tree models too
ep  = ensemble.predict(Xs_test)
epr = ensemble.predict_proba(Xs_test)[:, 1]

results['Ensemble (Top-3)'] = {
    'y_pred':    ep,    'y_prob':   epr,
    'accuracy':  accuracy_score(y_test, ep),
    'roc_auc':   roc_auc_score(y_test, epr),
    'f1':        f1_score(y_test, ep),
    'precision': precision_score(y_test, ep, zero_division=0),
    'recall':    recall_score(y_test, ep),
    'cv_auc': 0, 'cv_std': 0,
}
r = results['Ensemble (Top-3)']
print(f"  {'Ensemble (Top-3)':<22}  ACC={r['accuracy']:.4f}  "
      f"AUC={r['roc_auc']:.4f}  F1={r['f1']:.4f}  Recall={r['recall']:.4f}")

best_name = max(results, key=lambda k: results[k]['roc_auc'])
best      = results[best_name]
print(f"\n  ★  Best model : {best_name}  (AUC = {best['roc_auc']:.4f})")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 – FIGURE 2 · Model Performance Comparison
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 5 · Plotting Model Comparison")
print("═"*60)

names    = list(results.keys())
metrics  = ['accuracy', 'roc_auc', 'f1', 'precision', 'recall']
m_labels = ['Accuracy', 'ROC-AUC', 'F1', 'Precision', 'Recall']
colors   = [ACCENT1, ACCENT2, ACCENT3, ACCENT4, ACCENT5]

fig, axes = plt.subplots(1, 5, figsize=(22, 6), facecolor=DARK_BG)
fig.suptitle("Model Performance Comparison",
             fontsize=15, fontweight='bold', color=TEXT_COL, y=1.01)

for ax, metric, label, color in zip(axes, metrics, m_labels, colors):
    vals = [results[n][metric] for n in names]
    bars = ax.bar(range(len(names)), vals, color=color,
                  alpha=0.85, edgecolor='white', linewidth=0.5)
    best_idx = int(np.argmax(vals))
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(2.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.003,
                f'{v:.3f}', ha='center', fontsize=7.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=7)
    ax.set_ylim(0, 1.15)
    ax.set_title(label)

fig.tight_layout()
save_fig("fig2_model_comparison.png", fig)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 – FIGURE 3 · ROC & Precision-Recall Curves
# ══════════════════════════════════════════════════════════════════════════════
print("  STEP 6 · ROC & PR Curves")

fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=DARK_BG)
fig.suptitle("ROC & Precision-Recall Curves",
             fontsize=15, fontweight='bold', color=TEXT_COL)

ax = axes[0]
ax.plot([0, 1], [0, 1], '--', color='#4A5568', lw=1, label='Random')
for (name, res), color in zip(results.items(), PALETTE):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax.plot(fpr, tpr, lw=1.8, color=color,
            label=f"{name} (AUC={res['roc_auc']:.3f})")
ax.set_title("ROC Curve")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=8, loc='lower right')

ax = axes[1]
for (name, res), color in zip(results.items(), PALETTE):
    prec, rec, _ = precision_recall_curve(y_test, res['y_prob'])
    ap = average_precision_score(y_test, res['y_prob'])
    ax.plot(rec, prec, lw=1.8, color=color,
            label=f"{name} (AP={ap:.3f})")
ax.set_title("Precision-Recall Curve")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.legend(fontsize=8, loc='upper right')

for ax in axes:
    ax.set_facecolor(CARD_BG)
    ax.grid(True, alpha=0.3)

fig.tight_layout()
save_fig("fig3_roc_pr.png", fig)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7 – FIGURE 4 · Confusion Matrices
# ══════════════════════════════════════════════════════════════════════════════
print("  STEP 7 · Confusion Matrices")

n_models = len(results)
ncols    = 4
nrows    = (n_models + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols,
                          figsize=(20, nrows * 5), facecolor=DARK_BG)
fig.suptitle("Confusion Matrices – All Models",
             fontsize=15, fontweight='bold', color=TEXT_COL)
axes_flat = axes.flatten() if nrows > 1 else list(axes)

for idx, (name, res) in enumerate(results.items()):
    ax = axes_flat[idx]
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                cmap=sns.light_palette(ACCENT1, as_cmap=True),
                linewidths=1, linecolor='#2D3748',
                xticklabels=['No Stroke', 'Stroke'],
                yticklabels=['No Stroke', 'Stroke'],
                cbar=False)
    ax.set_title(f"{name}\nACC={res['accuracy']:.3f}  F1={res['f1']:.3f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

for idx in range(n_models, len(axes_flat)):
    axes_flat[idx].set_visible(False)

fig.tight_layout()
save_fig("fig4_confusion_matrices.png", fig)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 8 – FIGURE 5 · Feature Importance (Random Forest)
# ══════════════════════════════════════════════════════════════════════════════
print("  STEP 8 · Feature Importance")

rf_model   = results['Random Forest']['model']
feat_imp   = rf_model.feature_importances_
top_idx    = np.argsort(feat_imp)[-18:]
top_names  = [feat_names[i] for i in top_idx]
top_vals   = feat_imp[top_idx]

fig, ax = plt.subplots(figsize=(12, 8), facecolor=DARK_BG)
cvals      = np.linspace(0.3, 1.0, len(top_vals))
bar_colors = [plt.cm.plasma(v) for v in cvals]
bars = ax.barh(top_names, top_vals, color=bar_colors, edgecolor='none')
for bar, v in zip(bars, top_vals):
    ax.text(bar.get_width() + 0.0005,
            bar.get_y() + bar.get_height()/2,
            f'{v:.4f}', va='center', fontsize=9)
ax.set_title("Feature Importance (Random Forest) – Top 18", fontsize=14)
ax.set_xlabel("Importance Score")
fig.tight_layout()
save_fig("fig5_feature_importance.png", fig)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 9 – FIGURE 6 · Grouped Metrics Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
print("  STEP 9 · Summary Grouped Bar Chart")

aucs = [results[n]['roc_auc'] for n in names]
accs = [results[n]['accuracy'] for n in names]
f1s  = [results[n]['f1']       for n in names]

x = np.arange(len(names))
w = 0.25

fig, ax = plt.subplots(figsize=(14, 7), facecolor=DARK_BG)
b1 = ax.bar(x - w, accs, w, label='Accuracy',  color=ACCENT2, alpha=0.85, edgecolor='white', linewidth=0.5)
b2 = ax.bar(x,     aucs, w, label='ROC-AUC',   color=ACCENT1, alpha=0.85, edgecolor='white', linewidth=0.5)
b3 = ax.bar(x + w, f1s,  w, label='F1 Score',  color=ACCENT3, alpha=0.85, edgecolor='white', linewidth=0.5)

for group_bars in [b1, b2, b3]:
    for bar in group_bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2,
                h + 0.005, f'{h:.3f}', ha='center', fontsize=7.5)

ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9)
ax.set_ylim(0, 1.18)
ax.set_ylabel("Score")
ax.set_title("All Models – Accuracy / ROC-AUC / F1 Comparison")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
save_fig("fig6_summary_metrics.png", fig)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 10 – FIGURE 7 · Classification Report Heatmap (Best Model)
# ══════════════════════════════════════════════════════════════════════════════
print("  STEP 10 · Classification Report Heatmap")

rep    = classification_report(y_test, best['y_pred'],
                                target_names=['No Stroke', 'Stroke'],
                                output_dict=True)
rep_df = pd.DataFrame(rep).T.drop('accuracy', errors='ignore')

fig, ax = plt.subplots(figsize=(9, 5), facecolor=DARK_BG)
sns.heatmap(rep_df.astype(float), annot=True, fmt='.3f', ax=ax,
            cmap=sns.light_palette(ACCENT2, as_cmap=True),
            linewidths=0.5, linecolor='#2D3748', cbar=False)
ax.set_title(f"Classification Report – {best_name}", fontsize=13)
fig.tight_layout()
save_fig("fig7_classification_report.png", fig)


# ══════════════════════════════════════════════════════════════════════════════
#  FINAL SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*68)
print(f"  FINAL RESULTS  (ranked by ROC-AUC)   ★ = {best_name}")
print("═"*68)
header = (f"{'Model':<24}  {'Accuracy':>9}  {'ROC-AUC':>9}"
          f"  {'F1':>7}  {'Recall':>8}  {'Precision':>10}")
print(header)
print("─"*68)
for name, res in sorted(results.items(),
                         key=lambda x: x[1]['roc_auc'],
                         reverse=True):
    star = "★" if name == best_name else " "
    print(f"{star} {name:<23}  {res['accuracy']:9.4f}  {res['roc_auc']:9.4f}"
          f"  {res['f1']:7.4f}  {res['recall']:8.4f}  {res['precision']:10.4f}")
print("═"*68)
print("\n  All 7 figures saved to:", OUTPUT_DIR)
print("  Done ✓\n")