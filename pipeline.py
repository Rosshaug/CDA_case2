import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.model_selection import LeaveOneGroupOut
import warnings
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


def bootstrap_ci(values, n_boot=2000, ci=0.95, seed=42):
    """Bootstrap CI on the mean of `values` (list/array of per-fold results)."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    boot_means = [np.mean(rng.choice(values, size=len(values), replace=True))
                  for _ in range(n_boot)]
    lo = np.percentile(boot_means, 100 * (1 - ci) / 2)
    hi = np.percentile(boot_means, 100 * (1 - (1 - ci) / 2))
    return lo, hi


def cluster_bootstrap_corr_ci(subjects, flags, vals, n_boot=2000, ci=0.95, seed=42):
    """Cluster bootstrap CI on point-biserial r.

    Resamples *subjects* (the natural independent unit) rather than individual
    observations.  Each resampled subject contributes all its observations, so
    within-subject correlation is preserved and the CI correctly reflects
    uncertainty at the subject level.

    Returns (nan, nan) when fewer than 50% of resamples yield a finite r
    (can occur with severely imbalanced anomaly flags).
    """
    rng = np.random.default_rng(seed)
    subjects = np.asarray(subjects)
    flags    = np.asarray(flags)
    vals     = np.asarray(vals)
    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)

    boot_corrs = []
    for _ in range(n_boot):
        boot_subjs = rng.choice(unique_subjects, size=n_subjects, replace=True)
        b_flags, b_vals = [], []
        for s in boot_subjs:
            mask = subjects == s
            b_flags.append(flags[mask])
            b_vals.append(vals[mask])
        b_flags = np.concatenate(b_flags)
        b_vals  = np.concatenate(b_vals)
        with warnings.catch_warnings(), np.errstate(invalid='ignore'):
            warnings.simplefilter("ignore")
            r, _ = stats.pointbiserialr(b_flags, b_vals)
        if np.isfinite(r):
            boot_corrs.append(r)

    if len(boot_corrs) < n_boot * 0.5:
        return np.nan, np.nan
    lo = np.percentile(boot_corrs, 100 * (1 - ci) / 2)
    hi = np.percentile(boot_corrs, 100 * (1 - (1 - ci) / 2))
    return lo, hi


data_path = os.path.join("data", 'HR_data_2.csv')
df = pd.read_csv(data_path)

df = df.drop(columns=["Unnamed: 0", "Round", "Puzzler", "original_ID", "raw_data_path", "Team_ID", "Cohort"])

quastinare_columns = ["Frustrated", "upset", "hostile", "alert", "ashamed", "inspired",
                      "nervous", "attentive", "afraid", "active", "determined"]
data_columns = [col for col in df.columns if col not in quastinare_columns + ["Phase"] + ["Individual"]]

df = df.dropna(subset=data_columns)

chart_data = []       # dicts with mean + CI per (model, phase)
all_corr_dfs = {}
results_phase1_phase3 = None

for experiment in [["phase1", "phase3"], ["phase1"]]:
    exp_label = "_".join(experiment)

    df_experiment = df.copy()
    scaled_data = []

    for subject_id, subject_data in df_experiment.groupby('Individual'):
        resting_data = subject_data[subject_data['Phase'].isin(experiment)]
        scaler = StandardScaler()
        scaler.fit(resting_data[data_columns])
        subject_scaled = subject_data.copy()
        subject_scaled[data_columns] = scaler.transform(subject_data[data_columns])
        scaled_data.append(subject_scaled)

    df_scaled = pd.concat(scaled_data, ignore_index=True)

    if "phase3" in experiment:
        df_scaled_p1p3 = df_scaled.copy()

    logo = LeaveOneGroupOut()
    groups = df_scaled['Individual']

    results = []
    all_test_evaluations = []

    for fold_idx, (train_index, test_index) in enumerate(logo.split(df_scaled, groups=groups)):
        train_df = df_scaled.iloc[train_index]
        test_df = df_scaled.iloc[test_index]
        test_subject = test_df['Individual'].iloc[0]

        train_rest = train_df[train_df['Phase'].isin(experiment)][data_columns]
        test_phase1 = test_df[test_df['Phase'] == "phase1"][data_columns]
        test_phase2 = test_df[test_df['Phase'] == "phase2"][data_columns]
        test_phase3 = test_df[test_df['Phase'] == "phase3"][data_columns]

        pca = PCA(n_components=0.95)
        pca.fit(train_rest)
        train_rest_pca = pca.transform(train_rest)
        test_phase1_pca = pca.transform(test_phase1)
        test_phase2_pca = pca.transform(test_phase2)
        test_phase3_pca = pca.transform(test_phase3)

        ocsvm = OneClassSVM(kernel='rbf', nu=0.1, gamma='scale')
        ocsvm.fit(train_rest_pca)

        predictions_p1 = ocsvm.predict(test_phase1_pca)
        predictions_p2 = ocsvm.predict(test_phase2_pca)
        predictions_p3 = ocsvm.predict(test_phase3_pca)

        rate_p1 = (np.sum(predictions_p1 == -1) / len(predictions_p1)) * 100 if len(predictions_p1) > 0 else 0
        rate_p2 = (np.sum(predictions_p2 == -1) / len(predictions_p2)) * 100 if len(predictions_p2) > 0 else 0
        rate_p3 = (np.sum(predictions_p3 == -1) / len(predictions_p3)) * 100 if len(predictions_p3) > 0 else 0

        results.append({
            'Test_Subject': test_subject,
            'Total_Puzzle_Samples': len(predictions_p2),
            'Anomalies_Flagged': np.sum(predictions_p2 == -1),
            'Rate_p1%': rate_p1,
            'Rate_p2%': rate_p2,
            'Rate_p3%': rate_p3,
            'PCA_Components_Used': pca.n_components_
        })

        test_q_data = test_df[test_df['Phase'] == "phase2"][quastinare_columns].copy()
        test_q_data['Anomaly_Flag'] = (predictions_p2 == -1).astype(int)
        test_q_data['Subject'] = test_subject
        all_test_evaluations.append(test_q_data)

    results_df = pd.DataFrame(results)
    loso_filename = os.path.join("output", f"loso_metrics_{exp_label}.csv")
    results_df.describe().to_csv(loso_filename)

    model_name = "Trained on P1 & P3" if "phase3" in experiment else "Trained on P1 Only"

    for phase_col, phase_label in [('Rate_p1%', 'Phase 1 (Rest)'),
                                    ('Rate_p2%', 'Phase 2 (Puzzle)'),
                                    ('Rate_p3%', 'Phase 3 (Post-Rest)')]:
        arr = results_df[phase_col].values
        ci_lo, ci_hi = bootstrap_ci(arr)
        chart_data.append({
            'Phase': phase_label,
            'Anomaly Rate (%)': arr.mean(),
            'CI_lo': ci_lo,
            'CI_hi': ci_hi,
            'Model': model_name,
        })

    eval_df = pd.concat(all_test_evaluations, ignore_index=True)
    correlation_results = []

    for emotion in quastinare_columns:
        clean_df = eval_df.dropna(subset=[emotion, 'Anomaly_Flag'])
        if len(clean_df) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", stats.ConstantInputWarning)
                corr, p_value = stats.pointbiserialr(clean_df['Anomaly_Flag'], clean_df[emotion])
            # Bootstrap CI is unreliable when the anomaly flag is severely imbalanced
            # (Exp 2 has 96% positives → biased bootstrap CIs that may not span zero
            # even for non-significant correlations). Only report CI for Exp 1.
            # Cluster bootstrap: resample subjects, not rows, to respect
            # within-subject correlation (4 rounds per subject).
            # Exp 2 (96% positive flag) → degenerate bootstrap; report n/a.
            if "phase3" in experiment:
                ci_lo, ci_hi = cluster_bootstrap_corr_ci(
                    clean_df['Subject'].values,
                    clean_df['Anomaly_Flag'].values,
                    clean_df[emotion].values)
            else:
                ci_lo, ci_hi = np.nan, np.nan
            correlation_results.append({
                'Emotion': emotion,
                'Correlation': round(corr, 3),
                'CI_lo': round(ci_lo, 3),
                'CI_hi': round(ci_hi, 3),
                'p_value': round(p_value, 3),
                'Significant': "Yes" if p_value < 0.05 else "No"
            })

    corr_df = pd.DataFrame(correlation_results)
    corr_filename = os.path.join("output", f"correlations_{exp_label}.csv")
    corr_df.to_csv(corr_filename, index=False)

    all_corr_dfs[exp_label] = corr_df.copy()
    if "phase3" in experiment:
        results_phase1_phase3 = results_df.copy()


# ── Nested LOSO: tune nu per outer fold (Experiment 1 baseline only) ──────────
nu_candidates = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
baseline_phases = ["phase1", "phase3"]

logo_nested = LeaveOneGroupOut()
groups_nested = df_scaled_p1p3['Individual']

nested_results = []
all_outer_nu_means = {nu: [] for nu in nu_candidates}

print("Running nested LOSO for nu tuning...")
for outer_idx, (outer_train_idx, outer_test_idx) in enumerate(
        logo_nested.split(df_scaled_p1p3, groups=groups_nested)):
    train_df = df_scaled_p1p3.iloc[outer_train_idx]
    test_df  = df_scaled_p1p3.iloc[outer_test_idx]
    test_subject = test_df['Individual'].iloc[0]

    # Inner LOSO over the 25 training subjects to select best nu
    inner_logo   = LeaveOneGroupOut()
    inner_groups = train_df['Individual']
    nu_scores    = {nu: [] for nu in nu_candidates}

    for inner_train_idx, inner_test_idx in inner_logo.split(train_df, groups=inner_groups):
        inner_train = train_df.iloc[inner_train_idx]
        inner_test  = train_df.iloc[inner_test_idx]

        inner_train_rest = inner_train[inner_train['Phase'].isin(baseline_phases)][data_columns]
        inner_test_p1    = inner_test[inner_test['Phase'] == "phase1"][data_columns]
        inner_test_p2    = inner_test[inner_test['Phase'] == "phase2"][data_columns]
        inner_test_p3    = inner_test[inner_test['Phase'] == "phase3"][data_columns]
        if len(inner_train_rest) == 0 or len(inner_test_p2) == 0:
            continue

        pca_inner = PCA(n_components=0.95)
        pca_inner.fit(inner_train_rest)
        inner_train_rest_pca = pca_inner.transform(inner_train_rest)
        inner_test_p1_pca    = pca_inner.transform(inner_test_p1) if len(inner_test_p1) > 0 else None
        inner_test_p2_pca    = pca_inner.transform(inner_test_p2)
        inner_test_p3_pca    = pca_inner.transform(inner_test_p3) if len(inner_test_p3) > 0 else None

        for nu in nu_candidates:
            ocsvm_inner = OneClassSVM(kernel='rbf', nu=nu, gamma='scale')
            ocsvm_inner.fit(inner_train_rest_pca)
            r_p2 = (np.sum(ocsvm_inner.predict(inner_test_p2_pca) == -1) / len(inner_test_p2_pca)) * 100
            r_p1 = (np.sum(ocsvm_inner.predict(inner_test_p1_pca) == -1) / len(inner_test_p1_pca)) * 100 \
                   if inner_test_p1_pca is not None else 0.0
            r_p3 = (np.sum(ocsvm_inner.predict(inner_test_p3_pca) == -1) / len(inner_test_p3_pca)) * 100 \
                   if inner_test_p3_pca is not None else 0.0
            # Discrimination score: reward Phase 2 detection, penalise false positives in rest phases
            score = r_p2 - (r_p1 + r_p3) / 2
            nu_scores[nu].append(score)

    nu_mean_rates = {nu: np.mean(nu_scores[nu]) if nu_scores[nu] else 0.0
                     for nu in nu_candidates}
    nu_std_rates  = {nu: np.std(nu_scores[nu],  ddof=1) if len(nu_scores[nu]) > 1 else 0.0
                     for nu in nu_candidates}

    # 1-SE rule: find the nu with the best mean score, compute its SE,
    # then pick the smallest nu (tightest boundary) within 1 SE of that best.
    best_nu_raw = max(nu_candidates, key=lambda nu: nu_mean_rates[nu])
    n_inner     = len(nu_scores[best_nu_raw])
    se_best     = nu_std_rates[best_nu_raw] / np.sqrt(n_inner) if n_inner > 0 else 0.0
    threshold   = nu_mean_rates[best_nu_raw] - se_best
    best_nu     = min(nu for nu in nu_candidates if nu_mean_rates[nu] >= threshold)

    for nu in nu_candidates:
        all_outer_nu_means[nu].append(nu_mean_rates[nu])

    # Outer fold evaluation with selected best_nu
    train_rest  = train_df[train_df['Phase'].isin(baseline_phases)][data_columns]
    test_p1     = test_df[test_df['Phase'] == "phase1"][data_columns]
    test_p2     = test_df[test_df['Phase'] == "phase2"][data_columns]
    test_p3     = test_df[test_df['Phase'] == "phase3"][data_columns]

    pca_outer = PCA(n_components=0.95)
    pca_outer.fit(train_rest)
    train_rest_pca = pca_outer.transform(train_rest)
    test_p1_pca    = pca_outer.transform(test_p1)
    test_p2_pca    = pca_outer.transform(test_p2)
    test_p3_pca    = pca_outer.transform(test_p3)

    ocsvm_outer = OneClassSVM(kernel='rbf', nu=best_nu, gamma='scale')
    ocsvm_outer.fit(train_rest_pca)
    preds_p1 = ocsvm_outer.predict(test_p1_pca)
    preds_p2 = ocsvm_outer.predict(test_p2_pca)
    preds_p3 = ocsvm_outer.predict(test_p3_pca)

    rate_p1 = (np.sum(preds_p1 == -1) / len(preds_p1)) * 100 if len(preds_p1) > 0 else 0
    rate_p2 = (np.sum(preds_p2 == -1) / len(preds_p2)) * 100 if len(preds_p2) > 0 else 0
    rate_p3 = (np.sum(preds_p3 == -1) / len(preds_p3)) * 100 if len(preds_p3) > 0 else 0

    row = {'Test_Subject': test_subject, 'Best_nu': best_nu,
           'Rate_p1%': rate_p1, 'Rate_p2%': rate_p2, 'Rate_p3%': rate_p3}
    for nu in nu_candidates:
        row[f'inner_mean_p2_nu{nu}'] = round(nu_mean_rates[nu], 2)
    nested_results.append(row)
    print(f"  Fold {outer_idx+1:2d}/26 — best nu={best_nu:.2f}  "
          f"P1={rate_p1:.0f}%  P2={rate_p2:.0f}%  P3={rate_p3:.0f}%")

nested_df = pd.DataFrame(nested_results)
nested_df.to_csv(os.path.join("output", "nested_cv_nu_results.csv"), index=False)
print(f"\nNested CV results (n=26 folds):")
print(f"  Phase 1: {nested_df['Rate_p1%'].mean():.1f} ± {nested_df['Rate_p1%'].std():.1f}%")
print(f"  Phase 2: {nested_df['Rate_p2%'].mean():.1f} ± {nested_df['Rate_p2%'].std():.1f}%")
print(f"  Phase 3: {nested_df['Rate_p3%'].mean():.1f} ± {nested_df['Rate_p3%'].std():.1f}%")
print(f"  Selected nu distribution:\n{nested_df['Best_nu'].value_counts().sort_index()}")


# Figure 5: nu selection — inner Phase 2 rate vs nu + nu histogram
means_by_nu = [np.mean(all_outer_nu_means[nu]) for nu in nu_candidates]
stds_by_nu  = [np.std(all_outer_nu_means[nu])  for nu in nu_candidates]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(nu_candidates, means_by_nu, 'o-', color='#1f77b4', linewidth=2, markersize=7)
axes[0].fill_between(nu_candidates,
                     np.array(means_by_nu) - np.array(stds_by_nu),
                     np.array(means_by_nu) + np.array(stds_by_nu),
                     alpha=0.2, color='#1f77b4')
axes[0].axvline(0.1, color='red', linestyle='--', linewidth=1.2, label='nu=0.1 (original)')
axes[0].set_xlabel('nu', fontsize=12)
axes[0].set_ylabel('Mean inner-fold discrimination score (%)', fontsize=12)
axes[0].set_title('Discrimination Score vs nu\n(Phase 2 rate − mean Phase 1+3 rate; mean ± SD across 26 outer folds)', fontsize=10)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

nu_counts = nested_df['Best_nu'].value_counts().reindex(nu_candidates, fill_value=0)
axes[1].bar([str(nu) for nu in nu_candidates], nu_counts.values,
            color='#1f77b4', edgecolor='black')
axes[1].set_xlabel('Selected nu', fontsize=12)
axes[1].set_ylabel('Number of outer folds', fontsize=12)
axes[1].set_title('nu Selected by Inner LOSO\n(across 26 outer folds)', fontsize=11)
axes[1].grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join("figs", "nu_selection.png"), dpi=300)
plt.show()


# Figure 1: Bar chart with 95% bootstrap CI error bars
plot_df = pd.DataFrame(chart_data)
phases = ['Phase 1 (Rest)', 'Phase 2 (Puzzle)', 'Phase 3 (Post-Rest)']
models = ['Trained on P1 & P3', 'Trained on P1 Only']
palette = {'Trained on P1 & P3': '#1f77b4', 'Trained on P1 Only': '#ff7f0e'}

sns.set_theme(style="whitegrid")
x = np.arange(len(phases))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))

for i, model in enumerate(models):
    sub = plot_df[plot_df['Model'] == model].set_index('Phase')
    means = [sub.loc[p, 'Anomaly Rate (%)'] for p in phases]
    ci_lo = [sub.loc[p, 'CI_lo'] for p in phases]
    ci_hi = [sub.loc[p, 'CI_hi'] for p in phases]
    yerr  = [[m - lo for m, lo in zip(means, ci_lo)],
              [hi - m for m, hi in zip(means, ci_hi)]]
    offset = (i - 0.5) * width
    ax.bar(x + offset, means, width, label=model, color=palette[model],
           yerr=yerr, capsize=5, error_kw={'linewidth': 1.5, 'ecolor': 'black'})

ax.set_xticks(x)
ax.set_xticklabels(phases, fontsize=11)
ax.set_ylabel('Average Anomaly Rate (%)', fontsize=12)
ax.set_xlabel('Phase', fontsize=12)
ax.set_ylim(0, 110)
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join("figs", "carry_over_effect_chart.png"), dpi=300)
plt.show()


# Figure 2: Correlation heatmap (r + significance star)
emotions_order = ["alert", "attentive", "active", "determined", "inspired",
                  "Frustrated", "upset", "hostile", "nervous", "ashamed", "afraid"]

p1p3 = all_corr_dfs['phase1_phase3'].set_index('Emotion')
p1   = all_corr_dfs['phase1'].set_index('Emotion')

corr_matrix = pd.DataFrame({
    'P1 & P3 Baseline': p1p3.loc[emotions_order, 'Correlation'].values,
    'P1 Only Baseline': p1.loc[emotions_order, 'Correlation'].values,
}, index=emotions_order)

annot_p1p3 = [f"{r:.2f}{'*' if s == 'Yes' else ''}"
              for r, s in zip(p1p3.loc[emotions_order, 'Correlation'],
                              p1p3.loc[emotions_order, 'Significant'])]
annot_p1   = [f"{r:.2f}{'*' if s == 'Yes' else ''}"
              for r, s in zip(p1.loc[emotions_order, 'Correlation'],
                              p1.loc[emotions_order, 'Significant'])]

annot_matrix = pd.DataFrame({'P1 & P3 Baseline': annot_p1p3,
                              'P1 Only Baseline': annot_p1},
                             index=emotions_order)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr_matrix, annot=annot_matrix, fmt='', cmap='coolwarm',
            center=0, vmin=-0.4, vmax=0.4, linewidths=0.5, ax=ax,
            cbar_kws={'label': 'Point-biserial r'})
ax.set_title('Emotion–Anomaly Correlations by Baseline Training Set', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join("figs", "correlation_heatmap.png"), dpi=300)
plt.show()


# Figure 3: Forest plot — r with 95% bootstrap CI 
fig, ax = plt.subplots(figsize=(9, 7))
y_pos = np.arange(len(emotions_order))
offsets = [-0.22, 0.22]
colors_f = ['#1f77b4', '#ff7f0e']
labels_f = ['P1 & P3 Baseline', 'P1 Only Baseline']
keys_f   = ['phase1_phase3', 'phase1']

for offset, color, label, key in zip(offsets, colors_f, labels_f, keys_f):
    df_c = all_corr_dfs[key].set_index('Emotion')
    for j, em in enumerate(emotions_order):
        r    = df_c.loc[em, 'Correlation']
        lo   = df_c.loc[em, 'CI_lo']
        hi   = df_c.loc[em, 'CI_hi']
        sig  = df_c.loc[em, 'Significant'] == 'Yes'
        y    = y_pos[j] + offset
        ax.errorbar(r, y,
                    xerr=[[r - lo], [hi - r]],
                    fmt='o', color=color, capsize=4, linewidth=1.5,
                    markersize=7 if sig else 5,
                    markerfacecolor=color if sig else 'white',
                    markeredgecolor=color,
                    label=label if j == 0 else '')

ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_yticks(y_pos)
ax.set_yticklabels(emotions_order, fontsize=11)
ax.set_xlabel('Point-biserial r  (95% bootstrap CI)', fontsize=12)
ax.set_title('Emotion–Anomaly Correlations with 95% Confidence Intervals', fontsize=12)
ax.legend(loc='lower right')
ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join("figs", "correlation_forest.png"), dpi=300)
plt.show()


# Figure 4: Per-subject Phase 2 anomaly rate (Experiment 1) 
if results_phase1_phase3 is not None:
    mean_rate = results_phase1_phase3['Rate_p2%'].mean()
    std_rate  = results_phase1_phase3['Rate_p2%'].std()
    ci_lo, ci_hi = bootstrap_ci(results_phase1_phase3['Rate_p2%'].values)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(results_phase1_phase3['Test_Subject'],
               results_phase1_phase3['Rate_p2%'],
               color='#1f77b4', s=80, zorder=3)
    ax.axhline(mean_rate, color='red', linestyle='--', linewidth=1.5,
               label=f'Mean: {mean_rate:.1f}%')
    ax.axhspan(ci_lo, ci_hi, color='red', alpha=0.10,
               label=f'95% CI [{ci_lo:.1f}%, {ci_hi:.1f}%]')
    ax.axhline(mean_rate + std_rate, color='orange', linestyle=':', linewidth=1.2,
               label='Mean ± 1 SD')
    ax.axhline(mean_rate - std_rate, color='orange', linestyle=':', linewidth=1.2)
    ax.set_xlabel('Subject ID', fontsize=12)
    ax.set_ylabel('Phase 2 Anomaly Rate (%)', fontsize=12)
    ax.set_title('Per-Subject Phase 2 (Puzzle) Anomaly Rate — P1 & P3 Baseline', fontsize=12)
    ax.set_xticks(results_phase1_phase3['Test_Subject'])
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join("figs", "per_subject_rates.png"), dpi=300)
    plt.show()
