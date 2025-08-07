import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# we load all the data
x = np.load('/home/janis/SCIML/summerschool/data/benchmarks/given/x.npy')
f_train = np.load('/home/janis/SCIML/summerschool/data/benchmarks/given/f_train_data.npy')
f_test = np.load('/home/janis/SCIML/summerschool/data/benchmarks/given/f_test_data.npy')

u_train_c02 = np.load('/home/janis/SCIML/summerschool/data/benchmarks/given/u_train_data_c0.2.npy')
u_train_c05 = np.load('/home/janis/SCIML/summerschool/data/benchmarks/given/u_train_data_c0.5.npy')
u_train_c10 = np.load('/home/janis/SCIML/summerschool/data/benchmarks/given/u_train_data_c1.0.npy')

u_test_c02 = np.load('/home/janis/SCIML/summerschool/data/benchmarks/given/u_test_data_c0.2.npy')
u_test_c05 = np.load('/home/janis/SCIML/summerschool/data/benchmarks/given/u_test_data_c0.5.npy')
u_test_c10 = np.load('/home/janis/SCIML/summerschool/data/benchmarks/given/u_test_data_c1.0.npy')

print("Dimensions des données:")
print(f"x: {x.shape}")
print(f"f_train: {f_train.shape}")
print(f"f_test: {f_test.shape}")
print(f"u_train_c02: {u_train_c02.shape}")
print(f"u_train_c05: {u_train_c05.shape}")
print(f"u_train_c10: {u_train_c10.shape}")

# we combine training and test data for complete analysis
f_all = np.concatenate([f_train, f_test], axis=0)
u_all_c02 = np.concatenate([u_train_c02, u_test_c02], axis=0)
u_all_c05 = np.concatenate([u_train_c05, u_test_c05], axis=0)
u_all_c10 = np.concatenate([u_train_c10, u_test_c10], axis=0)

print(f"\nDonnées combinées:")
print(f"f_all: {f_all.shape}")
print(f"u_all_c02: {u_all_c02.shape}")

# we define batch ranges from 0.0-0.1, 0.1-0.2, ..., 0.9-1.0
num_batches = 10
batch_edges = np.linspace(0, 1, num_batches + 1)
batch_labels = [f"{batch_edges[i]:.1f}-{batch_edges[i+1]:.1f}" for i in range(num_batches)]

def analyze_batch_variations(data, data_name):
    """we analyze min-max variations for each batch"""
    batch_stats = []
    
    for i in range(num_batches):
        start_idx = int(i * len(data) / num_batches)
        end_idx = int((i + 1) * len(data) / num_batches)
        
        batch_data = data[start_idx:end_idx]  # we select batch data
        
        # we calculate min-max variation for each sample in the batch
        variations = np.max(batch_data, axis=1) - np.min(batch_data, axis=1)
        
        # we compute statistics for this batch
        batch_stats.append({
            'batch': batch_labels[i],
            'batch_idx': i,
            'n_samples': len(batch_data),
            'var_mean': np.mean(variations),
            'var_std': np.std(variations),
            'var_min': np.min(variations),
            'var_max': np.max(variations),
            'var_median': np.median(variations)
        })
    
    return pd.DataFrame(batch_stats)

# we analyze μ (f) data
print("\n" + "="*60)
print("ANALYSE DES VARIATIONS POUR μ (f)")
print("="*60)
f_stats = analyze_batch_variations(f_all, 'μ (f)')
print(f_stats.round(4))

# we analyze solution data for each c value
print("\n" + "="*60)
print("ANALYSE DES VARIATIONS POUR SOLUTIONS u")
print("="*60)

for c_val, u_data in [('c=0.2', u_all_c02), ('c=0.5', u_all_c05), ('c=1.0', u_all_c10)]:
    print(f"\n--- Solutions pour {c_val} ---")
    u_stats = analyze_batch_variations(u_data, f'u ({c_val})')
    print(u_stats.round(4))

# we create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# we plot μ (f) variations
axes[0,0].bar(range(num_batches), f_stats['var_mean'], alpha=0.7, color='blue')
axes[0,0].errorbar(range(num_batches), f_stats['var_mean'], yerr=f_stats['var_std'], 
                   fmt='none', color='black', capsize=3)
axes[0,0].set_title('Variations moyennes μ (f) par batch')
axes[0,0].set_xlabel('Batch')
axes[0,0].set_ylabel('Variation moyenne (max-min)')
axes[0,0].set_xticks(range(num_batches))
axes[0,0].set_xticklabels(batch_labels, rotation=45)

# we plot solution variations for each c
colors = ['red', 'green', 'orange']
c_values = ['c=0.2', 'c=0.5', 'c=1.0']
u_datasets = [u_all_c02, u_all_c05, u_all_c10]

for idx, (c_val, u_data, color) in enumerate(zip(c_values, u_datasets, colors)):
    u_stats = analyze_batch_variations(u_data, f'u ({c_val})')
    
    ax = axes[0,1] if idx == 0 else axes[1,0] if idx == 1 else axes[1,1]
    
    ax.bar(range(num_batches), u_stats['var_mean'], alpha=0.7, color=color)
    ax.errorbar(range(num_batches), u_stats['var_mean'], yerr=u_stats['var_std'], 
                fmt='none', color='black', capsize=3)
    ax.set_title(f'Variations moyennes solutions {c_val} par batch')
    ax.set_xlabel('Batch')
    ax.set_ylabel('Variation moyenne (max-min)')
    ax.set_xticks(range(num_batches))
    ax.set_xticklabels(batch_labels, rotation=45)

plt.tight_layout()
plt.show()

# we identify potentially problematic batches
print("\n" + "="*60)
print("DÉTECTION DES BATCHES POTENTIELLEMENT PROBLÉMATIQUES")
print("="*60)

# we check for batches with unusually low variation (less expressive)
f_mean_var = f_stats['var_mean'].mean()
f_std_var = f_stats['var_mean'].std()
threshold_low = f_mean_var - 1.5 * f_std_var

print(f"\nPour μ (f):")
print(f"Variation moyenne globale: {f_mean_var:.4f}")
print(f"Écart-type: {f_std_var:.4f}")
print(f"Seuil bas (moyenne - 1.5*std): {threshold_low:.4f}")

low_var_batches_f = f_stats[f_stats['var_mean'] < threshold_low]
if len(low_var_batches_f) > 0:
    print("Batches avec faible variation (potentiellement peu expressifs):")
    print(low_var_batches_f[['batch', 'var_mean', 'var_std']].round(4))
else:
    print("Aucun batch avec variation anormalement faible détecté pour μ (f)")

# we do the same for solutions
for c_val, u_data in [('c=0.2', u_all_c02), ('c=0.5', u_all_c05), ('c=1.0', u_all_c10)]:
    u_stats = analyze_batch_variations(u_data, f'u ({c_val})')
    u_mean_var = u_stats['var_mean'].mean()
    u_std_var = u_stats['var_mean'].std()
    threshold_low_u = u_mean_var - 1.5 * u_std_var
    
    print(f"\nPour solutions {c_val}:")
    print(f"Variation moyenne globale: {u_mean_var:.4f}")
    print(f"Écart-type: {u_std_var:.4f}")
    print(f"Seuil bas (moyenne - 1.5*std): {threshold_low_u:.4f}")
    
    low_var_batches_u = u_stats[u_stats['var_mean'] < threshold_low_u]
    if len(low_var_batches_u) > 0:
        print(f"Batches avec faible variation pour {c_val}:")
        print(low_var_batches_u[['batch', 'var_mean', 'var_std']].round(4))
    else:
        print(f"Aucun batch avec variation anormalement faible détecté pour {c_val}")

# we add this enhanced analysis to identify low variation intervals more clearly

# we calculate relative variations compared to the overall dataset
def identify_low_variation_intervals(data, data_name, percentile_threshold=25):
    """we identify intervals with low max-min variations"""
    batch_stats = analyze_batch_variations(data, data_name)
    
    # we calculate percentile threshold for "low" variation
    variation_values = batch_stats['var_mean'].values
    threshold = np.percentile(variation_values, percentile_threshold)
    
    # we identify low variation batches
    low_variation_batches = batch_stats[batch_stats['var_mean'] <= threshold]
    
    print(f"\n{'='*50}")
    print(f"INTERVALLES À FAIBLE VARIATION POUR {data_name.upper()}")
    print(f"{'='*50}")
    print(f"Seuil {percentile_threshold}e percentile: {threshold:.6f}")
    print(f"Variation globale moyenne: {variation_values.mean():.6f}")
    print(f"Variation globale médiane: {np.median(variation_values):.6f}")
    
    if len(low_variation_batches) > 0:
        print(f"\nIntervalles avec variation ≤ {threshold:.6f}:")
        for _, batch in low_variation_batches.iterrows():
            ratio_vs_mean = batch['var_mean'] / variation_values.mean()
            print(f"  • Batch {batch['batch']}: variation = {batch['var_mean']:.6f} "
                  f"({ratio_vs_mean:.2%} de la moyenne globale)")
    else:
        print("Aucun intervalle avec variation particulièrement faible")
    
    # we show relative comparison
    print(f"\nComparaison relative (ratio par rapport à la moyenne globale):")
    for _, batch in batch_stats.iterrows():
        ratio = batch['var_mean'] / variation_values.mean()
        status = "🔴 FAIBLE" if ratio < 0.7 else "🟡 MODÉRÉ" if ratio < 0.9 else "🟢 NORMAL"
        print(f"  {batch['batch']}: {ratio:.2%} {status}")
    
    return batch_stats, low_variation_batches

# we run enhanced analysis for μ (f)
f_stats_detailed, f_low_var = identify_low_variation_intervals(f_all, 'μ (f)', percentile_threshold=25)

# we run enhanced analysis for each solution type
for c_val, u_data in [('c=0.2', u_all_c02), ('c=0.5', u_all_c05), ('c=1.0', u_all_c10)]:
    u_stats_detailed, u_low_var = identify_low_variation_intervals(u_data, f'Solutions {c_val}', percentile_threshold=25)

# we create a more detailed visualization focusing on low variations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# we enhance μ (f) plot with threshold line
ax = axes[0,0]
bars = ax.bar(range(num_batches), f_stats_detailed['var_mean'], alpha=0.7, color='blue')
threshold_f = np.percentile(f_stats_detailed['var_mean'], 25)
ax.axhline(y=threshold_f, color='red', linestyle='--', alpha=0.8, 
           label=f'25e percentile: {threshold_f:.6f}')
ax.axhline(y=f_stats_detailed['var_mean'].mean(), color='orange', linestyle='--', alpha=0.8,
           label=f'Moyenne: {f_stats_detailed["var_mean"].mean():.6f}')

# we highlight low variation bars
for i, bar in enumerate(bars):
    if f_stats_detailed.iloc[i]['var_mean'] <= threshold_f:
        bar.set_color('red')
        bar.set_alpha(0.9)

ax.set_title('Variations μ (f) par batch\n(Barres rouges = faible variation)')
ax.set_xlabel('Intervalle')
ax.set_ylabel('Variation moyenne (max-min)')
ax.set_xticks(range(num_batches))
ax.set_xticklabels(batch_labels, rotation=45)
ax.legend()
ax.grid(True, alpha=0.3)

# we do the same for solutions
colors = ['red', 'green', 'orange']
c_values = ['c=0.2', 'c=0.5', 'c=1.0']
u_datasets = [u_all_c02, u_all_c05, u_all_c10]

for idx, (c_val, u_data, color) in enumerate(zip(c_values, u_datasets, colors)):
    u_stats_detailed = analyze_batch_variations(u_data, f'u ({c_val})')
    
    ax = axes[0,1] if idx == 0 else axes[1,0] if idx == 1 else axes[1,1]
    
    bars = ax.bar(range(num_batches), u_stats_detailed['var_mean'], alpha=0.7, color=color)
    threshold_u = np.percentile(u_stats_detailed['var_mean'], 25)
    ax.axhline(y=threshold_u, color='red', linestyle='--', alpha=0.8,
               label=f'25e percentile: {threshold_u:.6f}')
    ax.axhline(y=u_stats_detailed['var_mean'].mean(), color='black', linestyle='--', alpha=0.8,
               label=f'Moyenne: {u_stats_detailed["var_mean"].mean():.6f}')
    
    # we highlight low variation bars
    for i, bar in enumerate(bars):
        if u_stats_detailed.iloc[i]['var_mean'] <= threshold_u:
            bar.set_color('darkred')
            bar.set_alpha(0.9)
    
    ax.set_title(f'Variations solutions {c_val} par batch\n(Barres foncées = faible variation)')
    ax.set_xlabel('Intervalle')
    ax.set_ylabel('Variation moyenne (max-min)')
    ax.set_xticks(range(num_batches))
    ax.set_xticklabels(batch_labels, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# we create a summary table of all low variation intervals
print("\n" + "="*80)
print("RÉSUMÉ DES INTERVALLES À FAIBLE VARIATION")
print("="*80)

all_low_var_intervals = []

# we collect all low variation intervals
for dataset_name, data in [('μ (f)', f_all), 
                          ('Solutions c=0.2', u_all_c02), 
                          ('Solutions c=0.5', u_all_c05), 
                          ('Solutions c=1.0', u_all_c10)]:
    stats = analyze_batch_variations(data, dataset_name)
    threshold = np.percentile(stats['var_mean'], 25)
    low_var = stats[stats['var_mean'] <= threshold]
    
    for _, batch in low_var.iterrows():
        all_low_var_intervals.append({
            'dataset': dataset_name,
            'interval': batch['batch'],
            'variation': batch['var_mean'],
            'ratio_vs_mean': batch['var_mean'] / stats['var_mean'].mean()
        })

if all_low_var_intervals:
    summary_df = pd.DataFrame(all_low_var_intervals)
    print("\nTous les intervalles identifiés comme ayant de faibles variations:")
    print(summary_df.round(6))
    
    # we group by interval to see which are problematic across multiple datasets
    interval_problems = summary_df.groupby('interval').size().reset_index(columns=['count'])
    interval_problems = interval_problems.sort_values('count', ascending=False)
    
    print("\nIntervalles problématiques (présents dans plusieurs datasets):")
    for _, row in interval_problems.iterrows():
        if row['count'] > 1:
            print(f"  • Intervalle {row['interval']}: problématique dans {row['count']} dataset(s)")
else:
    print("Aucun intervalle avec variation particulièrement faible détecté")
