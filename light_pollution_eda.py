import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

# Set style plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12


df = pd.read_csv('/Users/meghanreilly/Downloads/light_pollution_data.csv')

print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Check cities
print("\nCity counts:")
print(df['city'].value_counts())

# Missing values
print("\nMissing values:")
print(df.isnull().sum())
df = df.fillna(0)

# Define variables
predictors = ['building_volume', 'population_density', 'greenhouse_coverage', 'road_density']
outcome = 'light_pollution'

print(df[predictors + [outcome]].describe())

# City-specific
print(df[df['city'] == 'Den Haag'][predictors + [outcome]].describe())
print(df[df['city'] == 'Amsterdam'][predictors + [outcome]].describe())

# Make directory
import os
os.makedirs('/Users/meghanreilly/Desktop/light_pollution/eda_plots', exist_ok=True)

# Distribution of both cities
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle('Variable Distributions - Both Cities', fontsize=14, fontweight='bold')

# Creating separate histogram plots for predictors and outcome variables
for i, var in enumerate(predictors + [outcome]):
    ax = axes[i // 2, i % 2]
    ax.hist(df[var], bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel(var.replace('_', ' ').title())
    ax.set_ylabel('Frequency')
    mean_val = df[var].mean()
    med_val = df[var].median()
    ax.set_title(f'{var} (mean={mean_val:.1f}, median={med_val:.1f})')
    ax.grid(True, alpha=0.3)

axes[2, 1].remove()
plt.tight_layout()
plt.savefig('/Users/meghanreilly/Desktop/light_pollution/eda_plots/01_distributions_overall.png', dpi=300, bbox_inches='tight')
plt.close()

# Distribution by city
fig, axes = plt.subplots(5, 2, figsize=(15, 18))
fig.suptitle('Amsterdam vs Den Haag Distributions', fontsize=14, fontweight='bold')

for i, var in enumerate(predictors + [outcome]):
    # amsterdam
    ax1 = axes[i, 0]
    ams = df[df['city'] == 'Amsterdam'][var]
    ax1.hist(ams, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel(var.replace('_', ' ').title())
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Amsterdam: {var} (n={len(ams)}, mean={ams.mean():.1f})')
    ax1.grid(True, alpha=0.3)
    
    # den haag
    ax2 = axes[i, 1]
    hague = df[df['city'] == 'Den Haag'][var]
    ax2.hist(hague, bins=50, alpha=0.7, color='coral', edgecolor='black')
    ax2.set_xlabel(var.replace('_', ' ').title())
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Den Haag: {var} (n={len(hague)}, mean={hague.mean():.1f})')
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/meghanreilly/Desktop/light_pollution/eda_plots/02_distributions_by_city.png', dpi=300, bbox_inches='tight')
plt.close()

# function for LOESS smoothing
def add_loess(x, y, frac=0.3):
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) > 10:
        idx = np.argsort(x_clean)
        x_sort = x_clean[idx]
        y_sort = y_clean[idx]
        smooth = lowess(y_sort, x_sort, frac=frac)
        return smooth[:, 0], smooth[:, 1]
    return None, None

# Bivariate scatterplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Bivariate Relationships with Light Pollution', fontsize=14, fontweight='bold')

for i, pred in enumerate(predictors):
    ax = axes[i // 2, i % 2]
    
    ax.scatter(df[pred], df[outcome], alpha=0.3, s=10, c='gray')
    
    # add LOESS
    x_sm, y_sm = add_loess(df[pred].values, df[outcome].values)
    if x_sm is not None:
        ax.plot(x_sm, y_sm, 'r-', linewidth=2.5, label='LOESS')
    
    # correlation
    corr = df[[pred, outcome]].corr(method='spearman').iloc[0, 1]
    ax.text(0.05, 0.95, f'ρ = {corr:.3f}', transform=ax.transAxes, 
            fontsize=11, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel(pred.replace('_', ' ').title())
    ax.set_ylabel('Light Pollution')
    ax.set_title(pred.replace('_', ' ').title())
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig('/Users/meghanreilly/Desktop/light_pollution/eda_plots/03_bivariate_overall.png', dpi=300, bbox_inches='tight')
plt.close()

# Bivariate plots by city
for pred in predictors:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{pred.replace("_", " ").title()} vs Light Pollution by City', fontsize=14)
    
    for idx, city in enumerate(['Amsterdam', 'Den Haag']):
        ax = axes[idx]
        city_data = df[df['city'] == city]
        
        color = 'steelblue' if city == 'Amsterdam' else 'coral'
        ax.scatter(city_data[pred], city_data[outcome], alpha=0.4, s=15, c=color)
        
        # LOESS
        x_sm, y_sm = add_loess(city_data[pred].values, city_data[outcome].values)
        if x_sm is not None:
            ax.plot(x_sm, y_sm, 'darkred', linewidth=2.5, label='LOESS')
        
        r = city_data[[pred, outcome]].corr(method='spearman').iloc[0, 1]
        ax.text(0.05, 0.95, f'ρ = {r:.3f}\nn = {len(city_data)}', 
                transform=ax.transAxes, fontsize=10, va='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel(pred.replace('_', ' ').title())
        ax.set_ylabel('Light Pollution')
        ax.set_title(city)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    fname = f'04_bivariate_{pred}_by_city.png'
    plt.savefig(f'/Users/meghanreilly/Desktop/light_pollution/eda_plots/{fname}', dpi=300, bbox_inches='tight')
    plt.close()

# Greenhouse in Den Haag visualizations
hague_data = df[df['city'] == 'Den Haag'].copy()
hague_data['has_gh'] = hague_data['greenhouse_coverage'] > 0

print(f"Den Haag cells with greenhouses: {hague_data['has_gh'].sum()}")
print(f"Den Haag cells without greenhouses: {(~hague_data['has_gh']).sum()}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Greenhouse Lighting Effect - Den Haag', fontsize=14, fontweight='bold')

# boxplot comparison
ax1 = axes[0]
hague_data.boxplot(column=outcome, by='has_gh', ax=ax1)
ax1.set_xlabel('Has Greenhouses')
ax1.set_ylabel('Light Pollution')
ax1.set_title('With vs Without Greenhouses')
ax1.set_xticklabels(['No', 'Yes'])
plt.sca(ax1)
plt.xticks(rotation=0)

# scatter for greenhouse areas
ax2 = axes[1]
gh_only = hague_data[hague_data['greenhouse_coverage'] > 0]
ax2.scatter(gh_only['greenhouse_coverage'], gh_only[outcome], 
           alpha=0.5, s=20, c='green')

# add LOESS
x_sm, y_sm = add_loess(gh_only['greenhouse_coverage'].values, gh_only[outcome].values)
if x_sm is not None:
    ax2.plot(x_sm, y_sm, 'darkgreen', linewidth=2.5)

r_gh = gh_only[['greenhouse_coverage', outcome]].corr(method='spearman').iloc[0, 1]
ax2.text(0.05, 0.95, f'ρ = {r_gh:.3f}\n(greenhouse areas only)', 
        transform=ax2.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
ax2.set_xlabel('Greenhouse Coverage')
ax2.set_ylabel('Light Pollution')
ax2.set_title('Greenhouse Coverage vs Light Pollution')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/meghanreilly/Desktop/light_pollution/eda_plots/05_greenhouse_effect.png', dpi=300, bbox_inches='tight')
plt.close()