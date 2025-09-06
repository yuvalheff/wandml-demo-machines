import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json

# Configuration
data_path = '/Users/yuvalheffetz/ds-agent-projects/session_ce64539f-782b-46c7-ab41-9bf37519daed/data/train_set.csv'
plots_dir = Path('/Users/yuvalheffetz/ds-agent-projects/session_ce64539f-782b-46c7-ab41-9bf37519daed/research/plots')
plots_dir.mkdir(exist_ok=True)

app_color_palette = [
    'rgba(99, 110, 250, 0.8)',   # Blue
    'rgba(239, 85, 59, 0.8)',    # Red/Orange-Red
    'rgba(0, 204, 150, 0.8)',    # Green
    'rgba(171, 99, 250, 0.8)',   # Purple
    'rgba(255, 161, 90, 0.8)',   # Orange
    'rgba(25, 211, 243, 0.8)',   # Cyan
    'rgba(255, 102, 146, 0.8)',  # Pink
    'rgba(182, 232, 128, 0.8)',  # Light Green
    'rgba(255, 151, 255, 0.8)',  # Magenta
    'rgba(254, 203, 82, 0.8)'    # Yellow
]

def apply_plot_styling(fig):
    """Apply consistent styling to plots"""
    fig.update_layout(
        height=550,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8B5CF6', size=12),
        title_font=dict(color='#7C3AED', size=16),
        xaxis=dict(
            gridcolor='rgba(139,92,246,0.2)',
            zerolinecolor='rgba(139,92,246,0.3)',
            tickfont=dict(color='#8B5CF6', size=11),
            title_font=dict(color='#7C3AED', size=12)
        ),
        yaxis=dict(
            gridcolor='rgba(139,92,246,0.2)',
            zerolinecolor='rgba(139,92,246,0.3)',
            tickfont=dict(color='#8B5CF6', size=11),
            title_font=dict(color='#7C3AED', size=12)
        ),
        legend=dict(font=dict(color='#8B5CF6', size=11))
    )
    return fig

# Load data
print("Loading dataset...")
df = pd.read_csv(data_path)
target_col = 'target'
features = [col for col in df.columns if col not in [target_col, 'id']]
numerical_features = df[features].select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df[features].select_dtypes(exclude=[np.number]).columns.tolist()

print(f"Dataset shape: {df.shape}")
print(f"Features: {len(features)} total, {len(numerical_features)} numerical, {len(categorical_features)} categorical")

# Store analysis results
eda_results = {}

# 1. Target Distribution Analysis
print("\n1. Analyzing target distribution...")
target_counts = df[target_col].value_counts()
target_props = df[target_col].value_counts(normalize=True)

fig = px.bar(x=['No Failure', 'Failure'], 
             y=[target_counts[0], target_counts[1]],
             labels={'x': 'Machine Status', 'y': 'Count'})

fig.update_traces(marker=dict(color=[app_color_palette[0], app_color_palette[1]]))
fig.update_layout(showlegend=False)
fig = apply_plot_styling(fig)
fig.write_html(plots_dir / "target_distribution.html", 
               include_plotlyjs=True, 
               config={'responsive': True, 'displayModeBar': False})

# 2. Missing Values Analysis
print("2. Analyzing missing values...")
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_data = missing_values[missing_values > 0]

if len(missing_data) > 0:
    fig = px.bar(x=missing_data.index, y=missing_data.values,
                 labels={'x': 'Features', 'y': 'Missing Count'})
    fig.update_traces(marker=dict(color=app_color_palette[1]))
    fig = apply_plot_styling(fig)
    fig.write_html(plots_dir / "missing_values.html", 
                   include_plotlyjs=True, 
                   config={'responsive': True, 'displayModeBar': False})
    has_missing = True
else:
    # Create a simple plot showing no missing values
    fig = px.bar(x=['No Missing Values'], y=[0],
                 labels={'x': 'Status', 'y': 'Count'})
    fig.update_traces(marker=dict(color=app_color_palette[2]))
    fig = apply_plot_styling(fig)
    fig.write_html(plots_dir / "missing_values.html", 
                   include_plotlyjs=True, 
                   config={'responsive': True, 'displayModeBar': False})
    has_missing = False

# 3. Feature Distributions
print("3. Analyzing feature distributions...")
# Calculate skewness
skewness = df[numerical_features].skew().sort_values(key=abs, ascending=False)
most_skewed_features = skewness.head(6).index.tolist()

fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=most_skewed_features,
    vertical_spacing=0.08
)

for i, feature in enumerate(most_skewed_features):
    row = i // 3 + 1
    col = i % 3 + 1
    
    fig.add_trace(
        go.Histogram(x=df[feature], name=feature, showlegend=False,
                    marker_color=app_color_palette[i % len(app_color_palette)]),
        row=row, col=col
    )

fig.update_layout(
    height=550,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#8B5CF6', size=10)
)
fig.update_xaxes(gridcolor='rgba(139,92,246,0.2)', tickfont=dict(color='#8B5CF6', size=9))
fig.update_yaxes(gridcolor='rgba(139,92,246,0.2)', tickfont=dict(color='#8B5CF6', size=9))

fig.write_html(plots_dir / "feature_distributions.html", 
               include_plotlyjs=True, 
               config={'responsive': True, 'displayModeBar': False})

# 4. Outlier Analysis
print("4. Analyzing outliers...")
outlier_counts = {}
for feature in numerical_features:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
    outlier_counts[feature] = len(outliers)

# Get top outlier features
top_outlier_features = sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)[:6]
top_outlier_feature_names = [f[0] for f in top_outlier_features]

fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=top_outlier_feature_names,
    vertical_spacing=0.1
)

for i, feature in enumerate(top_outlier_feature_names):
    row = i // 3 + 1
    col = i % 3 + 1
    
    fig.add_trace(
        go.Box(y=df[feature], name=feature, showlegend=False,
               marker_color=app_color_palette[i % len(app_color_palette)]),
        row=row, col=col
    )

fig.update_layout(
    height=550,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#8B5CF6', size=10)
)
fig.update_xaxes(gridcolor='rgba(139,92,246,0.2)', tickfont=dict(color='#8B5CF6', size=9))
fig.update_yaxes(gridcolor='rgba(139,92,246,0.2)', tickfont=dict(color='#8B5CF6', size=9))

fig.write_html(plots_dir / "outlier_analysis.html", 
               include_plotlyjs=True, 
               config={'responsive': True, 'displayModeBar': False})

# 5. Correlation Analysis
print("5. Analyzing correlations...")
corr_matrix = df[numerical_features + [target_col]].corr()
target_correlations = corr_matrix[target_col].drop(target_col).sort_values(key=abs, ascending=False)

# Create correlation heatmap for top features
top_features = target_correlations.head(15).index.tolist() + [target_col]
corr_subset = df[top_features].corr()

fig = go.Figure(data=go.Heatmap(
    z=corr_subset.values,
    x=corr_subset.columns,
    y=corr_subset.columns,
    colorscale='RdBu',
    zmid=0,
    text=np.round(corr_subset.values, 2),
    texttemplate="%{text}",
    textfont={"size": 8},
    hoverongaps=False
))

fig.update_layout(
    height=550,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#8B5CF6', size=10),
    xaxis=dict(tickangle=45, tickfont=dict(color='#8B5CF6', size=9)),
    yaxis=dict(tickfont=dict(color='#8B5CF6', size=9))
)

fig.write_html(plots_dir / "correlation_heatmap.html", 
               include_plotlyjs=True, 
               config={'responsive': True, 'displayModeBar': False})

# 6. Feature-Target Relationship
print("6. Analyzing feature-target relationships...")
top_target_features = target_correlations.head(8).index.tolist()

fig = make_subplots(
    rows=2, cols=4,
    subplot_titles=top_target_features,
    vertical_spacing=0.1
)

for i, feature in enumerate(top_target_features):
    row = i // 4 + 1
    col = i % 4 + 1
    
    # Create separate traces for each class
    for class_val in [0, 1]:
        class_data = df[df[target_col] == class_val][feature]
        class_name = 'No Failure' if class_val == 0 else 'Failure'
        
        fig.add_trace(
            go.Box(y=class_data, 
                   name=class_name,
                   showlegend=(i == 0),
                   marker_color=app_color_palette[class_val]),
            row=row, col=col
        )

fig.update_layout(
    height=550,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#8B5CF6', size=9),
    legend=dict(font=dict(color='#8B5CF6', size=10))
)
fig.update_xaxes(gridcolor='rgba(139,92,246,0.2)', tickfont=dict(color='#8B5CF6', size=8))
fig.update_yaxes(gridcolor='rgba(139,92,246,0.2)', tickfont=dict(color='#8B5CF6', size=8))

fig.write_html(plots_dir / "feature_target_relationship.html", 
               include_plotlyjs=True, 
               config={'responsive': True, 'displayModeBar': False})

# 7. Class Separation Analysis
print("7. Analyzing class separation...")
most_discriminative = target_correlations.head(6).index.tolist()

# Calculate class differences
class_differences = []
for feature in most_discriminative:
    mean_0 = df[df[target_col] == 0][feature].mean()
    mean_1 = df[df[target_col] == 1][feature].mean()
    std_0 = df[df[target_col] == 0][feature].std()
    std_1 = df[df[target_col] == 1][feature].std()
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((std_0**2 + std_1**2) / 2)
    cohens_d = abs(mean_1 - mean_0) / pooled_std if pooled_std > 0 else 0
    
    class_differences.append({
        'feature': feature,
        'cohens_d': cohens_d,
        'mean_difference': abs(mean_1 - mean_0)
    })

# Create scatter plot matrix for top discriminative features
top_3_features = sorted(class_differences, key=lambda x: x['cohens_d'], reverse=True)[:3]
top_3_feature_names = [f['feature'] for f in top_3_features]

fig = px.scatter_matrix(
    df, 
    dimensions=top_3_feature_names,
    color=df[target_col].map({0: 'No Failure', 1: 'Failure'}),
    color_discrete_map={'No Failure': app_color_palette[0], 'Failure': app_color_palette[1]}
)

fig.update_layout(
    height=550,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#8B5CF6', size=9),
    legend=dict(font=dict(color='#8B5CF6', size=10))
)

fig.update_traces(diagonal_visible=False, marker=dict(size=3, opacity=0.7))

fig.write_html(plots_dir / "class_separation_analysis.html", 
               include_plotlyjs=True, 
               config={'responsive': True, 'displayModeBar': False})

print(f"\nEDA analysis completed!")
print(f"Generated {len(list(plots_dir.glob('*.html')))} plots in {plots_dir}")

# Store results for report generation
results = {
    'dataset_shape': df.shape,
    'target_distribution': target_counts.to_dict(),
    'target_proportions': target_props.to_dict(),
    'feature_counts': {'numerical': len(numerical_features), 'categorical': len(categorical_features)},
    'has_missing_values': has_missing,
    'top_skewed_features': most_skewed_features,
    'top_outlier_features': top_outlier_feature_names,
    'top_target_correlations': target_correlations.head(10).to_dict(),
    'class_differences': class_differences
}

# Save analysis results
with open('/Users/yuvalheffetz/ds-agent-projects/session_ce64539f-782b-46c7-ab41-9bf37519daed/research/eda_analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("Analysis results saved!")