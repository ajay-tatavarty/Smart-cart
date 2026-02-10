# Smart Cart - Customer Segmentation Analysis

## Overview

Smart Cart is a comprehensive customer segmentation analysis project that uses machine learning clustering techniques to identify distinct customer groups based on their purchasing behavior and demographics.

## Project Structure

```
smart-cart/
├── data/                      # Customer data directory
│   └── smartcart_customers.csv
├── notebooks/                 # Jupyter notebooks for exploration
├── results/                   # Generated visualizations and reports
├── src/                       # Main Python modules
│   ├── __init__.py
│   ├── data_preprocessing.py  # Data cleaning and preprocessing
│   ├── feature_engineering.py # Feature creation and transformation
│   ├── clustering.py          # Clustering algorithms
│   └── visualization.py       # Plotting utilities
├── .vscode/                   # VS Code configuration
│   ├── settings.json
│   └── launch.json
├── .gitignore
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── main.py                    # Main analysis script
```

## Features

### Data Preprocessing
- Load customer data from CSV
- Handle missing values using median imputation
- Remove statistical outliers
- Data quality validation

### Feature Engineering
- **Age Calculation**: From birth year
- **Customer Tenure**: Days since customer joined
- **Total Spending**: Aggregated across product categories
- **Children Count**: Total number of children at home
- **Education Standardization**: Grouped into 3 categories (Undergraduate, Graduate, Postgraduate)
- **Marital Status**: Simplified to living situation (Partner, Alone)

### Clustering Analysis
- **Dimensionality Reduction**: PCA (Principal Component Analysis)
- **K-Means Clustering**: Traditional distance-based clustering
- **Hierarchical Clustering**: Agglomerative clustering with Ward linkage
- **Optimal K Selection**:
  - Elbow method using WCSS
  - Silhouette score analysis
  - KneeLocator for automatic elbow detection

### Visualization
- 3D PCA scatter plots colored by clusters
- Elbow curves for K selection
- Silhouette score plots
- Cluster distribution histograms
- Spending vs Income scatter plots
- Correlation heatmaps

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or navigate to the project directory**:
   ```bash
   cd Smart\ cart
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - **Windows (PowerShell)**:
     ```bash
     .\venv\Scripts\Activate.ps1
     ```
   - **Windows (Command Prompt)**:
     ```bash
     venv\Scripts\activate.bat
     ```
   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Full Analysis

Execute the main analysis pipeline:

```bash
python main.py
```

This will:
1. Load customer data from `data/smartcart_customers.csv`
2. Preprocess and clean the data
3. Engineer relevant features
4. Scale features and apply PCA
5. Perform clustering analysis
6. Generate visualizations in the `results/` folder

### Using in Python Code

```python
from src import data_preprocessing as dp
from src import feature_engineering as fe
from src import clustering as clust
from src import visualization as viz

# Load and preprocess data
df = dp.load_data("data/smartcart_customers.csv")
df = dp.handle_missing_values(df)
df = fe.engineer_features(df)
df = dp.remove_outliers(df)

# Prepare for clustering
df_encoded = clust.encode_categorical_features(df, ["Education", "Living_With"])
X_scaled, scaler = clust.scale_features(df_encoded)
X_pca, pca = clust.apply_pca(X_scaled, n_components=3)

# Perform clustering
labels = clust.kmeans_clustering(X_pca, n_clusters=4)

# Visualize
viz.plot_3d_scatter(X_pca, labels)
```

### Running Jupyter Notebooks

For interactive exploration, use Jupyter:

```bash
jupyter notebook notebooks/
```

## Data Input

The project requires a CSV file with the following columns:

**Demographics:**
- `ID`: Customer identifier
- `Year_Birth`: Birth year
- `Marital_Status`: Marital status
- `Dt_Customer`: Customer acquisition date
- `Income`: Annual income

**Spending Patterns:**
- `MntWines`: Spending on wine products
- `MntFruits`: Spending on fruit products
- `MntMeatProducts`: Spending on meat
- `MntFishProducts`: Spending on fish
- `MntSweetProducts`: Spending on sweets
- `MntGoldProds`: Spending on gold products

**Family:**
- `Kidhome`: Number of children at home
- `Teenhome`: Number of teenagers at home

**Engagement:**
- `Recency`: Days since last purchase
- `Response`: Customer response rate

Place your data file at: `data/smartcart_customers.csv`

## Output

The analysis generates:

1. **Cluster Visualizations** (results/ folder):
   - `01_3d_clustering.png` - 3D PCA visualization with cluster colors
   - `02_elbow_curve.png` - WCSS elbow plot for K selection
   - `03_silhouette_scores.png` - Silhouette analysis
   - `04_cluster_distribution.png` - Cluster sizes
   - `05_spending_vs_income.png` - Customer segments by spending/income
   - `06_correlation_heatmap.png` - Feature correlations

2. **Console Output**:
   - Data statistics
   - Feature engineering summary
   - Cluster analysis metrics
   - Optimal K recommendations

## Module Documentation

### `data_preprocessing.py`
- `load_data()` - Load CSV data
- `handle_missing_values()` - Impute missing values
- `remove_outliers()` - Remove statistical outliers
- `get_data_info()` - Get dataset summary

### `feature_engineering.py`
- `calculate_age()` - Compute age from birth year
- `calculate_customer_tenure()` - Compute customer lifetime days
- `calculate_total_spending()` - Sum spending across categories
- `calculate_total_children()` - Total children count
- `standardize_education()` - Standardize education levels
- `standardize_marital_status()` - Simplify marital status
- `engineer_features()` - Apply all feature engineering steps

### `clustering.py`
- `encode_categorical_features()` - One-hot encode categories
- `scale_features()` - Standardize feature scales
- `apply_pca()` - Dimensionality reduction
- `find_optimal_k()` - Elbow method
- `calculate_silhouette_scores()` - Silhouette analysis
- `kmeans_clustering()` - K-means clustering
- `agglomerative_clustering()` - Hierarchical clustering
- `analyze_clusters()` - Cluster characteristics

### `visualization.py`
- `plot_3d_scatter()` - 3D PCA visualization
- `plot_correlation_heatmap()` - Correlation matrix
- `plot_elbow_curve()` - WCSS curve
- `plot_silhouette_scores()` - Silhouette plot
- `plot_cluster_distribution()` - Cluster sizes
- `plot_spending_vs_income()` - Customer segmentation plot
- `save_figure()` - Export plots to PNG

## Configuration

### VS Code Settings
The `.vscode/settings.json` file configures:
- Python interpreter path
- Code formatting (Black)
- Linting (Pylint)
- Auto-format on save

### Debug Configuration
The `.vscode/launch.json` file includes:
- Main analysis script debugger
- Current file debugging

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 2.1.4 | Data manipulation |
| numpy | 1.24.3 | Numerical computing |
| scikit-learn | 1.3.2 | Machine learning |
| matplotlib | 3.8.2 | Plotting |
| seaborn | 0.13.0 | Statistical visualization |
| kneed | 0.8.1 | Elbow point detection |
| jupyter | 1.0.0 | Interactive notebooks |

## Example Results

When run successfully, the analysis typically identifies 3-5 customer segments:
- **High-Value Customers**: High income, high spending
- **Moderate Spenders**: Medium income, consistent purchases
- **Price-Conscious**: Lower income, lower spending
- **Inactive Customers**: Low engagement metrics

## Troubleshooting

**ModuleNotFoundError**: Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

**Data file not found**: Place `smartcart_customers.csv` in the `data/` directory

**PCA error**: Ensure all features are numeric before scaling

**Visualization issues**: Check that matplotlib backend is configured correctly

## Future Enhancements

- [ ] Customer demographic profiling
- [ ] RFM (Recency, Frequency, Monetary) analysis
- [ ] Cluster interpretation and naming
- [ ] Model persistence and prediction on new data
- [ ] Interactive dashboard with Plotly
- [ ] Automated report generation
- [ ] Time-series customer behavior analysis

## License

This project is provided as-is for educational and analytical purposes.

## Contact & Support

For questions or issues, please refer to the code documentation or Python docstrings in each module.

---

**Last Updated**: February 2026
**Version**: 1.0.0
