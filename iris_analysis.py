"""
Data Analysis with Pandas and Visualization with Matplotlib
Assignment Solution - Iris Dataset Analysis
Final Working Version
"""

# ===== Import Libraries =====
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ===== Task 1: Load and Explore the Dataset =====
print("\n" + "="*50)
print("=== Task 1: Load and Explore the Dataset ===")
print("="*50 + "\n")

# Load dataset
try:
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['species'] = iris.target
    iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    print("✓ Dataset loaded successfully from sklearn.datasets")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Display first few rows
print("\nFirst 5 rows of the dataset:")
print(iris_df.head())

# Dataset structure
print("\nDataset info:")
print(iris_df.info())

# Check for missing values
print("\nMissing values summary:")
print(iris_df.isnull().sum())

# ===== Task 2: Basic Data Analysis =====
print("\n" + "="*50)
print("=== Task 2: Basic Data Analysis ===")
print("="*50 + "\n")

# Basic statistics
print("Basic statistics for numerical columns:")
print(iris_df.describe())

# Group by species and calculate mean
print("\nMean values by species:")
print(iris_df.groupby('species').mean(numeric_only=True))

# ===== Task 3: Data Visualization =====
print("\n" + "="*50)
print("=== Task 3: Data Visualization ===")
print("="*50 + "\n")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# 1. Line Chart
print("\n1. Creating Line Chart...")
plt.figure()
iris_df['sepal length (cm)'].plot(kind='line', color='royalblue')
plt.title('Sepal Length Across Samples')
plt.xlabel('Sample Index')
plt.ylabel('Sepal Length (cm)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('sepal_length_line.png')  # Save the figure
plt.close()
print("Saved as sepal_length_line.png")

# 2. Bar Chart
print("\n2. Creating Bar Chart...")
plt.figure()
iris_df.groupby('species')['petal length (cm)'].mean().plot(
    kind='bar', 
    color=['#1f77b4', '#ff7f0e', '#2ca02c'],
    edgecolor='black'
)
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('petal_length_bar.png')
plt.close()
print("Saved as petal_length_bar.png")

# 3. Histogram
print("\n3. Creating Histogram...")
plt.figure()
iris_df['sepal width (cm)'].hist(bins=15, edgecolor='black', color='#17becf')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('sepal_width_hist.png')
plt.close()
print("Saved as sepal_width_hist.png")

# 4. Scatter Plot
print("\n4. Creating Scatter Plot...")
plt.figure()
colors = {'setosa': '#d62728', 'versicolor': '#9467bd', 'virginica': '#8c564b'}
plt.scatter(
    iris_df['sepal length (cm)'], 
    iris_df['petal length (cm)'], 
    c=iris_df['species'].map(colors), 
    alpha=0.8,
    edgecolor='black',
    s=80
)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(
    handles=[plt.Line2D([0], [0], marker='o', color='w', label=species,
             markerfacecolor=color, markersize=10, markeredgecolor='black')
             for species, color in colors.items()],
    title='Species'
)
plt.tight_layout()
plt.savefig('sepal_petal_scatter.png')
plt.close()
print("Saved as sepal_petal_scatter.png")

# 5. Pairplot
print("\n5. Creating Pairplot...")
try:
    sns.pairplot(iris_df, hue='species', palette='husl', corner=True, height=2.5)
    plt.suptitle('Pairwise Relationships in Iris Dataset', y=1.02)
    plt.tight_layout()
    plt.savefig('pairplot.png')
    plt.close()
    print("Saved as pairplot.png")
except Exception as e:
    print(f"Could not create pairplot: {e}")

print("\n" + "="*50)
print("=== Analysis Complete ===")
print("All visualizations saved as PNG files in current directory")
print("="*50)