import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Uncomment seaborn if allowed, otherwise use matplotlib equivalents 
import seaborn as sns

# Load datasets as before
red = pd.read_csv('Wine_Quality_Red.csv', sep=';')
white = pd.read_csv('Wine_Quality_White.csv', sep=';')

# Add type and merge
red['type'] = 'red'
white['type'] = 'white'
wine = pd.concat([red, white], axis=0).reset_index(drop=True)
wine['quality_label'] = (wine['quality'] >= 7).astype(int)

# 1. Class balance by wine type
plt.figure(figsize=(5,3))
sns.countplot(x='type', data=wine, palette=['#B22222','#ffa726'])
plt.title('Wine Type Count')
plt.show()

# 2. Quality distribution
plt.figure(figsize=(6,3))
sns.countplot(x='quality', data=wine, palette="Blues")
plt.title('Wine Quality Score Distribution')
plt.show()

# 3. Quality by wine type (stacked bar)
pd.crosstab(wine['quality'], wine['type']).plot(kind='bar', stacked=True, colormap='RdGy', figsize=(7,4))
plt.title('Quality Scores by Wine Type')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.show()

# 4. Feature distribution by wine type
features = ['alcohol', 'pH', 'sulphates', 'citric acid']
for feat in features:
    plt.figure(figsize=(5,3))
    sns.kdeplot(data=wine, x=feat, hue='type', fill=True, alpha=0.4)
    plt.title(f'Distribution of {feat} by Wine Type')
    plt.show()

# 5. Boxplots of top features by quality
for feat in features:
    plt.figure(figsize=(6,3))
    sns.boxplot(x='quality', y=feat, data=wine, palette='Set2')
    plt.title(f'{feat} by Quality')
    plt.show()

# 6. Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(wine.corr(numeric_only=True), annot=False, cmap='coolwarm', linewidths=0.2)
plt.title('Feature Correlation Matrix')
plt.show()

# 7. Pairplot for multivariate patterns (smaller sample for speed)
sns.pairplot(wine.sample(500), vars=['alcohol', 'pH', 'sulphates', 'volatile acidity', 'quality'], hue='type')
plt.suptitle('Pairwise Feature Relationships (sample)', y=1.01)
plt.show()

# 8. Scatterplot: alcohol vs density colored by quality
plt.figure(figsize=(6,4))
plt.scatter(wine['alcohol'], wine['density'], c=wine['quality'], cmap='viridis', alpha=0.3)
plt.xlabel('Alcohol')
plt.ylabel('Density')
plt.title('Alcohol vs Density colored by Quality')
plt.colorbar(label='Quality')
plt.show()
