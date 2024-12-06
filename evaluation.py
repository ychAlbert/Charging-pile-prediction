import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import scipy.stats as stats

def preliminary_analysis(X):
    """
    初步数据分析
    
    参数:
        X (DataFrame): 特征矩阵
    
    返回:
        dict: 综合统计见解
    """
    # 描述统计
    desc_stats = X.describe()
    
    # 正态性检验 (Shapiro-Wilk)，可以看这篇文章：https://blog.csdn.net/lvsehaiyang1993/article/details/80473265
    normality_tests = {}
    for column in tqdm(X.columns, desc="Normality Tests"):
        _, p_value = stats.shapiro(X[column])
        normality_tests[column] = {
            'is_normal': p_value > 0.05,
            'p_value': p_value
        }
    
    # 做相关分析
    correlation_matrix = X.corr()
    
    plt.figure(figsize=(15, 10))
    X.hist(bins=30, figsize=(15, 10))
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    plt.close()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()
    
    return {
        'descriptive_stats': desc_stats,
        'normality_tests': normality_tests,
        'correlation_matrix': correlation_matrix
    }