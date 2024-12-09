'''
@Time    : 2024-12-09
@File    : ablationExpMain.py
@writer  : ychAlbert 
本代码用于实现模型对比实验，包括数据探索性分析、数据预处理、特征选择等步骤。为消融实验中使用。
为方便调试，本代码中没有使用任何第三方库，仅使用Python标准库和sklearn库。可单独运行，也可以作为模块被其他代码调用。
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report, 
    roc_auc_score, 
    roc_curve
)

from sklearn.model_selection import GridSearchCV

# Statistical Analysis
import scipy.stats as stats

class AdvancedMLPipeline:
    def __init__(self, data_path):
        """
        使用综合数据处理初始化机器学习流水线
        
        参数:
            data_path (str): 输入CSV文件的路径
        """

        self.raw_data = pd.read_csv(data_path)
        
        self.X = self.raw_data.drop(['id', 'label'], axis=1)
        self.y = self.raw_data['label']
        
        self.analysis_results = {}
    
    def preliminary_analysis(self):
        """
        初步数据分析
        
        返回:
            dict: 综合统计见解
        """
        # 描述统计
        desc_stats = self.X.describe()
        
        # 正态性检验 (Shapiro-Wilk)
        normality_tests = {}
        for column in tqdm(self.X.columns, desc="Normality Tests"):
            _, p_value = stats.shapiro(self.X[column])
            normality_tests[column] = {
                'is_normal': p_value > 0.05,
                'p_value': p_value
            }
        
        # 做相关分析
        correlation_matrix = self.X.corr()
        
        plt.figure(figsize=(15, 10))
        self.X.hist(bins=30, figsize=(15, 10))
        plt.tight_layout()
        plt.savefig('feature_distributions.png')
        plt.close()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        plt.close()
        
        self.analysis_results['descriptive_stats'] = desc_stats
        self.analysis_results['normality_tests'] = normality_tests
        
        return {
            'descriptive_stats': desc_stats,
            'normality_tests': normality_tests,
            'correlation_matrix': correlation_matrix
        }
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        """
        使用多种缩放和特征选择技术进行高级数据预处理
        
        参数:
            test_size (float): 测试数据的比例
            random_state (int): 随机种子以保证结果可重复
            
        返回:
            tuple: 处理后的训练和测试数据集
        """

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            stratify=self.y, 
            random_state=random_state
        )
        
        feature_selector = Pipeline([
            ('scaler', RobustScaler()),  # 更加鲁棒地处理异常值
            ('feature_selection', SelectKBest(score_func=f_classif, k=4)),  # 选择前4个特征
            ('pca', PCA(n_components=0.95))  # 保留95%的方差
        ])
        
        X_train_selected = feature_selector.fit_transform(X_train, y_train)
        X_test_selected = feature_selector.transform(X_test)
        
        return (
            X_train_selected, X_test_selected, 
            y_train, y_test, 
            feature_selector
        )
    
    def create_stacked_model(self):
        """
        创建一个堆叠集成模型
        
        返回:
            StackingClassifier: 机器学习模型
        """
        # Base models
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('svm', SVC(probability=True, kernel='rbf')),
            ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
            ('lgb', lgb.LGBMClassifier(n_estimators=100, random_state=42))
        ]
        
        meta_classifier = LogisticRegression()
        
        stacked_model = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_classifier,
            cv=5
        )
        
        return stacked_model
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        综合模型训练和评估
        
        参数:
            X_train, X_test: 训练和测试特征矩阵
            y_train, y_test: 训练和测试标签
        
        返回:
            dict: 综合模型性能指标
        """

        model = self.create_stacked_model()
        
        # 使用 tqdm 包装 model.fit
        with tqdm(total=1, desc="Training Model") as pbar:
            model.fit(X_train, y_train)
            pbar.update(1)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        performance_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # 这里创建一个混淆矩阵（目的是评估性能）
        cm = confusion_matrix(y_test, y_pred)
        
        # ROC Curve
        plt.figure()
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {performance_metrics["roc_auc"]:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig('roc_curve.png')
        plt.close()
        
        # Cross-Validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=5))
        
        # Store results
        performance_metrics['cross_val_scores'] = cv_scores
        performance_metrics['confusion_matrix'] = cm
        
        return performance_metrics
    
    def calculate_model_contributions(self, X_train, X_test, y_train, y_test):
        """
        计算每个基模型的贡献
        
        参数:
            X_train, X_test: 训练和测试特征矩阵
            y_train, y_test: 训练和测试标签
        
        返回:
            dict: 每个基模型的贡献
        """
        model = self.create_stacked_model()
        
        # 使用 tqdm 包装 model.fit
        with tqdm(total=1, desc="Training Model") as pbar:
            model.fit(X_train, y_train)
            pbar.update(1)
        
        # 获取每个基模型的预测结果
        base_predictions = {}
        for name, estimator in model.named_estimators_:
            base_predictions[name] = estimator.predict_proba(X_test)[:, 1]
        
        # 计算每个基模型的贡献
        contributions = {}
        for name, pred in base_predictions.items():
            contributions[name] = roc_auc_score(y_test, pred)
        
        return contributions
    
    def ablation_study(self, X_train, X_test, y_train, y_test, model_names):
        """
        进行消融实验
        
        参数:
            X_train, X_test: 训练和测试特征矩阵
            y_train, y_test: 训练和测试标签
            model_names (list): 需要移除的基模型名称列表
        
        返回:
            dict: 消融实验结果
        """
        ablation_results = {}
        
        for name in model_names:
            base_models = [
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('svm', SVC(probability=True, kernel='rbf')),
                ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                ('lgb', lgb.LGBMClassifier(n_estimators=100, random_state=42))
            ]
            
            # 移除指定的基模型
            base_models = [model for model in base_models if model[0] != name]
            
            meta_classifier = LogisticRegression()
            
            stacked_model = StackingClassifier(
                estimators=base_models,
                final_estimator=meta_classifier,
                cv=5
            )
            
            # 使用 tqdm 包装 model.fit
            with tqdm(total=1, desc=f"Training Model without {name}") as pbar:
                stacked_model.fit(X_train, y_train)
                pbar.update(1)
            
            y_pred = stacked_model.predict(X_test)
            y_pred_proba = stacked_model.predict_proba(X_test)[:, 1]
            
            performance_metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            ablation_results[name] = performance_metrics
        
        return ablation_results
    
    def run_complete_analysis(self):
        """
        执行整个机器学习流水线
        
        返回:
            dict: 完整分析结果
        """

        print("Conducting Preliminary Data Analysis...")
        prelim_analysis = self.preliminary_analysis()
        
        print("Preprocessing Data...")
        X_train, X_test, y_train, y_test, feature_selector = self.preprocess_data()
        
        print("Training and Evaluating Model...")
        model_performance = self.train_and_evaluate(X_train, X_test, y_train, y_test)
        
        print("Calculating Model Contributions...")
        model_contributions = self.calculate_model_contributions(X_train, X_test, y_train, y_test)
        
        print("Running Ablation Study...")
        ablation_results = self.ablation_study(X_train, X_test, y_train, y_test, ['rf', 'svm', 'xgb', 'lgb'])
        
        final_results = {
            'preliminary_analysis': prelim_analysis,
            'model_performance': model_performance,
            'model_contributions': model_contributions,
            'ablation_results': ablation_results
        }
        
        return final_results

# Main Execution
if __name__ == '__main__':
    pipeline = AdvancedMLPipeline('charging_pile.csv')
    
    results = pipeline.run_complete_analysis()
    
    print("\n===== Preliminary Analysis =====")
    print("Descriptive Statistics:")
    print(results['preliminary_analysis']['descriptive_stats'])
    
    print("\n===== Model Performance =====")
    for metric, value in results['model_performance'].items():
        print(f"{metric}: {value}")
    
    print("\n===== Model Contributions =====")
    for model, contribution in results['model_contributions'].items():
        print(f"{model}: {contribution:.4f}")
    
    print("\n===== Ablation Study Results =====")
    for model, metrics in results['ablation_results'].items():
        print(f"Without {model}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
