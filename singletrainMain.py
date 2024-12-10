import os
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
        
        # 创建结果文件夹
        self.result_folder = "result"
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)
    
    def preliminary_analysis(self):
        """
        初步数据分析
        
        返回:
            dict: 综合统计见解
        """
        print("Conducting Preliminary Data Analysis...")
        
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
        
        # 保存特征分布图
        plt.figure(figsize=(15, 10))
        self.X.hist(bins=30, figsize=(15, 10))
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_folder, 'feature_distributions.png'))
        plt.close()
        
        # 保存相关性热图
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_folder, 'correlation_heatmap.png'))
        plt.close()
        
        self.analysis_results['descriptive_stats'] = desc_stats
        self.analysis_results['normality_tests'] = normality_tests
        
        # 保存初步分析结果到 txt 文件
        with open(os.path.join(self.result_folder, 'preliminary_analysis.txt'), 'w') as f:
            f.write("===== Descriptive Statistics =====\n")
            f.write(str(desc_stats))
            f.write("\n\n===== Normality Tests =====\n")
            f.write(str(normality_tests))
        
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
        print("Preprocessing Data...")

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
        print("Training and Evaluating Model...")

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
        
        # 保存混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(self.result_folder, 'confusion_matrix.png'))
        plt.close()
        
        # 保存 ROC 曲线
        plt.figure()
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {performance_metrics["roc_auc"]:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig(os.path.join(self.result_folder, 'roc_curve.png'))
        plt.close()
        
        # 保存性能指标到 CSV 文件
        metrics_df = pd.DataFrame([performance_metrics])
        metrics_df.to_csv(os.path.join(self.result_folder, 'model_performance.csv'), index=False)
        
        return performance_metrics
    
    def evaluate_individual_models(self, X_train, X_test, y_train, y_test):
        """
        对每个单个模型进行测试性能评估并绘图
        
        参数:
            X_train, X_test: 训练和测试特征矩阵
            y_train, y_test: 训练和测试标签
        """
        print("Evaluating Individual Models...")

        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, kernel='rbf'),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42)
        }
        
        for model_name, model in models.items():
            print(f"Training and Evaluating {model_name}...")
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # 计算性能指标
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            # 保存性能指标到 CSV 文件
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(os.path.join(self.result_folder, f'{model_name}_performance.csv'), index=False)
            
            # 保存 ROC 曲线
            plt.figure()
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["roc_auc"]:.2f})')
            plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
            plt.title(f'{model_name} ROC Curve')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            plt.savefig(os.path.join(self.result_folder, f'{model_name}_roc_curve.png'))
            plt.close()
    
    def run_complete_analysis(self):
        """
        执行整个机器学习流水线
        
        返回:
            dict: 完整分析结果
        """
        prelim_analysis = self.preliminary_analysis()
        
        X_train, X_test, y_train, y_test, feature_selector = self.preprocess_data()
        
        model_performance = self.train_and_evaluate(X_train, X_test, y_train, y_test)
        
        self.evaluate_individual_models(X_train, X_test, y_train, y_test)
        
        final_results = {
            'preliminary_analysis': prelim_analysis,
            'model_performance': model_performance
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
