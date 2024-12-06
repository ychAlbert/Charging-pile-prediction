from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt

def create_stacked_model():
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

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    综合模型训练和评估
    
    参数:
        X_train, X_test: 训练和测试特征矩阵
        y_train, y_test: 训练和测试标签
    
    返回:
        dict: 综合模型性能指标
    """

    model = create_stacked_model()
    
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
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=5))
    
    performance_metrics['cross_val_scores'] = cv_scores
    performance_metrics['confusion_matrix'] = cm
    
    return performance_metrics