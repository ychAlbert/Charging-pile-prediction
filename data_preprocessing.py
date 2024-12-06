from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

def preprocess_data(X, y, test_size=0.2, random_state=42):
    """
    使用多种缩放和特征选择技术进行高级数据预处理
    
    参数:
        X (DataFrame): 特征矩阵
        y (Series): 目标标签
        test_size (float): 测试数据的比例
        random_state (int): 随机种子以保证结果可重复
        
    返回:
        tuple: 处理后的训练和测试数据集
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
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