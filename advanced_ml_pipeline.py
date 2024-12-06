import pandas as pd
from data_preprocessing import preprocess_data
from model_training import create_stacked_model, train_and_evaluate
from evaluation import preliminary_analysis

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
    
    def run_complete_analysis(self):
        """
        执行整个机器学习流水线
        
        返回:
            dict: 完整分析结果
        """

        print("Conducting Preliminary Data Analysis...")
        prelim_analysis = preliminary_analysis(self.X)
        
        print("Preprocessing Data...")
        X_train, X_test, y_train, y_test, feature_selector = preprocess_data(self.X, self.y)
        
        print("Training and Evaluating Model...")
        model_performance = train_and_evaluate(X_train, X_test, y_train, y_test)
        
        final_results = {
            'preliminary_analysis': prelim_analysis,
            'model_performance': model_performance
        }
        
        return final_results