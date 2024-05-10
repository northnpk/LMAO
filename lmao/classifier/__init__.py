from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report, f1_score
from ..classifier import lgbm

model_map = {
    'lgbm': lgbm.clf
}

fit_map = {
    'lgbm': lgbm.fit
}

pred_map = {
    'lgbm': lgbm.pred
}

label_map = {
    'lgbm':None
}

class LMAOClassifier:
    def __init__(self, model_type:str='lgbm'):
        super().__init__()
        self.model = None
        self.model = model_map[model_type]
        self.label = label_map[model_type]
        self.model_type = model_type
    
    def fit(self, X, y, val_X:None, val_y:None):
        self.model = fit_map[self.model_type](self.model,
                                              X=X, y=y,
                                              val_X=val_X, val_y=val_y)
        
    def predict(self, X):
        return pred_map[self.model_type](self.model, X=X)
    
    def confusion_matrix_plot(self, X, y):
        pred = self.predict(X)
        ConfusionMatrixDisplay.from_predictions(y_true=y, y_pred=pred, display_labels=self.label)
        
    def eval_report(self, X, y):
        pred = self.predict(X)
        return classification_report(y_true=y, y_pred=pred, digits=3, target_names=self.label)
    
    def confusion_report(self, X, y):
        pred = self.predict(X)
        return confusion_matrix(y_true=y, y_pred=pred, labels=self.label)
    
    def f1_score(self, X, y):
        return f1_score(X, y, average='weighted')
    
    def set_label(self, labels):
        self.label = labels
        print(f'Set labels to : {labels}')