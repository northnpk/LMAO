import lightgbm as lgb

clf = lgb.LGBMClassifier()

def fit(model, X, y, val_X:None, val_y:None):
    return model.fit(X, y)

def pred(model, X):
    return model.predict(X)