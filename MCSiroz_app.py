import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd

# Kategorik ve sayısal özniteliklerin listesi
categorical_columns = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
numeric_columns = ['N_Days', 'Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Stage']

# LabelEncoder'ı tüm kategorik sütunlara uygulayan fonksiyon
def apply_label_encoder(df):
    le = LabelEncoder()
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])
    return df

# LabelEncoder ile dönüşüm uygulayan bir FunctionTransformer
label_encoder_transformer = FunctionTransformer(apply_label_encoder, validate=False)

def model_fit_evaluation(models=[GradientBoostingClassifier(), XGBClassifier(), LGBMClassifier()], 
                         X=None, y=None, 
                         categorical_transformer='onehot', 
                         numeric_transformer='standard'):
    results = []
    best_model = None
    best_loss = float('inf')
    
    if categorical_transformer == 'label':
        cat_transformer = label_encoder_transformer
    else:
        cat_transformer = OneHotEncoder(handle_unknown='ignore')

    if numeric_transformer == 'minmax':
        num_transformer = MinMaxScaler()
    else:
        num_transformer = StandardScaler()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', cat_transformer, categorical_columns),
            ('numeric', num_transformer, numeric_columns)
        ]
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    for model in models:
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred_proba = pipeline.predict_proba(X_test)
        loss = log_loss(y_test, y_pred_proba)
        results.append({
            'Model': type(model).__name__,
            'Log Loss': loss
        })
        if loss < best_loss:
            best_loss = loss
            best_model = pipeline  # En iyi modeli pipeline olarak kaydet

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Log Loss')
    print(results_df)
    
    return best_model

# Veriyi hazırla
train = pd.read_csv("/kaggle/input/playground-series-s3e26/train.csv")
X = train.drop(columns=['Status'])
y = train['Status']
y = y.map({'C': 0, 'CL': 1, 'D': 2})

# Modelleri eğit ve değerlendir
best_pipeline = model_fit_evaluation(X=X, y=y, categorical_transformer='label', numeric_transformer='minmax')

# En iyi modeli ve ön işlemcileri kaydet
joblib.dump(best_pipeline, 'best_model_pipeline.pkl')
