import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

# 1. Fetch
adult = fetch_ucirepo(id=2)
X, y = adult.data.features, adult.data.targets.iloc[:, 0]

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Preprocessing
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
])

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# 4. Target Cleaning
y_train_proc = LabelEncoder().fit_transform(y_train.astype(str).str.replace('.', '', regex=False).str.strip())
y_test_proc = LabelEncoder().fit_transform(y_test.astype(str).str.replace('.', '', regex=False).str.strip())

# Salvar
np.savez('data_processed.npz', xt=X_train_proc, xv=X_test_proc, yt=y_train_proc, yv=y_test_proc)
print("Dados processados e salvos em 'data_processed.npz'.")