import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'x1': [50, 50, 50, 60, 60, 60, 70, 70, 70, 80, 80, 80],  # Sıcaklık (°C)
    'x2': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],              # H2:O2 oranı (hacimce)
    'y': [221, 227.67, 233.75, 262.46, 256.83, 239.06, 291.69, 289.83, 271.67, 280.38, 284.94, 280.95]  # Güç yoğunluğu
}
df = pd.DataFrame(data)

X = df[['x1', 'x2']]
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(name, model, X_t, y_t, original_X=None):
    y_pred = model.predict(X_t)
    mse = mean_squared_error(y_t, y_pred)
    r2 = r2_score(y_t, y_pred)

    print(f"\n{name}")
    print(f"  MSE: {mse:.4f}")
    print(f"  R² : {r2:.4f}")
    print("  Tahminler:")
    base_X = original_X if original_X is not None else pd.DataFrame(X_t, columns=['x1', 'x2'])
    results_df = base_X.copy()
    results_df['Gerçek y'] = y_t.values
    results_df['Tahmin y'] = y_pred
    print(results_df.to_string(index=False))

    return mse, r2

# ----------------------------
# Model Eğitimi ve Değerlendirme
# ----------------------------
model_scores = {}
best_tree_model = None
best_tree_mse = float('inf')
best_tree_name = None


# Lineer
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
mse, _ = evaluate_model("Lineer Regresyon", lin_model, X_test, y_test)
model_scores['Lineer'] = mse

# Polinom
poly = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
mse, _ = evaluate_model("Polinomik Regresyon (d=3)", poly_model, X_test_poly, y_test, original_X=X_test)
model_scores['Polinom'] = mse

# Karar ağaçları
for d in range(1, 6):
    tree = DecisionTreeRegressor(max_depth=d, random_state=42)
    tree.fit(X_train, y_train)
    mse, _ = evaluate_model(f"Karar Ağacı Regresyon (derinlik={d})", tree, X_test, y_test)
    model_scores[f'Ağaç d={d}'] = mse
    if mse < best_tree_mse:
        best_tree_model = tree
        best_tree_mse = mse
        best_tree_name = f'Ağaç d={d}'
        
# ----------------------------
# En iyi model ile tüm veri setinde tahmin
# ----------------------------

best_model_name = min(model_scores, key=model_scores.get)
#best_model_name = "Lineer"
if best_model_name == 'Lineer':
    final_model = lin_model
    best_model_type = "linear"
    y_model = final_model.predict(X)
elif best_model_name == 'Polinom':
    final_model = poly_model
    best_model_type = "poly"
    X_all_poly = poly.transform(X)
    y_model = final_model.predict(X_all_poly)
else:
    final_model = best_tree_model
    best_model_type = "tree"
    y_model = final_model.predict(X)

print(f"\nEn iyi model: {best_model_name}")
print("\nx1  x2  y_deneysel  y_model")
for i in range(len(X)):
    print(f"{X.iloc[i,0]:>2} {X.iloc[i,1]:>3}   {y.iloc[i]:>9.2f}   {y_model[i]:>8.2f}")

# ----------------------------
# Bayesian optimizasyon
# ----------------------------
from bayes_opt import BayesianOptimization # type: ignore

def hedef_fonksiyon(x1, x2):
    x_input = pd.DataFrame([[x1, x2]], columns=['x1', 'x2'])
    if best_model_type == "poly":
        x_input = poly.transform(x_input)
    return final_model.predict(x_input)[0]

optimizer = BayesianOptimization(
    f=hedef_fonksiyon,
    pbounds={"x1": (50, 80), "x2": (1, 3)},
    random_state=42
)

optimizer.maximize(init_points=5, n_iter=30)

optimum = optimizer.max
x1_opt = round(optimum["params"]["x1"], 1)
x2_opt = round(optimum["params"]["x2"], 1)
y_opt = round(optimum["target"], 2)

# Optimizer sonrası aynı maksimum hedef değere sahip olanlar içinden en küçük x1 ve x2'yi seç

# Tüm sonuçlar
all_results = pd.DataFrame(optimizer.res)

# En yüksek tahmin edilen güç yoğunluğu
max_y = all_results['target'].max()

# Bu maksimum değere sahip tüm noktalar
max_rows = all_results[all_results['target'] == max_y]

# x1 ve x2 değerleri en küçük olanı seç (önce x1'e, sonra x2'ye göre)
best_row = max_rows.sort_values(by=['params'], key=lambda x: x.apply(lambda d: (d['x1'], d['x2']))).iloc[0]

x1_opt = round(best_row['params']['x1'], 1)
x2_opt = round(best_row['params']['x2'], 1)
y_opt = round(best_row['target'], 2)

print(f"\nBayesian Optimizasyon Sonucu (Düşük x1/x2 tercihli):")
print(f"En yüksek tahmin edilen güç yoğunluğu: {y_opt} mW/cm²")
print(f"Bu değer, x1 = {x1_opt} °C ve x2 = {x2_opt} oranında elde edilmiştir.")
