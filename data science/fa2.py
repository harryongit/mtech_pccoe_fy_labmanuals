# ================================
# 1. Import Libraries
# ================================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ================================
# 2. Load Dataset
# ================================
df = pd.read_csv("train.csv")

# Convert Order Date to datetime (day first)
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)

# ================================
# 2b. Create Year and Month Columns
# ================================
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month

# ================================
# 3. Monthly Aggregation
# ================================
monthly_sales = df.groupby(
    ['Year', 'Month', 'Category', 'Region'], as_index=False
)['Sales'].sum()

# ================================
# 4. Feature Engineering
# ================================
# Lag features
monthly_sales['Lag_1M'] = monthly_sales.groupby(
    ['Category', 'Region']
)['Sales'].shift(1)

monthly_sales['Lag_12M'] = monthly_sales.groupby(
    ['Category', 'Region']
)['Sales'].shift(12)

# 3-month rolling mean using transform (safe for index)
monthly_sales['MA_3M'] = monthly_sales.groupby(
    ['Category', 'Region']
)['Sales'].transform(lambda x: x.rolling(3).mean())

# Drop rows with missing values due to lag/rolling
monthly_sales.dropna(inplace=True)

# ================================
# 5. Encode Categorical Variables
# ================================
le_category = LabelEncoder()
le_region = LabelEncoder()

monthly_sales['Category_Enc'] = le_category.fit_transform(monthly_sales['Category'])
monthly_sales['Region_Enc'] = le_region.fit_transform(monthly_sales['Region'])

# ================================
# 6. Prepare Features & Target
# ================================
X = monthly_sales[['Lag_1M', 'Lag_12M', 'MA_3M', 'Category_Enc', 'Region_Enc']]
y = monthly_sales['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# 7. Train Random Forest Model
# ================================
rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

# ================================
# 8. Feature Importance Plot
# ================================
importances = rf_model.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=True)  # ascending for horizontal bar

plt.figure(figsize=(8, 5))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")
plt.savefig("feature_importance.png", dpi=300, bbox_inches='tight')
plt.show()

print("feature_importance.png generated successfully!")
