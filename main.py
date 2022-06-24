import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier
import missingno as msno
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTEN
from sklearn import tree
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score


df = pd.read_csv("glass.data", header=None)

# veri setinni ilk 5 elemanını getiriyor
df.head(5)

# veri setinde sütun isimleri tanımlı değildi biz tanımlıyoruz.
# bu değişkenler glass.names içinden bulunmuştur.
df.columns = ["Id", "RI", "Na", "Mg", "Al",
              "Si", "K", "Ca", "Ba", "Fe", "Type"]
print(df.columns)
df.head(5)


df.info()
df.describe().T

print(df["Id"].value_counts)  # ID değişkeni silinecek
df = df.drop("Id", axis=1)  # Id değişkenini düşürdük.

df.corr()  # Korelasyon matrisini gösterir


msno.bar(df)  # Eksik veri gözlemlenmiyor
plt.show()

msno.heatmap(df)  # Eksik değerlerin korelasyonu gözlemlendi -yok
plt.show()

cat_cols = ["Type"]
num_cols = ["RI", "Na", "Mg", "Al",
            "Si", "K", "Ca", "Ba", "Fe"]

df.isnull().sum()  # null olan değerlerin sayısı 0'dır yani bu adımı geçebiliriz


# ayrik değerleri baskılayan fonksiyon
def aykiri_deger_baskilama(df, df_without_outliers):
    for i in df.columns:
        tenth_percentile = np.percentile(df[i], 10)
        ninetieth_percentile = np.percentile(df[i], 90)
        df_without_outliers[i] = np.where(
            df[i] < tenth_percentile, tenth_percentile, df[i])
        df_without_outliers[i] = np.where(
            df_without_outliers[i] > ninetieth_percentile, ninetieth_percentile, df_without_outliers[i])


df.describe().T


plt.figure(figsize=(12, 12))
sns.heatmap(df[num_cols].corr(), annot=True)
plt.show()

# Bütün Değişkenlerin Distplot Grafiğini Oluşturan for
for col in num_cols:
    sns.distplot(df[col])
    plt.show()

# Bütün Değişkenlerin BoxPlot Grafiğini Oluşturan for
for col in num_cols:
    sns.boxplot(df[col])
    plt.show()

df_without_outliers = df.copy()
aykiri_deger_baskilama(df, df_without_outliers)
df = df_without_outliers.copy()

# Ayrıkı Değerler baskılandıktan boxPlot grafiği
for col in num_cols:
    sns.boxplot(df[col])
    plt.show()


for col in cat_cols:
    sns.countplot(x=col, data=df)
    plt.show()


x = df.drop("Type", axis=1)
y = df["Type"]
df["Type"].value_counts()


sampler = SMOTEN(random_state=0)
X_res, y_res = sampler.fit_resample(x, y)

x_train, x_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.20, random_state=42)

x_train.shape, x_test.shape, y_train.shape, y_test.shape

# over sampling uygulandıktan sonra sınıfların veri sayıları
y_res.value_counts()

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

preds = model.predict(x_test)
confusion_matrix(y_test, preds)
print(classification_report(y_test, preds))

f1_score(y_true=y_test, y_pred=preds, average='weighted')
precision_score(y_true=y_test, y_pred=preds, average='weighted')
recall_score(y_true=y_test, y_pred=preds, average='weighted')
accuracy_score(y_true=y_test, y_pred=preds)

params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"]
}

best_model = GridSearchCV(model, params, cv=10, n_jobs=-1, verbose=False)
best_model.fit(x_train, y_train)

final_model = best_model.best_estimator_
final_model.fit(x_train, y_train)
cv_results = cross_validate(
    final_model, x_train, y_train, cv=10, scoring=["accuracy"])
print(cv_results["test_accuracy"].mean())


final_preds = final_model.predict(x_test)
print(classification_report(y_test, final_preds))

f1_score(y_true=y_test, y_pred=final_preds, average='weighted')
precision_score(y_true=y_test, y_pred=final_preds, average='weighted')
recall_score(y_true=y_test, y_pred=final_preds, average='weighted')
accuracy_score(y_true=y_test, y_pred=final_preds)


plt.figure(figsize=(12, 12))
tree.plot_tree(final_model)
plt.show()


feature_imp = pd.DataFrame(
    {'Value': final_model.feature_importances_, 'Feature': x_train.columns})
plt.figure(figsize=(10, 10))
sns.set(font_scale=1)
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(
    by="Value", ascending=False)[0:10])
plt.title('Features')
plt.tight_layout()
plt.show()
