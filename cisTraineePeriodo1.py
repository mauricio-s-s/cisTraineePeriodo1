"""
CIS - Trainee
Maurício S. Silva


Dataset escolhido :"Weather in Szeged 2006-2016"
Link do Dataset: https://www.kaggle.com/datasets/budincsevity/szeged-weather


Este projeto de Machine Learning foi desenvolvido com o objetivo de
prever a temperatura em determinado local, a partir de dados como
humidade, visibilidade, velocidade do ar e etc, contidos no Dataset.
 
"""



import os
import pandas as pd
import numpy as np

from pandas.plotting import scatter_matrix


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score


import matplotlib.pyplot as plt


pcaReduction = False



csv_path = os.path.join(os.getcwd(), "weatherHistory.csv")
wt = pd.read_csv(csv_path)




#Preprocessamento

#Retirada das features "daily summary e temperatura aparente
# e loud cover
    # a temperatura aparente é severamente correlacionada ao valor
    # buscado da temperatura em si.



wt = wt[['Temperature (C)',
         'Formatted Date',
         'Summary', 
         'Precip Type', 
         #'Apparent Temperature (C)', 
         'Humidity',
         'Wind Speed (km/h)',
         'Wind Bearing (degrees)',
         'Visibility (km)',
         #'Loud Cover',             
         'Pressure (millibars)'
         #,'Daily Summary'
         ]]

#Deixar somente mês da data
#Deliberadamente desconsiderando o dia e o ano.
wt['Formatted Date'] = wt['Formatted Date'].apply(lambda y: int(y[5:7]))


wt_num = wt[[#"Temperature (C)",
             "Humidity",
             "Wind Speed (km/h)",
             "Wind Bearing (degrees)",
             "Visibility (km)",
             "Pressure (millibars)"]]

#Não há valores nulos
#caso houvesse, a abordagem utilizada seria deletar as instâncias,
#tendo em vista se tratar de um dataset de quantidade considerável
#de instâncias


#Verificando correlações

corr_matrix = wt.corr()
corr_matrix["Temperature (C)"].sort_values(ascending=False)
wt.plot(kind="scatter",x = "Humidity",y="Temperature (C)")

attributes = ["Temperature (C)", 
              "Humidity",
              "Visibility (km)",
              "Pressure (millibars)"]

scatter_matrix(wt[attributes], figsize=(12, 8))



#Repatir dataset entre Train set e Test sets
train_set, test_set = train_test_split(wt, test_size=0.2, random_state=88)


#Separar features preditoras e os labes, para não aplicar
#as transformações nos valores alvos (temperatura)
wt = train_set.drop("Temperature (C)", axis=1)
wt_labels = train_set["Temperature (C)"].copy()



#Lidando com atributos textuais e categóricos




#Transformation Pipelines


#### Pipeline valores numéricos

num_pipeline = Pipeline([
    ('std_scaler', StandardScaler()) #Feature Scaling
    ])
wt_num_tr = num_pipeline.fit_transform(wt_num)
num_attribs = list(wt_num)
cat_attribs = ["Summary","Precip Type"]


#Pipeline Total
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    #("cat", OneHotEncoder(), [cat_attribs]),
    ("cat-precType", OneHotEncoder(), ["Precip Type"]),
    ("cat-summary", OrdinalEncoder(), ["Summary"])
    ])

#utilizar oneHotEncoder no atributo "Summary" gera muitas 
#colunas adicionais, motivo pelo qual optou-se por utilizar
#ordinalEncoder.

wt_prepared = full_pipeline.fit_transform(wt)



#Redução de dimensionalidade - PCA

nums = np.arange(9)
var_ratio = []
chosenD = 0
for num in nums:
  pca = PCA(n_components=num)
  pca.fit(wt_prepared)
  cumsum = np.sum(pca.explained_variance_ratio_)
  var_ratio.append(cumsum)
  if cumsum >= 0.95 and chosenD== 0:
      chosenD = num
  
  

plt.figure(figsize=(4,2),dpi=150)
plt.grid()
plt.plot(nums,var_ratio,marker='o')
plt.xlabel('n_components')
plt.ylabel('Explained variance ratio')
plt.title('n_components vs. Explained Variance Ratio')

#redução de dimensionalidade:
if pcaReduction:
    pca = PCA(n_components=chosenD)
    wt_prepared = pca.fit_transform(wt_prepared)

#A redução se mostrou contraproducente, pois foi observado
#um aumento nos erros quadráticos médios



#Testar diferentes modelos

##Regressão Linear

lin_reg = LinearRegression()
lin_reg.fit(wt_prepared, wt_labels)


wt_predictions = lin_reg.predict(wt_prepared)
lin_mse = mean_squared_error(wt_labels, wt_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
#5.939053805567509 (Sem PCA)
#7.412732440225006 (Com PCA)

##Decision Tree

tree_reg = DecisionTreeRegressor()
tree_reg.fit(wt_prepared, wt_labels)

wt_predictions = tree_reg.predict(wt_prepared)
tree_mse = mean_squared_error(wt_labels, wt_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
#0.005631498996448093


#Dividir dataset entre dataset de treinamento e dataset de validação
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    

scores = cross_val_score(tree_reg, wt_prepared, wt_labels,
                         scoring="neg_mean_squared_error", 
                         cv=10)
tree_rmse_scores = np.sqrt(-scores)


print()
display_scores(tree_rmse_scores)


lin_scores = cross_val_score(lin_reg, wt_prepared, wt_labels,
                             scoring="neg_mean_squared_error",
                             cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)


print()
display_scores(lin_rmse_scores)


#Avaliar Modelo escolhido com testSet
final_model = lin_reg

X_test = test_set.drop("Temperature (C)", axis=1)
y_test = test_set["Temperature (C)"].copy()
X_test_prepared = full_pipeline.transform(X_test)


#redução:
if pcaReduction:
    pcaTest = PCA(n_components=chosenD)
    X_test_prepared = pcaTest.fit_transform(X_test_prepared)


final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse
#5.960691434165934 (Sem PCA)
#7.378412112559818 (Com PCA)








