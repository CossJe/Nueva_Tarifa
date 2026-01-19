# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 13:32:53 2025

@author: Jesus Coss
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import model_from_json
from datetime import timedelta

import statsmodels.formula.api as smf

from src.dynamic_pricing_data_loader import cargar_y_preparar_datos

from Tools.ExtractData import GetDataFromETL
from Tools.Tools4TrainRN import RedNeuronal
from Tools.Tools4ProofRN import ProofRedNeuronal
from Tools.Tools4Elasticity import Elasticity
from Tools.Tools4Clasify import K_means
from Tools.Tools4ClasSuper import ClusteringSupervisado
from Tools.Tools4ProofCluster import PredictorClusters

    

Data= GetDataFromETL(False)

#Df_ClusSuper= Data.D4_ClusteringSuper()

# Entrena modelo ClusteringSupervisado
#clustering = ClusteringSupervisado()


# Crear predictor con el modelo entrenado
#predictor = PredictorClusters(Df_ClusSuper)
#resultados= predictor.Get_values()



"""

#Elasticidad
Df_Elas= Data.D4_Elasticity() # Extraccion de los datos para elasticidad

Elas4OriDest= Elasticity(Df_Elas).Df


Df_TRN= Data.D4_Train_NN() # Extraccion de datos para entrenar la Red Neuronal
Df_PRN= Data.D4_Proof_NN() # Extraccion de datos para probar la Red Neuronal

RN= RedNeuronal(Df_TRN) # Clase para entrenar la red

# Dataframe de pronosticos de nuevos clientes
lista_resultados = []
for row in range(len(Df_PRN)):
    InfoClient = pd.DataFrame(Df_PRN.iloc[row]).T
    PRN=ProofRedNeuronal(InfoClient)# Clase para pronosticar nuevos con la red
    InfoClient["TARIFA DINAMICA"]= PRN.PrecioDin[0,0]
    lista_resultados.append(InfoClient)

DataTD = pd.concat(lista_resultados, ignore_index=True)

# clustering no supervisado
Df_Cluster= Data.D4_ClusterNS() # Extraccion de los datos para el clustering no supervisado
# Uso de la clase para el clustering no supervisado
kmeans = K_means()
DF_K_means = kmeans.fit(Df_Cluster,optimal_k=5,aplicar_fik=False)


"""