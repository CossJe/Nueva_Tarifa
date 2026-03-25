# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 17:42:59 2026

@author: Jesus Coss
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import os # Solo si decides descomentar la parte final de guardado de archivos

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

class ClusteringSupervisado:
    def __init__(self):
        self.test_size = 0.2
        self.random_state = 42
        self.NomArch = "modelo_xgboost_clientes.json"
        self.ruta_principal = os.getcwd()
        self.csv_path = os.path.join(self.ruta_principal, "Files", "DB_ClientesAgrupados.csv")
        self.config_path = os.path.join(self.ruta_principal, "Models", self.NomArch)
        self.DF_K_means = pd.read_csv(self.csv_path)
        self.Fit()
        
    def Fit(self):
        # 1. Limpieza y Definición de X y y
        X = self.DF_K_means.drop(['EMAIL', 'Cluster'], axis=1) 
        y = self.DF_K_means['Cluster']
        num_clases_final = y.nunique()

        # 2. Dividir los datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )
     
        # 3. Entrenamiento del Modelo XGBoost
        xgb_model_baseline = XGBClassifier(
            objective='multi:softmax',
            num_class=num_clases_final,
            eval_metric='mlogloss',
            n_estimators=100,
            random_state=self.random_state
        )
        
        xgb_model_baseline.fit(X_train, y_train)
        y_pred = xgb_model_baseline.predict(X_test)
        """
        # --- 4. Evaluación y Comparación ---
        score = accuracy_score(y_test, y_pred)
        print(f"\n--- Evaluación del Modelo ---")
        print(f"Accuracy Score: {score:.4f}")
        print("\nReporte de Clasificación:")
        print(classification_report(y_test, y_pred))

        # Crear un DataFrame para comparar Real vs Predicho
        df_comparativo = pd.DataFrame({
            'Valor Real': y_test.values,
            'Predicción': y_pred
        })
        
        print("\nPrimeras 10 comparaciones:")
        print(df_comparativo.head(10))
        """
        
        # Guardado del modelo
        xgb_model_baseline.save_model(self.config_path)
        print(f"\nModelo XGBoost guardado exitosamente como: {self.NomArch}")
        
        return 
        
