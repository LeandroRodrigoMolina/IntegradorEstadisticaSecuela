import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson


# Cargaremos el archivo Excel
file_path_excel = 'dataset_empleados.xlsx'
# Utilizamos Pandas para cargar el archivo Excel
df_empleados = pd.read_excel(file_path_excel)

################################################
#                                              #
#          --- Prueba de Hipótesis ---         #
#                                              #
################################################

# Contar el número de mujeres en la muestra
num_mujeres = df_empleados['Genero'].str.strip().str.lower().value_counts().get('femenino', 0)
total_empleados = len(df_empleados)

# Proporción observada de mujeres en la muestra
proporcion_mujeres = num_mujeres / total_empleados

# Nueva hipótesis nula
p_h0 = 0.32

# Realizar el test de hipótesis con la nueva hipótesis nula
# Usando 'greater' como alternativa porque H1: p > 0.32
# Realizar el test de hipótesis con la nueva hipótesis nula usando 'binomtest'
resultado_test = binomtest(num_mujeres, total_empleados, p_h0, alternative='greater')

# Obtener el valor p del resultado
p_value_nueva = resultado_test.pvalue

# Imprimir los resultados
print("Numero de mujeres en la muestra:", num_mujeres)
print("Total de empleados en la muestra:", total_empleados)
print("Proporcion observada de mujeres:", proporcion_mujeres)
print("Valor p del test de hipotesis:", p_value_nueva)

print("-----------------------------------------------------")

################################################
#                                              #
#    --- Regresión y Correlación Lineal ---    #
#                                              #
################################################

# REGRESIÓN LINEAL SIMPLE

# Examinando el cumplimiento de los supuestos

# Preparando los datos para la regresión lineal solo con 'Edad' y 'Salario'
X_age_salary = df_empleados[['Edad']]
y_salary = df_empleados['Salario']

# Añadir una constante a la variable independiente 'Edad'
X_age_salary = sm.add_constant(X_age_salary)

# Crear el modelo de regresión lineal solo con 'Edad' y 'Salario'
model_age_salary = sm.OLS(y_salary, X_age_salary).fit()

# Calcular los residuos para este modelo
residuals_age_salary = model_age_salary.resid

# 1. Verificar la linealidad
# Gráfico de dispersión de 'Edad' vs 'Salario'
plt.figure(figsize=(6, 4))
sns.scatterplot(x=df_empleados['Edad'], y=df_empleados['Salario'])
plt.title('Edad vs Salario')
plt.show()

# 2. Independencia de los errores (Prueba de Durbin-Watson para este modelo)
dw_test_age_salary = durbin_watson(residuals_age_salary)

# 3. Homocedasticidad
# Gráfico de residuos vs valores ajustados para este modelo
plt.figure(figsize=(6, 4))
plt.scatter(model_age_salary.fittedvalues, residuals_age_salary)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Valores Ajustados')
plt.ylabel('Residuos')
plt.title('Residuos vs Valores Ajustados para Edad-Salario')
plt.show()

# 4. Normalidad de los errores
# Gráfico Q-Q para los residuos de este modelo
plt.figure(figsize=(6, 4))
sm.qqplot(residuals_age_salary, line ='45')
plt.title('Gráfico Q-Q de los Residuos (Edad-Salario)')
plt.show()

# Resultados
model_summary_age_salary = model_age_salary.summary()
print("Resultado del prueba de Durbin-Watson",dw_test_age_salary)

# ======================================================
# Calculando y Graficando el modelo de Regresión

# Preparación de los datos para el modelo de regresión lineal
X = df_empleados['Edad']  # Variable independiente
y = df_empleados['Salario']  # Variable dependiente

# Agregar una constante a la variable independiente
X = sm.add_constant(X)

# Crear el modelo de regresión lineal
modelo = sm.OLS(y, X).fit()

# Obtenemos los valores de la edad y los salarios predichos por el modelo
X_valores = df_empleados['Edad']
y_pred = modelo.predict()

# Gráfico de dispersión de los datos reales
plt.scatter(X_valores, y, label='Datos Reales', color='blue')

# Línea de tendencia del modelo de regresión
plt.plot(X_valores, y_pred, label='Línea de Regresión', color='red')

# Etiquetas y leyenda
plt.title('Relación entre Edad y Salario')
plt.xlabel('Edad')
plt.ylabel('Salario')
plt.legend()

plt.show()

print("-----------------------------------------------------")

################################################
#                                              #
#        --- Intervalo de Confianza ---        #
#                                              #
################################################

# INTERVALO DE CONFIANZA DE LAS HORAS DE TRABAJO

# Extraemos la columna de horas de trabajo del DataFrame
horas_trabajo = df_empleados['Horas_Trabajo']

# Calculamos la media y la desviación estándar de las horas de trabajo
media_horas = np.mean(horas_trabajo)
std_horas = np.std(horas_trabajo, ddof=1)  # ddof=1 para muestra

# Número de observaciones
n = len(horas_trabajo)

# Calculamos el intervalo de confianza del 95% para la media
confianza = 0.95
sem = std_horas / np.sqrt(n)
intervalo_confianza = stats.norm.interval(confianza, loc=media_horas, scale=sem)

print("intervalo de confianza para las horas de trabajo", intervalo_confianza)

print("-----------------------------------------------------")

# INTERVALO DE CONFIANZA DE PARA LOS SUELDOS DE LOS LICENCIADOS

# Filtrando los datos para obtener solo los salarios de los licenciados
salarios_licenciados = df_empleados[df_empleados['Nivel_Educacion'] == 'Licenciatura']['Salario']

# Aplicando la transformación logarítmica
salarios_transformados = np.log(salarios_licenciados)

# Visualizando la distribución de los salarios transformados
plt.figure(figsize=(10, 6))
plt.hist(salarios_transformados, bins=10, edgecolor='black', color='lightgreen')
plt.title('Distribución de Salarios Transformados (Logarítmica)')
plt.xlabel('Logaritmo del Salario')
plt.ylabel('Frecuencia')
plt.show()

# Calculando la media y la desviación estándar de los salarios transformados
media_transformada = salarios_transformados.mean()
std_transformada = salarios_transformados.std()

# Calculando el intervalo de confianza del 95% para la media transformada
intervalo_confianza_transformado = stats.t.interval(confianza, df=len(salarios_licenciados)-1, loc=media_transformada, scale=std_transformada / np.sqrt(n))

# Transformando de nuevo el intervalo de confianza a la escala original de salarios
intervalo_confianza_original = np.exp(intervalo_confianza_transformado)

print("Intervalo de confianza para el salario medio de los Licenciados", intervalo_confianza_original)

print("-----------------------------------------------------")