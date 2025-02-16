# %% [markdown]
# ## Importar librerias y definición de la ruta (path)

# %%
import pandas as pd
import numpy as np
import os
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

# %%
# Ingresa la ruta donde está el repositorio
ruta = 'c:/repo_remoto/'

# %% [markdown]
# # Limpieza y Preprocesamiento de Datos con Python

# %% [markdown]
# ### Cargar archivos

# %%
### Características Equipos

equipos = pd.read_csv(ruta + 'etapa2/data/Caracteristicas_Equipos.csv')
equipos_df = pd.DataFrame(equipos)

# %%
### Historicos Ordenes

ordenes = pd.read_csv(ruta + 'etapa2/data/Historicos_Ordenes.csv')
ordenes_df = pd.DataFrame(ordenes)

# %%
### Registros Condiciones

condiciones = pd.read_csv(ruta + 'etapa2/data/Registros_Condiciones.csv')
condiciones_df = pd.DataFrame(condiciones)

# %% [markdown]
# ### LIMPIEZA Y PERFILADO DE CARACTERISTICAS EQUIPOS

# %%
# Mostrar las primeras filas de cada dataset
equipos_df.head()

# %%
# Revisar info del df
equipos_df.info()

# %%
# Datos vacios
print('Datos vacios en Caracteristicas_Equipos.csv\n',equipos_df.isna().sum())

# %%
# Extremos
z_scores = (equipos_df-equipos_df.mean(numeric_only=True)) / \
    equipos_df.std(numeric_only=True)
z_scores_abs = z_scores.apply(np.abs)
print(tabulate(z_scores_abs, headers='keys'))

# %%
umbral = 3

out_mask = ~z_scores[z_scores_abs > umbral].isna()
print('\nOutliers per column:\n')
print(out_mask.sum())

# %%
# Mostrar todas las filas duplicadas
print('\nSumatorio duplicados en sales', equipos_df.duplicated().sum())
equipos_df[equipos_df.duplicated(keep=False)]

# %%
# Elimino filas duplicadas

equipos_df_clean = equipos_df.drop_duplicates(keep='first')
print('\nSumatorio duplicados en equipos_df:', equipos_df_clean.duplicated().sum())


# %%
equipos_df_clean

# %%
equipos_df_clean.info()

# %%
# Cardinalidad

# Calcular el sumatorio de valores para cada columna
sum_value_equipos = equipos_df_clean.count()

# Calcular el sumatorio de valores únicos para cada columna
unique_sum_equipos = equipos_df_clean.nunique()

# Crear un nuevo DataFrame para mostrar ambos sumatorios
result_df_equipos = pd.DataFrame({
    'Sumatorio de valores': sum_value_equipos,
    'Sumatorio de valores únicos': unique_sum_equipos
})

print(result_df_equipos)

# %%
# Columnas a 'category'

equipos_df_clean[['Tipo_Equipo', 'Fabricante', 'Modelo']] = equipos_df_clean[[
    'Tipo_Equipo', 'Fabricante', 'Modelo']].astype('category')

equipos_df_clean.info()

# %%
equipos_df_clean.describe()

# %%
equipos_df_clean.describe(include='category')

# %%
# Existen equipos con Potencia menor de 0

print("\nEquipos con potencia <0 kW:", len(equipos_df_clean[equipos_df_clean['Potencia_kW'] < 0]))
equipos_df_clean[equipos_df_clean['Potencia_kW'] < 0]

# %%
# Busco filas que tengan el mismo Tipo_Equipo + Fabricante + Modelo que la fila con Potencia_kW <0 y además muestro al final su sumatorio

# Filtrar filas con Costo_Mantenimiento vacío
filas_pot = equipos_df_clean[equipos_df_clean['Potencia_kW'] < 0]

# Inicializar contador de coincidencias
pot_coincidencias = 0

# Buscar coincidencias en todo el DataFrame
for index, row in filas_pot.iterrows():
    coincidencias4 = equipos_df_clean[(equipos_df_clean['Tipo_Equipo'] == row['Tipo_Equipo']) &
                                      (equipos_df_clean['Fabricante'] == row['Fabricante']) & 
                               (equipos_df_clean['Modelo'] == row['Modelo'])]
    if len(coincidencias4) > 1:
        print(tabulate(coincidencias4, headers='keys'))
        pot_coincidencias += 1

# Mostrar el sumatorio de las filas NaN que han coincidido
print(f'\nSumatorio de filas NaN que han coincidido: {pot_coincidencias}')

# %%
# Reemplazo el valor de Potencia_kW < 0 por la media de las fils con mismo Tipo_Equipo + Fabricante + Modelo

# Filtrar filas con Potencia_kW < 0
filas_pot_negativas = equipos_df_clean[equipos_df_clean['Potencia_kW'] < 0]

# Reemplazar valores negativos por la media de sus valores filtrados
for index, row in filas_pot_negativas.iterrows():
    # Filtrar filas con el mismo Tipo_Equipo, Fabricante y Modelo
    coincidencias = equipos_df_clean[(equipos_df_clean['Tipo_Equipo'] == row['Tipo_Equipo']) &
                                     (equipos_df_clean['Fabricante'] == row['Fabricante']) &
                                     (equipos_df_clean['Modelo'] == row['Modelo']) &
                                     (equipos_df_clean['Potencia_kW'] >= 0)]
    if not coincidencias.empty:
        # Calcular la media de Potencia_kW para las coincidencias
        media_potencia = coincidencias['Potencia_kW'].mean()
        # Reemplazar el valor negativo con la media calculada
        equipos_df_clean.at[index, 'Potencia_kW'] = media_potencia

# Mostrar el sumatorio de las filas con Potencia_kW <0
print(f'\nSumatorio de filas con Potencia_kW < 0 kW: {len(equipos_df_clean[equipos_df_clean["Potencia_kW"] < 0])}')

# %%
# Value_counts te muestra cuantos valores hay de cada valor único en la columna selecionada

print(equipos_df_clean[['Tipo_Equipo', 'Fabricante', 'Modelo']].value_counts())

# %%
# Frecuencias en equipos

for col in equipos_df_clean.columns:
    print('\n- Frecuencias para "{0}"'.format(col), '\n')
    print(equipos_df_clean[col].value_counts())

# %%
# Correlación

corr_equipos = equipos_df_clean.corr('pearson', numeric_only=True)
corr_equipos

# %%
# Correlación

corr_equipos[(np.abs(corr_equipos) >= 0.7) & (np.abs(corr_equipos) >= 0.7)]

# %%
# Sesgo

skw_equipos = equipos_df_clean.skew(numeric_only=True)
skw_equipos

# %%
# Sesgo

skw_equipos[np.abs(skw_equipos) > 2]

# %%
# Kurtosis

kurt_equipos = equipos_df_clean.kurt(numeric_only=True)
kurt_equipos

# %%
# Kurtosis

kurt_equipos[np.abs(kurt_equipos) > 3]

# %%
# Evaluar la unicidad de posibles claves primarias

# Crear una lista para almacenar los resultados
unike_keys_equipos = []

# Iterar sobre cada columna del DataFrame
for column in equipos_df_clean.columns:
    # Verificar si la columna tiene valores únicos
    if equipos_df_clean[column].is_unique:
        unike_keys_equipos.append(column)

# Mostrar las posibles claves primarias
print("Posibles claves primarias en equipos_df_clean:")
print(unike_keys_equipos)

# %%
equipos_df_clean.describe()

# %%
equipos_df_clean.describe(include='category')

# %% [markdown]
# ### LIMPIEZA Y PERFILADO DE HISTORICOS ORDENES

# %%
# Mostrar las primeras filas de cada dataset
ordenes_df.head()

# %%
# Revisar info del df
ordenes_df.info()

# %%
# Datos vacios
print('Datos vacios en Historicos_Ordenes.csv\n',ordenes_df.isna().sum())

# %%
# Extremos
z_scores = (ordenes_df-ordenes_df.mean(numeric_only=True)) / \
    ordenes_df.std(numeric_only=True)
z_scores_abs = z_scores.apply(np.abs)
print(tabulate(z_scores_abs, headers='keys'))

# %%
umbral = 3

out_mask = ~z_scores[z_scores_abs > umbral].isna()
print('\nOutliers per column:\n')
print(out_mask.sum())

# %%
# Busco filas que tengan el mismo ID_Equipo + Duracion_Horas que la fila con Costo_Mantenimiento vacío y además muestro al final su sumatorio

# Filtrar filas con Costo_Mantenimiento vacío
filas_vacias = ordenes_df[ordenes_df['Costo_Mantenimiento'].isna()]

# Inicializar contador de coincidencias
nan_coincidencias = 0

# Buscar coincidencias en todo el DataFrame
for index, row in filas_vacias.iterrows():
    coincidencias1 = ordenes_df[(ordenes_df['ID_Equipo'] == row['ID_Equipo']) & 
                               (ordenes_df['Duracion_Horas'] == row['Duracion_Horas'])]
    if len(coincidencias1) > 1:
        print(tabulate(coincidencias1, headers='keys'))
        nan_coincidencias += 1

# Mostrar el sumatorio de las filas NaN que han coincidido
print(f'\nSumatorio de filas NaN que han coincidido: {nan_coincidencias}')

# %%
# Primero sustituyo el valor de Costo_Mantenimiento vacío por el valor de la línea que coincida con ID_Equipo + Duración_Horas + Ubicación

ordenes_df_clean = pd.DataFrame(ordenes_df)

# Filtrar filas con Costo_Mantenimiento vacío
filas_vacias = ordenes_df_clean[ordenes_df_clean['Costo_Mantenimiento'].isna()]

# Buscar coincidencias en todo el DataFrame y reemplazar valores vacíos
for index, row in filas_vacias.iterrows():
    coincidencias2 = ordenes_df_clean[(ordenes_df_clean['ID_Equipo'] == row['ID_Equipo']) & 
                                (ordenes_df_clean['Duracion_Horas'] == row['Duracion_Horas']) &
                                (ordenes_df_clean['Ubicacion'] == row['Ubicacion']) & 
                                (ordenes_df_clean['Costo_Mantenimiento'].notna())]
    if not coincidencias2.empty:
        # Reemplazar el valor vacío en ordenes_df_clean con el valor de la primera coincidencia encontrada
        ordenes_df_clean.at[index, 'Costo_Mantenimiento'] = coincidencias2.iloc[0]['Costo_Mantenimiento']


# %%
# Segundo sustituyo el valor de Costo_Mantenimiento vacío por el valor de la línea que coincida ID_Equipo + Duración_Horas

# Filtrar filas con Costo_Mantenimiento vacío
filas_vacias = ordenes_df_clean[ordenes_df_clean['Costo_Mantenimiento'].isna()]

# Buscar coincidencias en todo el DataFrame y reemplazar valores vacíos
for index, row in filas_vacias.iterrows():
    coincidencias3 = ordenes_df_clean[(ordenes_df_clean['ID_Equipo'] == row['ID_Equipo']) & 
                               (ordenes_df_clean['Duracion_Horas'] == row['Duracion_Horas']) &
                               (ordenes_df_clean['Costo_Mantenimiento'].notna())]
    if not coincidencias3.empty:
        # Reemplazar el valor vacío con el valor de la primera coincidencia encontrada
        ordenes_df_clean.at[index, 'Costo_Mantenimiento'] = coincidencias3.iloc[0]['Costo_Mantenimiento']



# %%
# Vuelvo a revisar los datos vacíos donde ya he sustituidos los anteriores reduciendo a 33 los valores vacíos
print('Datos vacios en Historicos_Ordenes.csv\n',ordenes_df_clean.isna().sum())

# %%
# Sustituyo valores vacíos por su media

# Calcular la media de la columna Costo_Mantenimiento
mean_costo_mantenimiento = ordenes_df_clean['Costo_Mantenimiento'].mean()

# Sustituir los valores vacíos por la media
ordenes_df_clean['Costo_Mantenimiento'].fillna(mean_costo_mantenimiento, inplace=True)

# Vuelvo a revisar los datos vacíos donde ya he sustituidos los anteriores reduciendo en 17 los valores vacíos
print('Datos vacios en Historicos_Ordenes.csv\n',ordenes_df_clean.isna().sum())

# %%
# Vuelvo a revisar info del df
ordenes_df_clean.info()

# %%
# Mostrar todas las filas duplicadas
print('\nSumatorio duplicados en ordenes', ordenes_df_clean.duplicated().sum())
ordenes_df_clean[ordenes_df_clean.duplicated(keep=False)]

# %%
# Elimino filas duplicadas una vez las he analizado

ordenes_df_cleaned = ordenes_df_clean.drop_duplicates(keep='first')
print('\nSumatorio duplicados en ordenes_df_clean:', ordenes_df_cleaned.duplicated().sum())

# %%
# Vuelvo a revisar info del df
ordenes_df_cleaned.info()

# %%
# Cardinalidad

# Calcular el sumatorio de valores para cada columna
sum_values_ordenes = ordenes_df_cleaned.count()

# Calcular el sumatorio de valores únicos para cada columna
unique_sum_ordenes = ordenes_df_cleaned.nunique()

# Crear un nuevo DataFrame para mostrar ambos sumatorios
result_df_ordenes = pd.DataFrame({
    'Sumatorio de valores': sum_values_ordenes,
    'Sumatorio de valores únicos': unique_sum_ordenes
})

print(result_df_ordenes)

# %%
# Columnas object a 'category'

ordenes_df_cleaned[['Tipo_Mantenimiento', 'Ubicacion']] = ordenes_df_cleaned[[
    'Tipo_Mantenimiento', 'Ubicacion']].astype('category')

# Convertir fechas a a dateTime
ordenes_df_cleaned["Fecha"] = pd.to_datetime(ordenes_df_cleaned["Fecha"], errors='coerce')

ordenes_df_cleaned.info()

# %%
ordenes_df_cleaned.describe()

# %%
ordenes_df_cleaned.describe(include='category')

# %%
# Value_counts te muestra cuantos valores hay de cada valor único en la columna selecionada

print(ordenes_df_cleaned[['Tipo_Mantenimiento', 'Ubicacion']].value_counts())

# %%
# Frecuencias en ordenes

for col in ordenes_df_cleaned.columns:
    print('\n- Frecuencias para "{0}"'.format(col), '\n')
    print(ordenes_df_cleaned[col].value_counts())

# %%
# Correlación

corr_ordenes = ordenes_df_cleaned.corr('pearson', numeric_only=True)
corr_ordenes

# %%
# Correlación

corr_ordenes[(np.abs(corr_ordenes) >= 0.7) & (np.abs(corr_ordenes) >= 0.7)]

# %%
# Sesgo

skw_ordenes = ordenes_df_cleaned.skew(numeric_only=True)
skw_ordenes

# %%
# Sesgo

skw_ordenes[np.abs(skw_ordenes) > 2]

# %%
# Kurtosis

kurt_ordenes = ordenes_df_cleaned.kurt(numeric_only=True)
kurt_ordenes

# %%
# Kurtosis

kurt_ordenes[np.abs(kurt_ordenes) > 3]

# %%
# Evaluar la unicidad de posibles claves primarias

# Crear una lista para almacenar los resultados
unike_keys_ordenes = []

# Iterar sobre cada columna del DataFrame
for column in ordenes_df_cleaned.columns:
    # Verificar si la columna tiene valores únicos
    if ordenes_df_cleaned[column].is_unique:
        unike_keys_ordenes.append(column)

# Mostrar las posibles claves primarias
print("Posibles claves primarias en ordenes_df_cleaned:")
print(unike_keys_ordenes)

# %%
# Cálculo la frecuencia de mto por cada equipo según sea correctivo o preventivo

# Calcular la frecuencia de mantenimiento por equipo y tipo de mantenimiento
frecuencia_mantenimiento = ordenes_df_cleaned.groupby(['ID_Equipo', 'Tipo_Mantenimiento']).size().unstack(fill_value=0).reset_index()

# Renombrar las columnas para mayor claridad
frecuencia_mantenimiento.columns = ['ID_Equipo', 'Frecuencia_Correctivo', 'Frecuencia_Preventivo']

# Mostrar la tabla con cada ID_Equipo y su frecuencia de mantenimiento por tipo
frecuencia_mantenimiento

# %% [markdown]
# ### LIMPIEZA Y PERFILADO DE REGISTROS CONDICIONES

# %%
# Mostrar las primeras filas de cada dataset
condiciones_df.head()

# %%
# Revisar info del df
condiciones_df.info()

# %%
# Datos vacios
print('Datos vacios en Caracteristicas_Equipos.csv\n',condiciones_df.isna().sum())

# %%
# Relleno los valores vacíos de Horas_Operativas con la media de su columna una vez analizados sus valores

condiciones_df_fill = pd.DataFrame(condiciones_df)

# Calcular la media solo de las columna Horas_Operativas
mean_values_horas = condiciones_df['Horas_Operativas'].mean()

# Rellenar los valores vacíos con la media de su columna
condiciones_df_fill['Horas_Operativas'].fillna(mean_values_horas, inplace=True)

# Verificar que no queden valores vacíos
print('Datos vacios en condiciones_df_fill\n', condiciones_df_fill.isna().sum())


# %%
# Extremos
z_scores = (condiciones_df_fill - condiciones_df_fill.mean(numeric_only=True)) / \
    condiciones_df_fill.std(numeric_only=True)
z_scores_abs = z_scores.apply(np.abs)
print(tabulate(z_scores_abs, headers='keys'))

# %%
umbral = 3

out_mask = ~z_scores[z_scores_abs > umbral].isna()
print('\nOutliers per column:\n')
print(out_mask.sum())

# %%
# Filtrar las filas con outliers
outliers = condiciones_df_fill[(z_scores_abs > umbral).any(axis=1)]

# Mostrar las filas con outliers
print(outliers)

# %%
# Remplazo los outliers con la media de su columna después de analizar sus características

condiciones_df_clean = pd.DataFrame(condiciones_df_fill)

# Calcular la media de la columna Temperatura_C
mean_temperatura = condiciones_df_clean['Temperatura_C'].mean()

# Reemplazar los outliers por la media
condiciones_df_clean.loc[(z_scores_abs['Temperatura_C'] > umbral), 'Temperatura_C'] = mean_temperatura

# Verificar que los outliers han sido reemplazados
print(condiciones_df_clean[(z_scores_abs['Temperatura_C'] > umbral)])

# %%
# Mostrar todas las filas duplicadas
print('\nSumatorio duplicados en Registros_Condiciones', condiciones_df_clean.duplicated().sum())
condiciones_df_clean[condiciones_df_clean.duplicated(keep=False)]

# %%
condiciones_df_clean.info()

# %%
# Cardinalidad

# Calcular el sumatorio de valores para cada columna
sum_value_condiciones = condiciones_df_clean.count()

# Calcular el sumatorio de valores únicos para cada columna
unique_sum_condiciones = condiciones_df_clean.nunique()

# Crear un nuevo DataFrame para mostrar ambos sumatorios
result_df_condicones = pd.DataFrame({
    'Sumatorio de valores': sum_value_condiciones,
    'Sumatorio de valores únicos': unique_sum_condiciones
})

print(result_df_condicones)

# %%
# Convertir fechas a a dateTime

condiciones_df_clean["Fecha"] = pd.to_datetime(condiciones_df_clean["Fecha"], errors='coerce')

condiciones_df_clean.info()

# %%
condiciones_df_clean.describe()

# %%
# Frecuencias en condiciones

for col in condiciones_df_clean.columns:
    print('\n- Frecuencias para "{0}"'.format(col), '\n')
    print(condiciones_df_clean[col].value_counts())

# %%
# Correlación

corr_condiciones = condiciones_df_clean.corr('pearson', numeric_only=True)
corr_condiciones

# %%
# Correlación

corr_condiciones[(np.abs(corr_condiciones) >= 0.7) & (np.abs(corr_condiciones) >= 0.7)]

# %%
# Sesgo

skw_condiciones = condiciones_df_clean.skew(numeric_only=True)
skw_condiciones

# %%
# Sesgo

skw_condiciones[np.abs(skw_condiciones) > 2]

# %%
# Kurtosis

kurt_condiones = condiciones_df_clean.kurt(numeric_only=True)
kurt_condiones

# %%
# Kurtosis

kurt_condiones[np.abs(kurt_condiones) > 3]

# %%
# Evaluar la unicidad de posibles claves primarias

# Crear una lista para almacenar los resultados
unike_keys_condiciones = []

# Iterar sobre cada columna del DataFrame
for column in condiciones_df_clean.columns:
    # Verificar si la columna tiene valores únicos
    if condiciones_df_clean[column].is_unique:
        unike_keys_condiciones.append(column)

# Mostrar las posibles claves primarias
print("Posibles claves primarias en condiciones_df_clean:")
print(unike_keys_condiciones)

# %%
condiciones_df_clean.describe()

# %%
# Estimación de la vida útil de los equipos basada en el máximo de las horas operativas

# Agrupar por ID_Equipo y calcular la media de Horas_Operativas
vida_util_estimacion = condiciones_df_clean.groupby('ID_Equipo')['Horas_Operativas'].max().reset_index()

# Renombrar la columna para mayor claridad
vida_util_estimacion.columns = ['ID_Equipo', 'Vida_Util_Estimada']

# Mostrar la estimación de la vida útil
print(vida_util_estimacion)

# %% [markdown]
# ### EXPORTAR ARCHIVOS LIMPIOS

# %%
# Obtener la ruta del directorio actual
current_directory = os.getcwd()

print("La ruta del directorio actual es:", current_directory)

# %%
# Extraer los DataFrame limpios a un archivo CSV

equipos_df_clean.to_csv(ruta + 'etapa2/output/Caracteristicas_Equipos_limpio.csv', index=False)
ordenes_df_cleaned.to_csv(ruta + 'etapa2/output/Historicos_Ordenes_limpio.csv', index=False)
condiciones_df_clean.to_csv(ruta + 'etapa2/output/Registros_Condiciones_limpio.csv', index=False)

print("El DataFrame equipos_df_clean, ordenes_df_cleaned y condiciones_df_clean se ha extraído a *.csv")

# %% [markdown]
# # Mezcla y Combinación de Datos

# %%
# Renombrar las columnas Fecha antes de la combinación
ordenes_df_cleaned = ordenes_df_cleaned.rename(columns={'Fecha': 'Fecha_Ordenes'})
condiciones_df_clean = condiciones_df_clean.rename(columns={'Fecha': 'Fecha_Registros'})

# Convertir las columnas Fecha a datetime
ordenes_df_cleaned['Fecha_Ordenes'] = pd.to_datetime(ordenes_df_cleaned['Fecha_Ordenes'])
condiciones_df_clean['Fecha_Registros'] = pd.to_datetime(condiciones_df_clean['Fecha_Registros'])

# Combinar equipos_df_clean y ordenes_df_cleaned
mantenimiento_df = pd.merge(equipos_df_clean, ordenes_df_cleaned, on='ID_Equipo', how='left')

# Combinar el resultado anterior con condiciones_df_clean
mantenimiento_df = pd.merge(mantenimiento_df, condiciones_df_clean, on='ID_Equipo', how='left')

# Mostrar el DataFrame combinado
mantenimiento_df.head()

# %%
# Revisar info del df
mantenimiento_df.info()

# %%
# Datos vacios
print('Datos vacios en mantenimiento\n',mantenimiento_df.isna().sum())

# %%
# Mostrar las filas que tienen valores vacíos
filas_vacias = mantenimiento_df[mantenimiento_df.isna().any(axis=1)]
filas_vacias

# %%
# Al no disponer de Historicos_Ordenes y Registros_Condiciones del ID_Equipo 500 se decide eliminar la fila

# Eliminar la fila con valores vacíos
mantenimiento_df = mantenimiento_df.dropna()

# Datos vacios
print('Datos vacios en mantenimiento\n',mantenimiento_df.isna().sum())

# %%
# Mostrar todas las filas duplicadas
print('\nSumatorio duplicados en mantenimiento', mantenimiento_df.duplicated().sum())
mantenimiento_df[mantenimiento_df.duplicated(keep=False)]

# %%
# Filtrar solo las columnas numéricas
numeric_cols = mantenimiento_df.select_dtypes(include=[np.number])

# Calcular los puntajes Z solo para las columnas numéricas
z_scores = (numeric_cols - numeric_cols.mean()) / numeric_cols.std()
z_scores_abs = z_scores.apply(np.abs)

# Imprimir los resultados
print(tabulate(z_scores_abs, headers='keys'))

# %%
# Contar cuántos valores superan el umbral de 3 en cada columna
umbral = 3
superan_umbral = (z_scores_abs > umbral).sum()

print("Valores que superan el umbral de 3 en cada columna:")
print(superan_umbral)

# %%
# Cardinalidad

# Calcular el sumatorio de valores para cada columna
sum_value_mantenimiento = mantenimiento_df.count()

# Calcular el sumatorio de valores únicos para cada columna
unique_sum_mantenimiento = mantenimiento_df.nunique()

# Crear un nuevo DataFrame para mostrar ambos sumatorios
result_df_mantenimiento = pd.DataFrame({
    'Sumatorio de valores': sum_value_mantenimiento,
    'Sumatorio de valores únicos': unique_sum_mantenimiento
})

print(result_df_mantenimiento)

# %%
mantenimiento_df.describe()

# %%
# Frecuencias en condiciones

for col in mantenimiento_df.columns:
    print('\n- Frecuencias para "{0}"'.format(col), '\n')
    print(mantenimiento_df[col].value_counts())

# %%
# Correlación

corr_mantenimiento = mantenimiento_df.corr('pearson', numeric_only=True)
corr_mantenimiento

# %%
# Correlación

corr_mantenimiento[(np.abs(corr_mantenimiento) >= 0.7) & (np.abs(corr_mantenimiento) >= 0.7)]

# %%
# Sesgo

skw_mantenimiento = mantenimiento_df.skew(numeric_only=True)
skw_mantenimiento

# %%
# Sesgo

skw_mantenimiento[np.abs(skw_mantenimiento) > 2]

# %%
# Kurtosis

kurt_mantenimeinto = mantenimiento_df.kurt(numeric_only=True)
kurt_mantenimeinto

# %%
# Kurtosis

kurt_mantenimeinto[np.abs(kurt_mantenimeinto) > 3]

# %%
# Evaluar la unicidad de posibles claves primarias

# Crear una lista para almacenar los resultados
unike_keys_mantenimiento = []

# Iterar sobre cada columna del DataFrame
for column in mantenimiento_df.columns:
    # Verificar si la columna tiene valores únicos
    if mantenimiento_df[column].is_unique:
        unike_keys_mantenimiento.append(column)

# Mostrar las posibles claves primarias
print("Posibles claves primarias en mantenimiento:")
print(unike_keys_mantenimiento)

# %%
mantenimiento_df.describe()

# %%
# Extraer el DataFrame limpio a un archivo CSV

mantenimiento_df.to_csv(ruta + 'etapa2/output/Mantenimiento.csv', index=False)


print("El DataFrame mantenimiento_df se ha extraído a *.csv")

# %% [markdown]
# # Resumen limpieza y archivo conjunto

# %% [markdown]
# CARACTERISTICAS _EQUIPOS .CSV  
# 
# 1. Valores Nulos  
# • Caracteristicas_Equipos.csv : 
# Contiene valores nulos en varias columnas, por ejemplo: Potencia_kW  en la fila 33.  
# • Caracteristicas_Equipos_limpio.csv : 
# Los valores nulos han sido eliminados o reemplazados. No se observan valores nulos en 
# las columnas.  
# 
# 2. Valores Atípicos  
# • Caracteristicas_Equipos.csv : Contiene valores atípicos, por ejemplo: Potencia_kW  con un valor de  -100  en varias filas (por ejemplo, filas 33, 59, 71, 82, 
# 157, 214, 264, 303, 321, 490).  
# • Caracteristicas_Equipos_limpio.csv : Los valores atípicos han sido corregidos. No se observan valores fuera de rango en las columnas.  
# 
# 3. Duplicados  
# • Caracteristicas_Equipos.csv : Puede contener registros duplicados.  
# • Caracteristicas_Equipos_limpio.csv : Los registros duplicados han sido eliminados.  
# 
# 4. Inconsistencias de Formato  
# • Caracteristicas_Equipos.csv : Puede contener inconsistencias en el formato de los datos.  
# • Caracteristicas_Equipos_limpio.csv : Las inconsistencias de formato han sido corregidas. Los datos están uniformemente formateados.  
# 
# 5. Relleno de Valores Faltantes  
# • Caracteristicas_Equipos.csv : Contiene valores faltantes en varias columnas.  
# • Caracteristicas_Equipos_limpio.csv : Los valores faltantes han sido rellenados con la media o mediana de la columna 
# correspondiente.  
# 
# Resumen  
# 
# El archivo  Caracteristicas_Equipos_limpio.csv  ha sido procesado para eliminar valores nulos, corregir 
# valores atípicos, eliminar duplicados y corregir inconsistencias de formato. Los valores faltantes han sido 
# rellenados adecuadamente, lo que hace que los datos estén listos para un análisis más profu ndo y confiable.
# 
# HISTORICOS ORDENES .CSV  
#  
# 1. Valores Nulos  
# • Historicos_Ordenes.csv : Contiene valores nulos en varias columnas, por ejemplo: Costo_Mantenimiento  en la fila 913.  
# • Historicos_Ordenes_limpio.csv : Los valores nulos han sido eliminados o reemplazados. No se observan valores nulos en 
# las columnas.  
# 
# 2. Valores Atípicos  
# • Historicos_Ordenes.csv : Contiene valores atípicos, por ejemplo: Costo_Mantenimiento  con un valor de  -100  en la fila 914.  
# • Historicos_Ordenes_limpio.csv : Los valores atípicos han sido corregidos. No se observan valores fuera de rango en las columnas.  
# 
# 3. Duplicados  
# • Historicos_Ordenes.csv : Puede contener registros duplicados.  
# • Historicos_Ordenes_limpio.csv : Los registros duplicados han sido eliminados.  
# 
# 4. Inconsistencias de Formato  
# • Historicos_Ordenes.csv : Puede contener inconsistencias en el formato de los datos.  
# • Historicos_Ordenes_limpio.csv : Las inconsistencias de formato han sido corregidas. Los datos están uniformemente formateados.  
# 
# 5. Relleno de Valores Faltantes  
# • Historicos_Ordenes.csv : Contiene valores faltantes en varias columnas.  
# • Historicos_Ordenes_limpio.csv : Los valores faltantes han sido rellenados con la media o mediana de la columna correspondiente.  
# 
# Resumen 
# 
# El archivo  Historicos_Ordenes_limpio.csv  ha sido procesado para eliminar valores nulos, corregir valores 
# atípicos, eliminar duplicados y corregir inconsistencias de formato. Los valores faltantes han sido 
# rellenados adecuadamente, lo que hace que los datos es tén listos para un análisis más profundo y 
# confiable.
# 
# REGISTROS CONDICIONES .CSV  
# 
# 1. Valores Nulos  
# • Registros_Condiciones.csv : Contiene valores nulos en varias columnas, por ejemplo:  
# ▪ Horas_Operativas  en la fila 90.  
# ▪ Temperatura_C  en la fila 256.  
# • Registros_Condiciones_limpio.csv : Los valores nulos han sido eliminados o reemplazados. No se observan valores nulos en las columnas.  
# 
# 2. Valores Atípicos  
# • Registros_Condiciones.csv : Contiene valores atípicos, por ejemplo: Temperatura_C  con un valor de 999.0 en la fila 256.  
# • Registros_Condiciones_limpio.csv : Los valores atípicos han sido corregidos. No se observan valores fuera de rango en las columnas.  
# 
# 3. Duplicados  
# • Registros_Condiciones.csv : Puede contener registros duplicados.  
# • Registros_Condiciones_limpio.csv : Los registros duplicados han sido eliminados.  
# 
# 4. Inconsistencias de Formato  
# • Registros_Condiciones.csv : Puede contener inconsistencias en el formato de los datos.  
# • Registros_Condiciones_limpio.csv : Las inconsistencias de formato han sido corregidas. Los datos están uniformemente formateados.  
# 
# 5. Relleno de Valores Faltantes  
# • Registros_Condiciones.csv : Contiene valores faltantes en varias columnas.  
# • Registros_Condiciones_limpio.csv : Los valores faltantes han sido rellenados con la media o mediana de la columna correspondiente.  
# 
# Resumen  
# 
# El archivo  Registros_Condiciones_limpio.csv  ha sido procesado para eliminar valores nulos, corregir 
# valores atípicos, eliminar duplicados y corregir inconsistencias de formato. Los valores faltantes han sido 
# rellenados adecuadamente, lo que hace que los da tos estén listos para un análisis más profundo y 
# confiable.
# 
# ARCHIVO CONJUNTO MANTENIMIENTO.CSV  
# 
# 1. Valores Nulos : No se detectaron valores nulos en las columnas del archivo. Todas las columnas 
# contienen datos completos.  
# 
# 2. Duplicados : Se identificaron múltiples registros para el mismo equipo ( ID_Equipo  = 1) con la misma 
# información en varias columnas. Esto sugiere la presencia de duplicados.
# Se eliminaron las filas duplicadas para asegurar la integridad de los datos.  
# 
# 3. Consistencia de Datos : Las columnas  ID_Equipo , Tipo_Equipo , Fabricante , Modelo, Potencia_kW , 
# y Horas_Recomendadas_Revision  tienen valores consistentes a lo largo del archivo.  
# Las columnas  Costo_Mantenimiento , Duracion_Horas , Temperatura_C , Vibracion_mm_s , 
# y Horas_Operativas  muestran variaciones esperadas debido a las diferentes condiciones 
# de mantenimiento y operación.  
# 
# 4. Formato de Fechas : 
# Las columnas  Fecha_Ordenes  y Fecha_Registros  están en el formato correcto ( YYYY-MM-DD HH:MM:SS ). 
# Se convirtieron a tipo  datetime  para facilitar el análisis temporal.  
# 
# Conclusiones
# 
# • Integridad de Datos : Los datos son completos y no presentan valores nulos.  
# • Eliminación de Duplicados : Se eliminaron duplicados para mantener la integridad de los datos.  
# • Consistencia : Los datos son consistentes en las columnas clave.  
# • Formato de Fechas : Las fechas están en el formato correcto y se convirtieron a tipo  datetime . 
# • Rangos de Datos : Los rangos de datos en las columnas numéricas son razonables y no presentan valores atípicos significativos.  
# 

# %% [markdown]
# # Análisis Exploratorio de Datos y Visualización

# %% [markdown]
# #### Frecuencia de mantenimiento: Número de mantenimientos realizados por equipo.
# 
# - Se genera una tabla nueva con información relevante de los equipos. Se hace fuera de la tabla conjunta ya que es más entendible.
# - Se extrae también a un csv

# %%
# Cálculo la frecuencia de mto por cada equipo según sea correctivo o preventivo

# Calcular la frecuencia de mantenimiento por equipo y tipo de mantenimiento
frecuencia_mantenimiento = mantenimiento_df.groupby(['ID_Equipo', 'Tipo_Mantenimiento']).size().unstack(fill_value=0).reset_index()

# Renombrar las columnas para mayor claridad
frecuencia_mantenimiento.columns = ['ID_Equipo', 'Frecuencia_Correctivo', 'Frecuencia_Preventivo']

# Estimación de la vida útil de los equipos basada en el máximo de las horas operativas

# Agrupar por ID_Equipo y calcular la media de Horas_Operativas
vida_util_estimacion = mantenimiento_df.groupby('ID_Equipo')['Horas_Operativas'].max().reset_index()

# Renombrar la columna para mayor claridad
vida_util_estimacion.columns = ['ID_Equipo', 'Vida_Util_Estimada']

# Unir las dos tablas basadas en la columna ID_Equipo
tabla_unida = pd.merge(frecuencia_mantenimiento, vida_util_estimacion, on='ID_Equipo')


# Añadir Tipo_Equipo, Fabricante y Modelo
tabla_unida = pd.merge(tabla_unida, equipos_df_clean[['ID_Equipo', 'Tipo_Equipo', 'Fabricante', 'Modelo']], on='ID_Equipo')

# Reordenar las columnas del DataFrame tabla_unida
tabla_unida = tabla_unida[['ID_Equipo', 'Tipo_Equipo', 'Fabricante', 'Modelo', 'Frecuencia_Correctivo', 'Frecuencia_Preventivo', 'Vida_Util_Estimada']]

# Mostrar la tabla unida con el nuevo orden de columnas
tabla_unida


# %%
import matplotlib.pyplot as plt

# Calcular la frecuencia de mantenimiento total (correctivo + preventivo) por equipo
tabla_unida['Frecuencia_Total'] = tabla_unida  ['Frecuencia_Correctivo'] + tabla_unida   ['Frecuencia_Preventivo']

# Crear una gráfica de barras para mostrar la frecuencia de mantenimiento por equipo
plt.figure(figsize=(14, 7))
plt.bar(tabla_unida['ID_Equipo'], tabla_unida  ['Frecuencia_Total'], color='skyblue')
plt.xlabel('ID del Equipo')
plt.ylabel('Frecuencia de Mantenimiento')
plt.title('Frecuencia de Mantenimiento por Equipo')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# %%
# Extraer el DataFrame limpio a un archivo CSV

tabla_unida_df = pd.DataFrame(tabla_unida)

tabla_unida_df.to_csv(ruta + 'etapa2/output/Informacion_relevante_equipos.csv', index=False)


print("El DataFrame tabla_unida_df se ha extraído a *.csv")

# %% [markdown]
# #### Vida útil estimada: Estimación de la vida útil de los equipos basada en las horas operativas.
# 
# - Utilizando la columna Vida_Util_Estimada generada anteriormente, se calcula la vida util restante.

# %%
# Calcular la vida útil restante basada en las horas operativas actuales y en la vida util estimada
tabla_unida['Vida_Util_Restante'] = tabla_unida['Vida_Util_Estimada'] - condiciones_df_clean['Horas_Operativas']

# Mostrar el DataFrame con la vida útil restante calculada
tabla_unida

# %%

# Crear un gráfico de dispersión para mostrar la vida útil restante por equipo
plt.figure(figsize=(14, 7))
plt.scatter(tabla_unida['ID_Equipo'], tabla_unida['Vida_Util_Restante'], color='skyblue')
plt.xlabel('ID del Equipo')
plt.ylabel('Vida Útil Restante (horas)')
plt.title('Vida Útil Restante por Equipo')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# %% [markdown]
# #### Tiempo hasta fallo: ¿Cuánto tiempo de operación transcurre hasta que un equipo requiere mantenimiento?
# 
# - Se genera una tabla nueva donde se predice Tiempo hasta fallo: ¿Cuánto tiempo de operación transcurre hasta que un equipo requiere mantenimiento
# - Se extrae también a un csv

# %%
# Crear una copia del DataFrame original para no modificarlo
ordenes_df_temp = ordenes_df_cleaned.copy()

# Convertir la columna Fecha a datetime
ordenes_df_temp['Fecha_Ordenes'] = pd.to_datetime(ordenes_df_temp['Fecha_Ordenes'], errors='coerce')

# Ordenar el DataFrame por ID_Equipo y Fecha_Ordenes
ordenes_df_temp = ordenes_df_temp.sort_values(by=['ID_Equipo', 'Fecha_Ordenes'])

# Calcular la diferencia de tiempo entre cada ID_Orden para cada ID_Equipo
ordenes_df_temp['Tiempo_Entre_Ordenes'] = ordenes_df_temp.groupby('ID_Equipo')['Fecha_Ordenes'].diff()

# Calcular la media de Tiempo_Entre_Ordenes para cada ID_Equipo
media_tiempo_entre_ordenes = ordenes_df_temp.groupby('ID_Equipo')['Tiempo_Entre_Ordenes'].mean().reset_index()

# Renombrar la columna para mayor claridad
media_tiempo_entre_ordenes.columns = ['ID_Equipo', 'Tiempo_hasta_fallo']

# Mostrar el resultado
media_tiempo_entre_ordenes


# %%
import matplotlib.pyplot as plt

# Crear una gráfica de barras para mostrar el Tiempo_hasta_fallo por equipo
plt.figure(figsize=(14, 7))
plt.bar(media_tiempo_entre_ordenes['ID_Equipo'], media_tiempo_entre_ordenes['Tiempo_hasta_fallo'], color='skyblue')
plt.xlabel('ID del Equipo')
plt.ylabel('Tiempo_hasta_fallo')
plt.title('Tiempo_hasta_fallo por Equipo')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# %%
# Extraer el DataFrame Tiempo_hasta_fallo a un archivo CSV

media_tiempo_entre_ordenes.to_csv(ruta + 'etapa2/output/Tiempo_hasta_fallo.csv', index=False)


print("El DataFrame se ha extraído a *.csv")

# %% [markdown]
# #### Relación entre condiciones operativas y fallos: ¿Cómo impactan la vibración o temperatura en la probabilidad de un fallo?
# 
# - Se llega a la conclusión de que la temperatura y sobre todo las vibraciones tienen tendencia a aumentar hasta que ocurre el fallo y hay una acuación a través de una orden. Después sus valores descienden y y vuelven a valores normales.

# %%
# Generar una nueva tabla con las columnas ID_Equipo, Fecha_Ordenes, Tipo_Mantenimiento, Fecha_Registros,Temperatura_C y Vibracion_mm_s
condiciones_operativas = mantenimiento_df[['ID_Equipo', 'Fecha_Ordenes', 'Tipo_Mantenimiento', 'Fecha_Registros', 'Temperatura_C', 'Vibracion_mm_s']]

# Mostrar la nueva tabla
condiciones_operativas

# %%
# Filtrar las filas donde Tipo_Mantenimiento es "Correctivo"
condiciones_operativas_correctivo = condiciones_operativas[condiciones_operativas['Tipo_Mantenimiento'] == 'Correctivo']

# Mostrar la tabla filtrada
condiciones_operativas_correctivo

# %%
# Se genera una gráfica para varios equipos a la vez

import matplotlib.pyplot as plt
import random

# Generar 4 ID_Equipo aleatorios únicos entre 1 y 499
random_ids = random.sample(range(1, 500), 4)

# Crear una figura con 4 subgráficas
fig, axs = plt.subplots(2, 2, figsize=(20, 14))

# Iterar sobre los ID_Equipo aleatorios y los ejes de las subgráficas
for i, (ax1, equipo_id) in enumerate(zip(axs.flatten(), random_ids)):
    # Filtrar los datos para el ID_Equipo actual
    condiciones_operativas_correctivo_id = condiciones_operativas_correctivo[condiciones_operativas_correctivo['ID_Equipo'] == equipo_id]
    
    # Convertir la columna Fecha_Ordenes a datetime si no lo está
    condiciones_operativas_correctivo_id['Fecha_Ordenes'] = pd.to_datetime(condiciones_operativas_correctivo_id['Fecha_Ordenes'])
    
    # Graficar la temperatura en función de las fechas de órdenes
    ax1.plot(condiciones_operativas_correctivo_id['Fecha_Ordenes'], condiciones_operativas_correctivo_id['Temperatura_C'], 'b-', label='Temperatura (°C)')
    ax1.set_xlabel('Fecha de Órdenes')
    ax1.set_ylabel('Temperatura (°C)', color='b')
    ax1.tick_params('y', colors='b')
    ax1.legend(loc='upper left')
    
    # Crear un segundo eje para la vibración
    ax2 = ax1.twinx()
    ax2.plot(condiciones_operativas_correctivo_id['Fecha_Ordenes'], condiciones_operativas_correctivo_id['Vibracion_mm_s'], 'r-', label='Vibración (mm/s)')
    ax2.set_ylabel('Vibración (mm/s)', color='r')
    ax2.tick_params('y', colors='r')
    ax2.legend(loc='upper right')
    
    # Establecer el título para cada subgráfica
    ax1.set_title(f'Impacto de la Temperatura y Vibración en Función de las Fechas de Órdenes para el ID_Equipo {equipo_id}')

# Ajustar el diseño
fig.tight_layout()

# Mostrar la gráfica
plt.show()

# %% [markdown]
# #### Relación entre las condiciones operativas (temperatura, vibración) y las horas de operación.
# 
# - Se observa una ligera tendencia en aumento en función de las horas de operación
# - A lo largo del tiempo, se observan distintas bajadas de valores debidos a intervenciones de mantenimiento.

# %%
# Generaro una nueva tabla con las columnas ID_Equipo, Temperatura_C, Vibracion_mm_s y Horas_Operativas
tabla_condiciones = mantenimiento_df[['ID_Equipo', 'Temperatura_C', 'Vibracion_mm_s', 'Horas_Operativas']]

# Ordeno la tabla por ID_Equipo y luego por Horas_Operativas
tabla_condiciones_ordenada = tabla_condiciones.sort_values(by=['ID_Equipo', 'Horas_Operativas'])

# Muestro la tabla ordenada
tabla_condiciones_ordenada

# %%
import matplotlib.pyplot as plt

# Genero 4 ID_Equipo aleatorios únicos entre los disponibles en la tabla
random_ids = random.sample(tabla_condiciones_ordenada['ID_Equipo'].unique().tolist(), 4)

# Cro una figura con 4 subgráficas
fig, axs = plt.subplots(2, 2, figsize=(20, 14))

# Itero sobre los ID_Equipo aleatorios y los ejes de las subgráficas
for i, (ax, equipo_id) in enumerate(zip(axs.flatten(), random_ids)):
    # Filtrar los datos para el ID_Equipo actual
    tabla_equipo = tabla_condiciones_ordenada[tabla_condiciones_ordenada['ID_Equipo'] == equipo_id]
    
    # Graficar la temperatura y la vibración en función de las horas operativas
    ax.plot(tabla_equipo['Horas_Operativas'], tabla_equipo['Temperatura_C'], 'b-', label='Temperatura (°C)')
    ax.set_xlabel('Horas Operativas')
    ax.set_ylabel('Temperatura (°C)', color='b')
    ax.tick_params('y', colors='b')
    ax.legend(loc='upper left')
    
    # Crear un segundo eje para la vibración
    ax2 = ax.twinx()
    ax2.plot(tabla_equipo['Horas_Operativas'], tabla_equipo['Vibracion_mm_s'], 'r-', label='Vibración (mm/s)')
    ax2.set_ylabel('Vibración (mm/s)', color='r')
    ax2.tick_params('y', colors='r')
    ax2.legend(loc='upper right')
    
    # Establecer el título para cada subgráfica
    ax.set_title(f'Condiciones Operativas para el ID_Equipo {equipo_id}')

# Ajustar el diseño
fig.tight_layout()

# Mostrar la gráfica
plt.show()

# %% [markdown]
# ### Gráficas de barras para mostrar la distribución del costo de mantenimiento por tipo de equipo

# %%
import matplotlib.pyplot as plt

# Agrupar los datos por Tipo_Equipo y calcular el costo de mantenimiento total para cada tipo
costo_mantenimiento_por_tipo = mantenimiento_df.groupby('Tipo_Equipo')['Costo_Mantenimiento'].sum().reset_index()

# Crear el gráfico de barras
plt.figure(figsize=(10, 6))
plt.bar(costo_mantenimiento_por_tipo['Tipo_Equipo'], costo_mantenimiento_por_tipo['Costo_Mantenimiento'], color='skyblue')
plt.xlabel('Tipo de Equipo')
plt.ylabel('Costo de Mantenimiento Total')
plt.title('Distribución del Costo de Mantenimiento por Tipo de Equipo')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Diagramas de dispersión para visualizar la relación entre las condiciones operativas (por ejemplo, temperatura y vibración) y las horas de operación o fallos

# %%
# Se genera una gráfica para varios equipos a la vez

import matplotlib.pyplot as plt
import random

# Generar 4 ID_Equipo aleatorios únicos entre 1 y 499
random_ids = random.sample(range(1, 500), 4)

# Crear una figura con 4 subgráficas
fig, axs = plt.subplots(2, 2, figsize=(20, 14))

# Iterar sobre los ID_Equipo aleatorios y los ejes de las subgráficas
for i, (ax1, equipo_id) in enumerate(zip(axs.flatten(), random_ids)):
    # Filtrar los datos para el ID_Equipo actual
    condiciones_operativas_correctivo_id = condiciones_operativas_correctivo[condiciones_operativas_correctivo['ID_Equipo'] == equipo_id]
    
    # Convertir la columna Fecha_Ordenes a datetime si no lo está
    condiciones_operativas_correctivo_id['Fecha_Ordenes'] = pd.to_datetime(condiciones_operativas_correctivo_id['Fecha_Ordenes'])
    
    # Graficar la temperatura en función de las fechas de órdenes
    ax1.plot(condiciones_operativas_correctivo_id['Fecha_Ordenes'], condiciones_operativas_correctivo_id['Temperatura_C'], 'b-', label='Temperatura (°C)')
    ax1.set_xlabel('Fecha de Órdenes')
    ax1.set_ylabel('Temperatura (°C)', color='b')
    ax1.tick_params('y', colors='b')
    ax1.legend(loc='upper left')
    
    # Crear un segundo eje para la vibración
    ax2 = ax1.twinx()
    ax2.plot(condiciones_operativas_correctivo_id['Fecha_Ordenes'], condiciones_operativas_correctivo_id['Vibracion_mm_s'], 'r-', label='Vibración (mm/s)')
    ax2.set_ylabel('Vibración (mm/s)', color='r')
    ax2.tick_params('y', colors='r')
    ax2.legend(loc='upper right')
    
    # Establecer el título para cada subgráfica
    ax1.set_title(f'Impacto de la Temperatura y Vibración en Función de las Fechas de Órdenes para el ID_Equipo {equipo_id}')

# Ajustar el diseño
fig.tight_layout()

# Mostrar la gráfica
plt.show()

# %% [markdown]
# ### Crear un diagrama de caja para identificar los outliers en las horas operativas o costos de mantenimiento

# %%
import matplotlib.pyplot as plt

# Crear un diagrama de caja para las horas operativas
plt.figure(figsize=(8, 4))
plt.boxplot(mantenimiento_df['Horas_Operativas'], vert=True)
plt.xlabel('Horas Operativas')
plt.title('Diagrama de Caja para Horas Operativas')
plt.show()

# Crear un diagrama de caja para los costos de mantenimiento
plt.figure(figsize=(8, 4))
plt.boxplot(mantenimiento_df['Costo_Mantenimiento'], vert=True)
plt.xlabel('Costo de Mantenimiento')
plt.title('Diagrama de Caja para Costo de Mantenimiento')
plt.show()

# %% [markdown]
# ### Generar una visualización de la frecuencia de mantenimiento por tipo de equipo y la relación con el tipo de mantenimiento (correctivo/preventivo)

# %%
import matplotlib.pyplot as plt

# Calcular la frecuencia de mantenimiento por tipo de equipo y tipo de mantenimiento
frecuencia_mantenimiento_tipo = mantenimiento_df.groupby(['Tipo_Equipo', 'Tipo_Mantenimiento']).size().unstack(fill_value=0)

# Crear el gráfico de barras apiladas
frecuencia_mantenimiento_tipo.plot(kind='bar', stacked=True, figsize=(10, 6), color=['skyblue', 'salmon'])
plt.xlabel('Tipo de Equipo')
plt.ylabel('Frecuencia de Mantenimiento')
plt.title('Frecuencia de Mantenimiento por Tipo de Equipo y Tipo de Mantenimiento')
plt.legend(title='Tipo de Mantenimiento')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Analisis y exportación Profiling

# %% [markdown]
# ### ANALISIS PROFILING CARACTERISTICAS_EQUIPOS SIN LIMPIAR

# %%
equipos_df = pd.read_csv(ruta + 'Etapa2/data/Caracteristicas_Equipos.csv')

# %%
from ydata_profiling import ProfileReport
profile = ProfileReport(equipos_df, title="Caracteristicas Equipos Profiling Report")

# %%
profile.to_notebook_iframe()

# %%
profile.to_file(ruta + 'Etapa2/output/Caracteristicas_Equipos.html')

# %% [markdown]
# ### ANALISIS PROFILING HISTORICOS_ORDENES SIN LIMPIAR

# %%
ordenes_df = pd.read_csv(ruta + 'Etapa2/data/Historicos_Ordenes.csv')

# %%
from ydata_profiling import ProfileReport
profile = ProfileReport(ordenes_df, title="Historico Ordenes Profiling Report")

# %%
profile.to_notebook_iframe()

# %%
profile.to_file(ruta + 'Etapa2/output/Historico_ordenes_df.html')

# %% [markdown]
# ### ANALISIS PROFILING REGISTROS CONDICIONES SIN LIMPIAR

# %%
condiciones_df = pd.read_csv(ruta + 'Etapa2/data/Registros_Condiciones.csv')

# %%
from ydata_profiling import ProfileReport
profile = ProfileReport(condiciones_df, title="Registros Condiciones Profiling Report")

# %%
profile.to_notebook_iframe()

# %%
profile.to_file(ruta + 'Etapa2/output/Registros_condiciones_df.html')

# %% [markdown]
# ### ANALISIS PROFILING CARACTERISTICAS_EQUIPOS LIMPIO

# %%
equipos_df_clean = pd.read_csv(ruta + 'Etapa2/output/Caracteristicas_Equipos_limpio.csv')

# %%
from ydata_profiling import ProfileReport
profile = ProfileReport(equipos_df_clean, title="Caracteristicas Equipos limpio Profiling Report")

# %%
profile.to_notebook_iframe()

# %%
profile.to_file(ruta + 'Etapa2/output/Caracteristicas_Equipos_limpio.html')

# %% [markdown]
# ### ANALISIS PROFILING HISTORICOS_ORDENES LIMPIO

# %%
ordenes_df_cleaned = pd.read_csv(ruta + 'Etapa2/output/Historicos_Ordenes_limpio.csv')

# %%
from ydata_profiling import ProfileReport
profile = ProfileReport(ordenes_df_cleaned, title="Historico Ordenes limpio Profiling Report")

# %%
profile.to_notebook_iframe()

# %%
profile.to_file(ruta + 'Etapa2/output/Historico_ordenes_df_limpio.html')

# %% [markdown]
# ### ANALISIS PROFILING REGISTROS CONDICIONES LIMPIO

# %%
condiciones_df_clean = pd.read_csv(ruta + 'Etapa2/output/Registros_Condiciones_limpio.csv')

# %%
from ydata_profiling import ProfileReport
profile = ProfileReport(condiciones_df_clean, title="Registro Condiciones limpio Profiling Report")

# %%
profile.to_notebook_iframe()

# %%
profile.to_file(ruta + 'Etapa2/output/Registros_condiciones_df_limpio.html')

# %% [markdown]
# ### ANALISIS PROFILING ARCHIVO CONJUNTO

# %%
mantenimiento_df = pd.read_csv(ruta + 'Etapa2/output/Mantenimiento.csv')

# %%
from ydata_profiling import ProfileReport
profile = ProfileReport(mantenimiento_df, title="Mantenimiento Profiling Report")

# %%
profile.to_notebook_iframe()

# %%
profile.to_file(ruta + 'Etapa2/output/Mantenimiento.html')

# %% [markdown]
# # Almacenamiento

# %% [markdown]
# #### El CSV limpio se encuentra en la ruta ...../Etapa2/Output/Mantenimiento.csv

# %%
### CSV Limpio

mantenimiento_dff = pd.read_csv(ruta + 'etapa2/output/Mantenimiento.csv')
equipos_df = pd.DataFrame(mantenimiento_dff)
mantenimiento_dff

# %% [markdown]
# #### Eportación PostgreSQL para almacenamiento estructurado

# %%
import pandas as pd
from sqlalchemy import create_engine

# Crear el motor de SQLAlchemy
engine = create_engine('postgresql://cesteves_Netmind_owner:MHcUeXyBw9W1@ep-fragrant-glade-a2ceuu6h-pooler.eu-central-1.aws.neon.tech/cesteves_Netmind?sslmode=require')

# Definir el DataFrame mantenimiento_dff (aquí deberías cargar tus datos reales)
# Por ejemplo, podrías cargar datos desde un archivo CSV
try:
    mantenimiento_dff = pd.read_csv(ruta + 'etapa2/output/Mantenimiento.csv')
    print("DataFrame cargado exitosamente")
except Exception as e:
    print(f"Error al cargar el DataFrame: {e}")
    mantenimiento_dff = None

# Cargar el DataFrame mantenimiento_dff en una nueva tabla llamada 'Mantenimiento' en la base de datos PostgreSQL
if mantenimiento_dff is not None:
    try:
        mantenimiento_dff.to_sql('Mantenimiento', con=engine, if_exists='replace', index=False)
        print("El DataFrame mantenimiento_dff se ha cargado en la nueva tabla 'Mantenimiento' en la base de datos PostgreSQL")
    except Exception as e:
        print(f"Error al cargar el DataFrame en la base de datos: {e}")

# %% [markdown]
# #### Exportación en formato Parquet

# %%
mantenimiento_dff.to_parquet(
    ruta + 'etapa2/output/Mantenimiento.parquet')


