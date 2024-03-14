import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


def verificar_tipo_datos(df):
    """
    Verifica el tipo de dato contenido en cada columna de un dataframe.
    Tiene como parámetro el dataframe a evaluar y devuelve un resumen de el/los tipos de datos,
    porcentaje de nulos y no nulos y cantidad de nulos por cada columna.
    """

    mi_dict = {
        "nombre_campo": [],
        "tipo_datos": [],
        "no_nulos_%": [],
        "nulos_%": [],
        "nulos": [],
    }

    for columna in df.columns:
        porcentaje_no_nulos = (df[columna].count() / len(df)) * 100
        mi_dict["nombre_campo"].append(columna)
        mi_dict["tipo_datos"].append(df[columna].apply(type).unique())
        mi_dict["no_nulos_%"].append(round(porcentaje_no_nulos, 2))
        mi_dict["nulos_%"].append(round(100 - porcentaje_no_nulos, 2))
        mi_dict["nulos"].append(df[columna].isnull().sum())

    df_info = pd.DataFrame(mi_dict)

    return df_info


def boxplot(df, columna):
    """
    Realiza un boxplot sencillo para una columna determinada.
    """
    plt.figure(figsize=(15, 2))
    sns.boxplot(data=df, x=df[columna])
    plt.title(f"Boxplot de la columna {columna}")
    plt.show()


def histplot(df, columna, bins=None):
    """
    Realiza un histplot sencillo para una columna determinada.
    """
    plt.figure(figsize=(5, 3))
    if bins is None:
        sns.histplot(data=df, x=df[columna])
    else:
        sns.histplot(data=df, x=df[columna], bins=bins)
    plt.title(f"Boxplot de la columna {columna}")
    plt.xticks(range(min(df[columna]), max(df[columna]) + 1, 1))

    plt.show()


def histplot_categoricas(df, nombres_columnas):
    ## Crear un grid de 2x2 para los histogramas
    fig, axes = plt.subplots(2, 2, figsize=(8, 5))

    # Obtener las columnas categóricas

    # Iterar a través de las columnas y crear histogramas en cada eje
    for i, column in enumerate(nombres_columnas):
        row = i // 2
        col = i % 2
        sns.histplot(data=df, x=column, ax=axes[row, col], multiple="dodge")

    # Ajustar los espacios entre los gráficos
    plt.tight_layout()

    # Mostrar los gráficos
    plt.show()


def countplot(df, columna):
    """
    Realiza un countplot sencillo para una columna determinada.
    """
    plt.figure(figsize=(14, 3))

    sns.countplot(data=df, y=df[columna])

    plt.title(f"Countplot de la columna {columna}")

    # Ajusta los espacios entre subplots y muestra
    plt.tight_layout()
    plt.show()


def countplot_vertical(df, columna):
    """
    Realiza un countplot sencillo para una columna determinada.
    """
    plt.figure(figsize=(7, 3))

    sns.countplot(data=df, x=df[columna])

    plt.title(f"Countplot de la columna {columna}")

    # Ajusta los espacios entre subplots y muestra
    plt.tight_layout()
    plt.show()


def pairplot(df, hue):
    sns.pairplot(df, hue=hue, diag_kind="hist", palette=["red", "green"])
    plt.show()


def graficar_balanceo(X_train, X_train_resampled, y_train_resampled):
    # Crear un DataFrame con los datos de entrenamiento y la columna 'hospitalizacion'
    df_train_resampled = pd.DataFrame(X_train_resampled, columns=X_train.columns)
    df_train_resampled["hospitalizacion"] = y_train_resampled

    countplot_vertical(df_train_resampled, "hospitalizacion")
    plt.show()


def reporte_train_test(y_train_ros, y_pred_train, y_test_ros, y_pred_test):
    # Obtener los reportes de clasificación como strings
    report_train = classification_report(
        y_train_ros, y_pred_train, target_names=["NO", "SI"], output_dict=True
    )
    report_test = classification_report(
        y_test_ros, y_pred_test, target_names=["NO", "SI"], output_dict=True
    )

    # Convertir los reportes en DataFrames de pandas
    df_report_train = pd.DataFrame(report_train).transpose()
    df_report_test = pd.DataFrame(report_test).transpose()

    # Agregar un nivel de índice a las columnas
    df_report_train.columns = pd.MultiIndex.from_tuples(
        [("Train", col) for col in df_report_train.columns]
    )
    df_report_test.columns = pd.MultiIndex.from_tuples(
        [("Test", col) for col in df_report_test.columns]
    )

    # Concatenar las dos tablas con los niveles de índice en las columnas
    combined_df = pd.concat([df_report_train, df_report_test], axis=1)

    # Imprimir la tabla combinada
    print("Reporte de clasificación en el conjunto de entrenamiento y prueba:")
    return combined_df


def bigote_max(columna):
    """
    Calcula el valor del bigote máximo y la cantidad de valores que se encuentran como valores atípicos.
    """
    # Cuartiles
    q1 = columna.describe()[4]
    q3 = columna.describe()[6]

    # Valor del vigote
    bigote_max = round(q3 + 1.5 * (q3 - q1), 2)
    print(f"El bigote superior de la variable {columna.name} se ubica en:", bigote_max)

    # Cantidad de atípicos
    print(
        f"Hay {(columna > bigote_max).sum()} valores atípicos en la variable {columna.name}"
    )


def valor_mas_frecuente(df, columna):
    """
    Calcula el valore mas frecuente en una columna, su cantidad y porcentaje respecto del total.
    """
    # Frecuencias
    moda = df[columna].mode()[0]
    # Cantidad de la mayor frecuencia
    cantidad = (df[columna] == moda).sum()
    # Total de registros
    total = df[columna].count()
    # Porcentaje de la mayor frecuencia
    porcentaje = round(cantidad / total * 100, 2)
    print(
        f"Valor mas frecuente de {columna} es {moda}, con una cantidad de {cantidad} y representa el {porcentaje}%."
    )


def label_encode_categoricals(df):
    """
    Genera un nuevo dataframe donde se aplica la codificación de etiquetas (label encoding) a las columnas categóricas
    """
    # Separar columnas numéricas y categóricas
    numeric_columns = df.select_dtypes(include=["number"]).columns
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns

    # Crear una copia del DataFrame original
    encoded_df = df.copy()

    # Aplicar label encoding a las columnas categóricas
    label_encoder = LabelEncoder()
    for col in categorical_columns:
        encoded_df[col] = label_encoder.fit_transform(encoded_df[col])

    return encoded_df


def convertir_si_no_edad(value):
    if value >= 65:
        return "mayor_65"
    else:
        return "menor_65"


def convertir_si_no_psa(value):
    if value <= 4.0:
        return "menor_4"
    elif value >= 31.0:
        return "mayor_31"
    else:
        return "entre_4_31"


def convertir_si_no_infecc(value):
    if value >= 1:
        return "SI"
    else:
        return "NO"


def convertir_si_no_muestras(value):
    if value <= 10:
        return "menor_10"
    elif value >= 15:
        return "mayor_20"
    else:
        return "entre_10_15"


def heatmap_categoricas(df):

    # Crea un dataframe de resumen
    summary_df = pd.DataFrame()

    # Iterar a través de cada columna en el DataFrame original
    for column in df.columns:
        # Verificar si la columna es de tipo 'object'
        if df[column].dtype == "object":
            # Obtener las categorías únicas y sus recuentos
            category_counts = df[column].value_counts(normalize=True)

            # Crear un DataFrame temporal para esta columna
            temp_df = pd.DataFrame(
                {
                    "Categoría": category_counts.index,
                    column: category_counts.values * 100,
                }
            )

            # Establecer la columna 'Categoría' como índice para el DataFrame temporal
            temp_df.set_index("Categoría", inplace=True)

            # Unir el DataFrame temporal al resumen general
            summary_df = pd.concat([summary_df, temp_df], axis=1, sort=True)

    # Reemplazar los valores NaN con "null"
    summary_df = summary_df.fillna(-1)

    # crea la visualización
    plt.figure(figsize=(14, 10))

    # Crear una escala de colores personalizada
    def custom_cmap(value, alpha=1.0):
        if value == -1:
            return (1, 1, 1, alpha)  # Blanco para el valor -1
        else:
            color = plt.get_cmap("Reds")(value / 100)
            adjusted_color = (color[0] * 0.9, color[1] * 0.5, color[2] * 0.5, alpha)
            return adjusted_color

    num_colors = 100
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap",
        [custom_cmap(i) for i in range(-1, num_colors + 1)],
        num_colors + 2,
    )

    ax = sns.heatmap(
        summary_df.transpose(),
        cmap=cmap,
        vmin=-1,
        vmax=num_colors,
        square=False,
        annot=True,
        fmt=".1f",
        cbar=False,
        annot_kws={"color": "white"},
        linewidths=0.5,
        linecolor="grey",
    )
    plt.title("Mapa de Calor de Porcentajes")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.show()


def plot_corre_heatmap(corr):
    """
    Definimos una función para ayudarnos a graficar un heatmap de correlación
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr,
        cbar=True,
        square=False,
        annot=True,
        fmt=".1f",
        annot_kws={"size": 15},
        cmap="coolwarm",
    )
    plt.xticks()
    plt.yticks()
    # Arreglamos un pequeño problema de visualización
    b, t = plt.ylim()  # discover the values for bottom and top
    b += 0.5  # Add 0.5 to the bottom
    t -= 0.5  # Subtract 0.5 from the top
    plt.ylim(b, t)  # update the ylim(bottom, top) values
    plt.show()


def GridSearch_MatrixConfusion(
    estimador, param_grid_clf, X_train_ros, y_train_ros, X_test_ros, y_test_ros
):
    # Crea una instancia de GridSearchCV
    grid_search = GridSearchCV(estimator=estimador, param_grid=param_grid_clf, cv=5)

    # Ajusta el GridSearch al conjunto de datos sobremuestreado
    grid_search.fit(X_train_ros, y_train_ros)

    # Obtiene el mejor modelo y sus hiperparámetros
    best_estimador = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Evalúa el modelo en el conjunto de prueba
    y_pred_test = best_estimador.predict(X_test_ros)
    y_pred_train = best_estimador.predict(X_train_ros)

    # Imprime un reporte de clasificación
    print("Mejores hiperparámetros:", best_params)

    # Graficar la matriz de confusión para el conjunto de prueba
    conf_matrix = confusion_matrix(y_test_ros, y_pred_test)
    cm_display = ConfusionMatrixDisplay(conf_matrix)
    cm_display.plot()
    plt.xlabel("Predicciones")
    plt.ylabel("Verdaderos")
    plt.show()
    return y_pred_test, y_pred_train, best_estimador


def GridSearch_model(model, param_grid, X_train, y_train, X_test, y_test):
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    return y_pred_train, y_pred_test, best_model


def resumen_metricas_PCA(results, y_train_ros, y_test_ros):
    # Crear una lista de diccionarios para almacenar los resultados
    summary_results = []

    # Resumir métricas de forma más concisa y almacenar en la lista
    for n, models in results.items():
        for model_name, (y_train_pred, y_test_pred, best_model) in models.items():
            report_train = classification_report(
                y_train_ros, y_train_pred, output_dict=True
            )
            report_test = classification_report(
                y_test_ros, y_test_pred, output_dict=True
            )

            summary_results.append(
                {
                    "Components": n,
                    "Model": model_name,
                    "Train_Precision": report_train["macro avg"]["precision"],
                    "Train_Recall": report_train["macro avg"]["recall"],
                    "Train_F1": report_train["macro avg"]["f1-score"],
                    "Test_Precision": report_test["macro avg"]["precision"],
                    "Test_Recall": report_test["macro avg"]["recall"],
                    "Test_F1": report_test["macro avg"]["f1-score"],
                }
            )

    # Crear un DataFrame a partir de la lista de resultados
    df_summary = pd.DataFrame(summary_results)

    # Imprimir el DataFrame
    return df_summary
