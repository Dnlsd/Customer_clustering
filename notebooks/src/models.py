# Importações
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42

# Main

def construir_pipeline_modelo_regressao(
    regressor, 
    preprocessor=None, 
    target_transformer=None
):
    
    """
    Constrói um objeto de modelagem do Scikit-Learn integrando pré-processamento e transformação do alvo.
    A função encapsula um regressor em um Pipeline se um pré-processador for fornecido e, 
    opcionalmente, aplica uma transformação na variável dependente (y) usando TransformedTargetRegressor.

    Args:
        regressor (BaseEstimator): O modelo de regressão do Scikit-Learn a ser utilizado.
        preprocessor (TransformerMixin, opcional): Objeto ColumnTransformer ou similar para 
            tratamento de features. Padrão é None.
        target_transformer (TransformerMixin, opcional): Transformador para a variável alvo 
            (ex: PowerTransformer ou Log). Padrão é None.

    Returns:
        Pipeline ou TransformedTargetRegressor: O objeto de estimador final pronto para ajuste (fit).
    """

    if preprocessor is not None:
        pipeline = Pipeline([("preprocessor", preprocessor), ("reg", regressor)])
    else:
        pipeline = Pipeline([("reg", regressor)])

    if target_transformer is not None:
        model = TransformedTargetRegressor(
            regressor=pipeline, transformer=target_transformer
        )
    else:
        model = pipeline
    return model


def treinar_e_validar_modelo_regressao(
    X,
    y,
    regressor,
    preprocessor=None,
    target_transformer=None,
    n_splits=5,
    random_state=RANDOM_STATE,
):

    """
    Realiza a validação cruzada de um modelo de regressão utilizando K-Fold.

    Args:
        X (array-like ou pd.DataFrame): Matriz de variáveis independentes.
        y (array-like ou pd.Series): Vetor da variável dependente.
        regressor (BaseEstimator): O modelo de regressão a ser avaliado.
        preprocessor (TransformerMixin, opcional): Pré-processador para as features. Padrão é None.
        target_transformer (TransformerMixin, opcional): Transformador para a variável alvo. Padrão é None.
        n_splits (int, opcional): Número de dobras (folds) para a validação cruzada. Padrão é 5.
        random_state (int, opcional): Semente para reprodutibilidade do embaralhamento do K-Fold.

    Returns:
        dict: Dicionário contendo os tempos de execução e os scores das métricas para cada fold.
    """


    model = construir_pipeline_modelo_regressao(
        regressor, preprocessor, target_transformer
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = cross_validate(
        model,
        X,
        y,
        cv=kf,
        scoring=[
            "r2",
            "neg_mean_absolute_error",
            "neg_root_mean_squared_error",
        ],
    )

    return scores


def grid_search_cv_regressor(
    regressor,
    param_grid,
    preprocessor=None,
    target_transformer=None,
    n_splits=5,
    random_state=RANDOM_STATE,
    return_train_score=False,
):
    
    """
    Configura a busca exaustiva de hiperparâmetros (GridSearch) para um pipeline de regressão.

    Args:
        regressor (BaseEstimator): O modelo cujos parâmetros serão otimizados.
        param_grid (dict): Dicionário com os nomes dos parâmetros e os valores para testar.
        preprocessor (TransformerMixin, opcional): Pré-processador para as features. Padrão é None.
        target_transformer (TransformerMixin, opcional): Transformador para a variável alvo. Padrão é None.
        n_splits (int, opcional): Número de dobras para a validação interna. Padrão é 5.
        random_state (int, opcional): Semente para o K-Fold interno.
        return_train_score (bool, opcional): Se True, calcula as métricas também para o treino. Padrão é False.

    Returns:
        GridSearchCV: Objeto configurado para a busca de parâmetros, pronto para o método .fit().
    """



    model = construir_pipeline_modelo_regressao(
        regressor, preprocessor, target_transformer
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        model,
        cv=kf,
        param_grid=param_grid,
        scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
        refit="neg_root_mean_squared_error",
        n_jobs=-1,
        return_train_score=return_train_score,
        verbose=1,
    )

    return grid_search


def organiza_resultados(resultados):

    """
    Processa e formata um dicionário de resultados de modelos em um DataFrame expandido.
    A função calcula o tempo total (fit + score), transpõe os dados para um formato tabular
    e utiliza o método 'explode' para transformar listas de scores de validação cruzada 
    em linhas individuais, facilitando a criação de boxplots e análises estatísticas.

    Args:
        resultados (dict): Dicionário onde as chaves são os nomes dos modelos e os valores 
            são os dicionários de métricas retornados por cross_validate.

    Returns:
        pd.DataFrame: DataFrame no formato 'long', pronto para visualização com Seaborn,
            com colunas numéricas convertidas e índice resetado.
    """

    for chave, valor in resultados.items():
        resultados[chave]["time_seconds"] = (
            resultados[chave]["fit_time"] + resultados[chave]["score_time"]
        )

    df_resultados = (
        pd.DataFrame(resultados).T.reset_index().rename(columns={"index": "model"})
    )

    df_resultados_expandido = df_resultados.explode(
        df_resultados.columns[1:].to_list()
    ).reset_index(drop=True)

    try:
        df_resultados_expandido = df_resultados_expandido.apply(pd.to_numeric)
    except ValueError:
        pass

    return df_resultados_expandido
