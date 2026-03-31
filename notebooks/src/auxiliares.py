# Importações
import pandas as pd

# Main

def dataframe_coeficientes(coefs, colunas):
    return pd.DataFrame(data=coefs, 
                        index=colunas, 
                        columns=["coeficiente"]).sort_values(by="coeficiente")


def inspect_outliers(
        dataframe, column, 
        whisker_width=1.5,
        view='both'
):

    """
    Identifica e retorna as linhas do DataFrame que contêm outliers em uma coluna específica.
    
    Baseia-se no método do Intervalo Interquartil (IQR) para definir limites
    inferiores e superiores de detecção.
    
    Args:
        dataframe (pd.DataFrame): O conjunto de dados original.
        column (str): Nome da coluna numérica a ser inspecionada.
        whisker_width (float): Multiplicador do IQR (padrão 1.5).
        view (str): Define quais outliers retornar ('both', 'lower' ou 'upper').
        
    Returns:
        pd.DataFrame: Subconjunto dos dados contendo apenas os outliers detectados.
    """


    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - whisker_width * iqr
    upper_bound = q3 + whisker_width * iqr

    if view == 'both':
        outliers = dataframe[
            (dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)
        ]

    elif view == 'lower':
        outliers = dataframe[
            dataframe[column] < lower_bound
        ]

    elif view == 'upper':
        outliers = dataframe[
            dataframe[column] > upper_bound
        ]
    
    else:
        raise ValueError("O parâmetro 'view' deve ser 'both', 'lower' ou 'upper'.")

    return outliers