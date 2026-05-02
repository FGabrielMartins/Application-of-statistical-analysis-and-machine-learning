from fastapi import APIRouter, HTTPException
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from api.upload import get_dataset
 
router = APIRouter()

#funções para calculos
def calcular_status_coluna(serie):
    """
    Calcula todas as estatísticas de uma coluna numérica.
 
    Parâmetro:
        serie → uma coluna do DataFrame (pd.Series)
 
    Retorna:
        dicionário com todas as estatísticas calculadas
    """
    # --- QUARTIS E IQR ---
    # np.percentile calcula o valor em que X% dos dados estão abaixo
    q1  = float(np.percentile(serie, 25))   # 25% dos dados abaixo daqui
    q2  = float(np.percentile(serie, 50))   # mediana
    q3  = float(np.percentile(serie, 75))   # 75% dos dados abaixo daqui
    iqr = q3 - q1                           # Intervalo Interquartil = Q3 - Q1

    #LIMITES PARA OUTLIERS
    #regra de tukey = ex:.
    #Limite inferior = Q1 - 1.5 × IQR = 22 - 1.5 × 8 = 22 - 12 = 10
    #Limite superior = Q3 + 1.5 × IQR = 30 + 1.5 × 8 = 30 + 12 = 42
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr

    #OUTLIER (Regra de Tukey: valores fora desses limites são considerados outliers)
    outliers = serie[(serie < limite_inferior) | (serie > limite_superior)]

    #ASSIMETRIA (SKEWNESS) Mede se os dados pendem para um lado:
    skewness = float(serie.skew())

    #CURTOSE (KURTOSIS) me de o achatamento da distribuição
    kurtosis = float(serie.kurt())

    #INTERPRETAÇÃO ASSIMETRICA EM TEXTO
    if skewness > 0.5:
        interpretacao_skew = "Assimetria positiva (cauda à direita)"
    elif skewness < -0.5:
        interpretacao_skew = "Assimetria negativa (cauda à esquerda)"
    else:
        interpretacao_skew = "Aproximadamente simétrica"

    return {
        #estatisticas básicas
        "count":  int(serie.count()),            # quantidade de valores não nulos
        "mean":   round(float(serie.mean()), 4), # média aritmética
        "std":    round(float(serie.std()), 4),  # desvio padrão
        "min":    round(float(serie.min()), 4),  # menor valor
        "max":    round(float(serie.max()), 4),  # maior valor

        # Quartis
        "Q1":  round(q1, 4),   # 1º quartil (25%)
        "Q2":  round(q2, 4),   # 2º quartil / mediana (50%)
        "Q3":  round(q3, 4),   # 3º quartil (75%)
        "IQR": round(iqr, 4),  # amplitude interquartil

        # Limites do boxplot
        "limite_inferior": round(limite_inferior, 4),
        "limite_superior": round(limite_superior, 4),

        # Outliers
        "outliers_count":  len(outliers),
        "outliers_valores": [round(float(v), 4) for v in outliers.values],

        # Forma da distribuição
        "skewness": round(skewness, 4),
        "kurtosis": round(kurtosis, 4),
        "interpretacao_assimetria": interpretacao_skew,
    }

#ROTA 1 GET /estatistica/resumo
@router.get("/resumo")
def resumo_estatistico():
    """
    Retorna estatísticas completas de todas as colunas numéricas:
    média, desvio padrão, quartis, IQR, outliers, assimetria e curtose.
    """

    df = get_dataset()

    # Seleciona apenas colunas numéricas (ignora colunas de texto)
    df_num = df.select_dtypes(include=[np.number])

    # Calcula as estatísticas para cada coluna
    resultado = {}
    for coluna in df_num.columns:
        resultado[coluna] = calcular_status_coluna(df_num[coluna])
 
    return {
        "total_linhas": len(df),
        "total_colunas_numericas": len(df_num.columns),
        "colunas": df_num.columns.tolist(),
        "estatisticas": resultado
    }

#ROTA 2: GET /estatisticas/dotplot
@router.get("/dotplot")
def dados_dotplot(coluna: str):
    """
    Retorna os dados de uma coluna formatados para o dotplot.
 
    Retorna:
    - lista de todos os valores (para plotar ponto por ponto)
    - frequência de cada valor (para dotplot empilhado)
    - estatísticas básicas para exibir junto ao gráfico
    """

    df = get_dataset()

    #valida se a coluna existe
    if coluna not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Coluna '{coluna}' não encontrada. Disponíveis: {df.columns.tolist()}"
        )
    
    # Valida se a coluna é numérica
    if not np.issubdtype(df[coluna].dtype, np.number):
        raise HTTPException(
            status_code=400,
            detail=f"Coluna '{coluna}' não é numérica. O dotplot só funciona com números."
        )
    
    serie = df[coluna].dropna()  # remove valores nulos

    #LISTA TODOS OS VALORES
    todos_os_valores = [round(float(v), 4) for v in serie.values]

    #FREQUÊNCIA DE CADA VALOR
    frequencia = serie.value_counts().sort_index()
    frequencia_dict = {
        round(float(k), 4): int(v)
        for k, v in frequencia.items()
    }

    # Estatísticas básicas para exibir junto ao gráfico
    stats = calcular_status_coluna(serie)
 
    return {
        "coluna": coluna,
        "total_pontos": len(todos_os_valores),
 
        # Lista bruta de valores — para plotar ponto por ponto
        "valores": todos_os_valores,
 
        # Frequência — para dotplot empilhado
        # Formato: [{"valor": 18, "frequencia": 5}, ...]
        "frequencia": [
            {"valor": k, "frequencia": v}
            for k, v in frequencia_dict.items()
        ],
 
        # Estatísticas para exibir no painel ao lado do gráfico
        "estatisticas": {
            "min":  stats["min"],
            "max":  stats["max"],
            "mean": stats["mean"],
            "std":  stats["std"],
            "Q1":   stats["Q1"],
            "Q2":   stats["Q2"],
            "Q3":   stats["Q3"],
        }
    }