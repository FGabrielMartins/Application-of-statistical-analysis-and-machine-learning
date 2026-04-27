from fastapi import APIRouter, HTTPException      #agrupar rotas para o modulo
from pydantic import BaseModel                    #BaseModel define o formato do JSON de entrada
from typing import Literal                        #Literal restringe valores possiveis

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, minmax_scale
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import sys, os                                                    # importa os modulos para manipular o SO e os caminhos de busca python
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
#comentarios de multiplas linhas
'''
__file__: Uma variável especial que contém o caminho completo do arquivo Python atual
os.path.dirname(...): Uma função que obtém o diretório pai de um caminho.
os.path.dirname(os.path.dirname(__file__)): Obtém o diretório pai do diretório pai. Ou seja, sobe dois níveis na árvore de diretórios.
sys.path.insert(0, ...): Insere esse novo diretório na posição 0 (o início) da lista sys.path. Isso garante que o Python procure arquivos nessa pasta
'''
#importa a função get_dataset() do upload, ele retorna o dataframe em memória (ou erro se não ouver dataset)
from api.upload import get_dataset

router = APIRouter()

modelo_treinado = None         #Regressão lienar após o .fit()
scaler_treinado = None         #scaler usado no treino
coluna_x: list = []
coluna_y: str = ""

#Fomatos das entradas das requisições
#pydantic, valida automaticamente o JSON enviado pelo Angular se o campo obrigatório não vier, FastAPI retorna erro 422

class TreinarRequest(BaseModel):
    coluna_alvo: str
    normalizacao: Literal["zscore", "minmax", "nehuma"] = "zscore"
    test_size: float = 0.2     #teste 20%
    random_state: int = 42

class PreverRequest(BaseModel):
    dados: dict


#ROTA 1: GET /regressão-linear/correlação
@router.get("/correlação")
def calcular_correlacao(coluna_alvo: str):
    """
    Calcula a correlação entre as variáveis do dataset.
    
    Retorna:
    - correlação de cada variável com a coluna alvo (questão 1)
    - matriz de correlação entre as variáveis independentes (questão 2)
    - interpretação automática (forte / moderada / fraca)
    """

    df = get_dataset()

    #Valida se a coluna alvo exite no dataset
    if coluna_alvo not in df.columns:
        raise HTTPException(              #o raise força uma execução de erro no código
            status_code=400,
            detail = f"Coluna '{coluna_alvo}' não encontrada. Colunas disponíveis: {df.columns.tolist()}"
        )
    
    #seleciona apenas colunas númericas
    df_num = df.select_dtypes(include = [np.number])

    #correlação de cada variável com a coluna alvo
    x = df_num.drop(columns=[coluna_alvo])
    y = df_num[coluna_alvo]
    correlacao_com_alvo = x.corrwith(y).round(6)

    #Classifica a força de cada correlação
    def classificar(valor):
        abs_valor = abs(valor)  #abs = valor absouluto
        if abs_valor >= 0.8:    return "Forte"
        elif abs_valor >= 0.5:  return "Moderada"
        else:                   return"Fraca"

    correlacao_classificada = {
        col: {
            "Correlação": round(float(correlacao_com_alvo[col], 6)),
            "Força": classificar(correlacao_com_alvo[col]),
            "Direcao": "Positiva" if correlacao_com_alvo[col] > 0 else "Negativa"
        }
        for col in correlacao_com_alvo.index
    }

    #Ordenar por valor absoluto decrescente
    