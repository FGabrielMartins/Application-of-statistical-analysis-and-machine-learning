from fastapi import APIRouter, HTTPException      #agrupar rotas para o modulo
from pydantic import BaseModel                    #BaseModel define o formato do JSON de entrada
from typing import Literal                        #Literal restringe valores possiveis

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

from api.dependencias import verificar_token
from fastapi import Depends

router = APIRouter()

modelo_treinado = None         #Regressão lienar após o .fit()
scaler_treinado = None         #scaler usado no treino
colunas_x: list = []
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
@router.get("/correlacao")
def calcular_correlacao(coluna_alvo: str, usario: str = Depends(verificar_token)):
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
            "correlacao": round(float(correlacao_com_alvo[col]), 6),
            "forca": classificar(correlacao_com_alvo[col]),
            "direcao": "Positiva" if correlacao_com_alvo[col] > 0 else "Negativa"
        }
        for col in correlacao_com_alvo.index
    }

    #Ordenar por valor absoluto decrescente
    ranking = sorted(
        correlacao_classificada.items(),
        key = lambda item: abs(item[1]["correlacao"]),
        reverse=True
    )

    matriz_correlacao = x.corr().round(6).to_dict()

    #detecta pares com multicolinearidade forte >0.8
    pares_multicolinearidade = []
    colunas = x.columns.tolist()
    for i in range(len(colunas)):
        for j in range(i+1, len(colunas)):
            val = x.corr().iloc[i,j]
            if abs(val) >= 0.8:
                pares_multicolinearidade.append({
                    "variavel_1": colunas[i],
                    "variavel_2": colunas[j],
                    "correlacao": round(float(val), 6),
                    "alerta": "Possível multicolinearidade!"
                })
 
    return {
        "coluna_alvo": coluna_alvo,
        "variaveis_independentes": colunas,
 
        "correlacao_com_alvo": {col: dados for col, dados in ranking},
        "ranking": [col for col, _ in ranking],
 
        "matriz_correlacao_independentes": matriz_correlacao,
        "multicolinearidade_detectada": pares_multicolinearidade,
    }

#Rota 2: post / Treinar a regressão linear
@router.post("/treinar")
def treinar_modelo(req: TreinarRequest, usario: str = Depends(verificar_token)):
    """
    Treina o modelo de regressão linear múltipla.
    
    1. Separa X e y
    2. Normaliza X (z-score ou min-max)
    3. Divide em treino/teste
    4. Treina o LinearRegression
    5. Retorna R², MSE, RMSE, MAE e os coeficientes θ
    """
    global modelo_treinado, scaler_treinado, colunas_x, coluna_y
 
    df = get_dataset()
 
    # Valida coluna alvo
    if req.coluna_alvo not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Coluna '{req.coluna_alvo}' não encontrada. Disponíveis: {df.columns.tolist()}"
        )
 
    # Seleciona apenas colunas numéricas
    df_num = df.select_dtypes(include=[np.number])
 
    # --- SEPARANDO X e Y ---
    # Seu código original:
    #   x = df.drop(columns=['PE'])
    #   y = df['PE']
    x = df_num.drop(columns=[req.coluna_alvo])
    y = df_num[req.coluna_alvo]
 
    colunas_x = x.columns.tolist()   # salva para usar na previsão
    coluna_y = req.coluna_alvo
 
    # --- NORMALIZAÇÃO ---
    # Seu código original: StandardScaler() ou MinMaxScaler()
    if req.normalizacao == "zscore":
        scaler = StandardScaler()
        x_norm = scaler.fit_transform(x)   # fit() calcula μ e σ, transform() aplica
    elif req.normalizacao == "minmax":
        scaler = MinMaxScaler()
        x_norm = scaler.fit_transform(x)   # fit() acha min/max, transform() aplica
    else:
        scaler = None
        x_norm = x.values                  # sem normalização, usa os valores brutos
 
    scaler_treinado = scaler  # salva para usar na previsão depois
 
    # --- DIVISÃO TREINO/TESTE ---
    # Seu código original: train_test_split(x, y, test_size=0.2, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(
        x_norm, y,
        test_size=req.test_size,
        random_state=req.random_state
    )
 
    # --- TREINAMENTO ---
    # Seu código original: model = LinearRegression(); model.fit(x_train, y_train)
    model = LinearRegression()
    model.fit(x_train, y_train)
 
    modelo_treinado = model   # salva o modelo treinado em memória
 
    # --- MÉTRICAS ---
    # Seu código original: model.predict, r2_score, mean_squared_error, etc.
    y_pred = model.predict(x_test)
 
    r2_treino = model.score(x_train, y_train)
    r2_teste  = r2_score(y_test, y_pred)
    mse       = mean_squared_error(y_test, y_pred)
    rmse      = float(np.sqrt(mse))
    mae       = mean_absolute_error(y_test, y_pred)
 
    # Diagnóstico automático (tabela do seu notebook)
    if r2_treino > 0.9 and r2_teste < r2_treino - 0.1:
        diagnostico = "Overfitting — treino muito melhor que teste"
    elif r2_teste >= 0.8:
        diagnostico = "Modelo muito bom"
    elif r2_teste >= 0.6:
        diagnostico = "Modelo razoável"
    else:
        diagnostico = "Modelo fraco"
 
    # --- COEFICIENTES THETA ---
    # Seu código original: theta = model.coef_
    theta = {col: round(float(coef), 8) for col, coef in zip(colunas_x, model.coef_)}
    intercepto = round(float(model.intercept_), 8)   # θ₀
 
    # Monta a equação como string (igual ao seu notebook)
    equacao = f"ŷ = {intercepto}"
    for col, coef in theta.items():
        sinal = "+" if coef >= 0 else "-"
        equacao += f" {sinal} {abs(coef)}·{col}"
 
    return {
        "modelo": "Regressão Linear Múltipla",
        "coluna_alvo": req.coluna_alvo,
        "variaveis_independentes": colunas_x,
        "normalizacao": req.normalizacao,
 
        "tamanhos": {
            "total":  len(df),
            "treino": len(x_train),   # ~80%
            "teste":  len(x_test),    # ~20%
        },
 
        "metricas": {
            "R2_treino": round(r2_treino, 4),
            "R2_teste":  round(r2_teste, 4),
            "MSE":       round(mse, 4),
            "RMSE":      round(rmse, 4),
            "MAE":       round(mae, 4),
            "diagnostico": diagnostico,
        },
 
        "coeficientes": {
            "intercepto_theta0": intercepto,
            "thetas": theta,
            "equacao": equacao,
        },
 
        # Primeiros 20 valores reais vs previstos (para o Angular plotar o gráfico)
        "comparacao_real_vs_previsto": [
            {"real": round(float(r), 4), "previsto": round(float(p), 4)}
            for r, p in zip(list(y_test)[:20], list(y_pred)[:20])
        ]
    }
 
#  ROTA 3: POST
@router.post("/prever")
def prever(req: PreverRequest, usario: str = Depends(verificar_token)):
    """
    Usa o modelo já treinado para prever um novo valor.
    Chame /treinar antes de usar esta rota.
    """
    if modelo_treinado is None:
        raise HTTPException(
            status_code=400,
            detail="Nenhum modelo treinado. Chame POST /regressao-linear/treinar primeiro."
        )

    # Valida se todas as colunas necessárias foram enviadas
    faltando = [col for col in colunas_x if col not in req.dados]
    if faltando:
        raise HTTPException(
            status_code=400,
            detail=f"Campos faltando no JSON: {faltando}. Necessários: {colunas_x}"
        )

    # Monta o DataFrame com os novos dados na ordem correta
    novo_dado = pd.DataFrame([req.dados])[colunas_x]

    # Aplica a mesma normalização usada no treino
    if scaler_treinado is not None:
        novo_dado_norm = scaler_treinado.transform(novo_dado)
    else:
        novo_dado_norm = novo_dado.values

    # Faz a previsão
    previsao = modelo_treinado.predict(novo_dado_norm)

    return {
        "dados_entrada": req.dados,
        "coluna_prevista": coluna_y,
        "previsao": round(float(previsao[0]), 4),
    }