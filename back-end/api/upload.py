#Agrupa rotas de um módulo
from fastapi import APIRouter, UploadFile, File, HTTPException

#io.StringIO transforma texto em um "arquivo virtual" que o pandar consegue ler
import io

import pandas as pd

#CRIANDO O ROTEADOR

#Todas as rotas ficam em /upload/...
router = APIRouter()

#ARMAZENAMENTO DO DATASET EM MÉMORIA

#inicialmente em uma variavel global
dataset_global: pd.DataFrame | None = None #começa vazio

def get_dataset() -> pd.DataFrame:
    """
    Função auxiliar usada pelos outros módulos (regressão, K=NN, etc.)
    para acessar p dataset que o usuário fez upload

    Se nenhum dataset foi enviado ainda, lança um erro HTTP 400.
    """
    if dataset_global is None:
        raise HTTPException(
            status_code =  400,
            detail = "Nenhum dataset foi enviado ainda. Faça upload de um CSV primeiro"
        )
    return dataset_global

@router.post("/csv")
async def upload_csv(
    file: UploadFile = File(...)        # File(...) = parâmetro obrigatório do tipo arquivo
):
    """
    Recebe um arquivo CSV, valida e salva em memória.
    
    Retorna um preview das primeiras 5 linhas e informações do dataset.
    """
 
    # --- VALIDAÇÃO: só aceita arquivos .csv ---
    # file.filename é o nome original do arquivo enviado pelo usuário
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,           # 400 = Bad Request (requisição inválida)
            detail="Arquivo inválido. Envie apenas arquivos .csv"
        )
 
    # --- LEITURA DO ARQUIVO ---
    # file.read() lê os bytes brutos do arquivo enviado
    # await = aguarda a leitura terminar (operação assíncrona)
    conteudo_bytes = await file.read()
 
    # Decodifica os bytes para texto (UTF-8 é o padrão para CSVs)
    conteudo_texto = conteudo_bytes.decode("utf-8")
 
    # io.StringIO transforma o texto em um "arquivo virtual"
    # para o pandas conseguir ler como se fosse um arquivo real
    arquivo_virtual = io.StringIO(conteudo_texto)
 
    # pd.read_csv lê o "arquivo virtual" e cria um DataFrame
    df = pd.read_csv(arquivo_virtual)
 
    # --- SALVANDO NA VARIÁVEL GLOBAL ---
    # "global" diz ao Python para usar a variável do escopo global
    # (fora da função), não criar uma nova variável local
    global dataset_global
    dataset_global = df
 
    # --- MONTANDO A RESPOSTA ---
    # Retornamos informações úteis sobre o dataset carregado:
    return {
        "mensagem": f"Dataset '{file.filename}' carregado com sucesso!",
 
        # shape retorna (linhas, colunas) — ex: (150, 5)
        "linhas": df.shape[0],
        "colunas": df.shape[1],
 
        # df.columns.tolist() transforma o índice de colunas em lista Python
        # ex: ["idade", "nota", "horas_estudo"]
        "nomes_colunas": df.columns.tolist(),
 
        # df.dtypes retorna o tipo de dado de cada coluna
        # astype(str) converte para string para serializar em JSON
        # to_dict() transforma em dicionário Python
        # ex: {"idade": "int64", "nota": "float64"}
        "tipos_colunas": df.dtypes.astype(str).to_dict(),
 
        # df.head(5) pega as 5 primeiras linhas
        # .to_dict(orient="records") transforma em lista de dicionários
        # ex: [{"idade": 20, "nota": 8.5}, {"idade": 22, "nota": 7.0}, ...]
        "preview": df.head(5).to_dict(orient="records"),
    }
 
 
# ==============================================================
#  ROTA: GET /upload/info
#
#  Retorna informações sobre o dataset atualmente carregado.
#  Útil para o Angular verificar se já tem um dataset ativo.
# ==============================================================
 
@router.get("/info")
def info_dataset():
    """
    Retorna informações do dataset atualmente em memória.
    """
    df = get_dataset()   # se não houver dataset, lança erro automaticamente
 
    return {
        "linhas": df.shape[0],
        "colunas": df.shape[1],
        "nomes_colunas": df.columns.tolist(),
        "tipos_colunas": df.dtypes.astype(str).to_dict(),
 
        # describe() calcula estatísticas básicas: média, min, max, desvio padrão...
        # round(4) = arredonda para 4 casas decimais
        "estatisticas_basicas": df.describe().round(4).to_dict(),
    }
 
 
# ==============================================================
#  ROTA: DELETE /upload/limpar
#
#  Remove o dataset da memória.
# ==============================================================
 
@router.delete("/limpar")
def limpar_dataset():
    """
    Remove o dataset da memória.
    """
    global dataset_global
    dataset_global = None
    return {"mensagem": "Dataset removido da memória com sucesso."}