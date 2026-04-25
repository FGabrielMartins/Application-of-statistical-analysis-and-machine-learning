#FastAPI é o framework que transforma funções python em rotas HTTP
from fastapi import FastAPI

#Biblioteca para resolver o problema do CROS (MEcanismo de segurança dos navegaroes, impedindo que sites hospedados no site A façam requisições para o site B)
from fastapi.middleware.cors import CORSMiddleware

#importamos as rotas
from api import upload #rota de upload de CSV
from api import regressao_linear

#INICANDO A APLICAÇÃO

#FastAPI() cria a instância principal da aplicação
app = FastAPI(
    title = "Plataforma de Análise estatística e ML",
    description = "API para cálculos estatíticos, regressão, K-NN, perceptron, Adaline",
    version = "1.0.0"
)

#CONFG DE CORS

#Lisata de quem tem permissão para chamar a API
origins = [
    "http://localhost:4200",
    "http://localhost:3000",
]

#Middleware = código que roda ANTES e DEPOIS de cada requisição
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,          #Quais origens podem acessar
    allow_credentials = True,         #Permite envio de cookies/autenticação
    allow_methods = ["*"],            #Permite todos os métodos: GET, POST, DELETE...
    allow_headers = ["*"],            #Permite todos os cabeçalhos HTTP
)

#REGISTRANDO OS ROTEADORES
#include_router() = "liga" cada roteador a aplicação principal
app.include_router(
    upload.router,
    prefix = "/upload",
    tags = ["Upload"]
)

app.include_router(
    regressao_linear.router,
    prefix = "/regressao_linear",
    tags = ["Regressão linear"]
)

#ROTA RAIZ - healthCheck
#rota mais simples possivel GET serve para ver se o servidor esta online (JSON: {"status": "online", "mensagem":...})

@app.get("/", tags = ["Status"])
def root():
    return{
        "status": "online",
        "mensagem": "API da plataforma de Análise Estatísticos está funcionando!"
    }