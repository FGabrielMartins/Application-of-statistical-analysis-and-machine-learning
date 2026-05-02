#Login, registro, geração do token
#Rotas disponíveis
#   POST /auth/registrar  → Cria um novo usuário
#   POST /auth/login      → autentica e retorna o token JWT
#   GET  /auth/me         → retorna dados do usuário logado (Rota protegida)

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm     
# OAuth2PasswordRequestForm = formulário padrão de login (username + password)
# O Swagger já gera o formulário automaticamente com isso!

from pydantic import BaseModel
from passlib.context import CryptContext
# passlib = biblioteca de hash de senha
# CryptContext com bcrypt = mesmo método do artigo (flask_bcrypt)

from jose import jwt #Biblioteca que gera e decodifica tokens JWT

from datetime import datetime, timedelta
# timedelta = para calcular quando o token expira

from api.dependencias import (
    SECRET_KEY,
    ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    verificar_token
)

router = APIRouter()

#  CONFIGURAÇÃO DO HASH DE SENHA
pwd_context = CryptContext(schemes = ["pbkdf2_sha256"], deprecated = "auto")
#   CryptContext(...): É uma classe do passlib.context que gerencia múltiplos algoritmos de hash. Ela decide qual algoritmo usar para novas senhas e como verificar senhas antigas.
#   schemes=["bcrypt"]: Define que o algoritmo primário para criptografar as senhas será o bcrypt.
#   deprecated="auto": Configura o Passlib para gerenciar automaticamente a migração de algoritmos. Se o bcrypt se tornar obsoleto no futuro, o passlib tentará usar um mais seguro automaticamente

#   BANCO DE DADOS DOS USUARIOS
usuarios_db: dict = {}

#SCHEMAS - Formatos de entradas

class RegistrarRequest(BaseModel):
    username: str
    password: str

#fUNÇÕES AUXILIARES
def hash_senha(senha: str) -> str:
    """
    Transforma a senha em hash bcrypt.
    Ex: "senha123" → "$2b$12$abc..."
    """
#Equivalente ao bcrypt.generate_password_hash() do artigo.
    return pwd_context.hash(senha)


def verificar_senha(senha_pura: str, senha_hash: str) -> bool:
    """
    Compara a senha digitada com o hash salvo.
    Retorna True se bater, False se não bater.
    """
#Equivalente ao bcrypt.check_password_hash() do artigo.
    return pwd_context.verify(senha_pura, senha_hash)

def criar_token(username: str) -> str:
    """
    Gera o token JWT com o username e tempo de expiração

    O token carrega:
        - "sub": o username (subject — padrão JWT)
        - "exp": quando o token expira
    """
#Equivalente ao create_access_token() do artigo.
    expiracao = datetime.utcnow() + timedelta(minutes = ACCESS_TOKEN_EXPIRE_MINUTES)

    # payload = dados que ficam dentro dele
    payload = {
        "sub": username,     #indentificador do usuario
        "exp": expiracao     # data/hora de expiração
    }

    #jwt.encode() assina e gera o token usando a SECRET_KEY
    token = jwt.encode(payload, SECRET_KEY, algorithm = ALGORITHM)
    return token

#ROTA 1 POST /auth/registrar
@router.post("/registrar", status_code=201)
def registrar(req: RegistrarRequest):
    """
    Registra um novo usuário.
    A senha é armazenada como hash bcrypt — nunca em texto puro.
    """

    # verificar se o username já existe
    if req.username in usuarios_db:  #req.username: Acessa o nome de usuário enviado na requisição (req) atual, geralmente vindo de um formulário ou corpo de uma API 
        raise HTTPException(
            status_code = 400,
            detail = f"Usuário '{req.username}' já existe"
        )
    
    # Salva o username com a senha em hash
    usuarios_db[req.username] = hash_senha(req.password)

    return {
        "mensagem": f"Usuário '{req.username}' criando com sucesso!",
        "dica": "Agora faça login em POST /auth/login para recer seu token."
    }

#ROTA 2 POST /auth/login
#Autentica o usuário e retorna o token JWT.
@router.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends()):
    """
    Autentica o usuário e retorna o token JWT.
 
    O token deve ser enviado em todas as requisições protegidas:
    Header → Authorization: Bearer <token>
    """
    # Verifica se o usuário existe
    if form.username not in usuarios_db:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuário ou senha incorretos.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verifica se a senha bate com o hash salvo
    # Equivalente ao bcrypt.check_password_hash() do artigo
    senha_correta = verificar_senha(form.password, usuarios_db[form.username])
    if not senha_correta:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuário ou senha incorretos.",
            headers={"WWW-Authenticate": "Bearer"},
        )
 
    # Gera o token JWT
    # Equivalente ao create_access_token() do artigo
    token = criar_token(form.username)
 
    # Retorna no formato padrão OAuth2
    # "token_type": "bearer" é obrigatório pelo padrão
    return {
        "access_token": token,
        "token_type": "bearer",
        "mensagem": f"Bem-vindo, {form.username}! Use o token acima para acessar as rotas protegidas."
    }

#ROTA 3 GET /auth/me
#Rota protegida - só funciona com token váliso

@router.get("/me")
def meu_perfil(username: str = Depends(verificar_token)):
    """
    Retorna os dados do usuário autenticado.
    Requer token JWT válido no header Authorization.
    """
    return {
        "username": username,
        "mensagem": f"Olá, {username}! Você está autenticado.",
        "rotas_disponiveis": [
            "POST /upload/csv",
            "GET  /regressao-linear/correlacao",
            "POST /regressao-linear/treinar",
            "POST /regressao-linear/prever",
        ]
    }