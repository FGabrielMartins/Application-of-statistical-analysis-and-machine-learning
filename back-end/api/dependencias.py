#Função que valida o token
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
# OAuth2PasswordBearer = diz ao FastAPI que o token vem no header
# Authorization: Bearer <token>

from jose import JWTError, jwt

#  CONFIGURAÇÕES DO JWT
#  SECRET_KEY: chave secreta usada para assinar o token.
#  ALGORITHM: algoritmo de assinatura — HS256 é o padrão
#  ACCESS_TOKEN_EXPIRE_MINUTES: tempo de vida do token.

SECRET_KEY = "sua_chave_secreta_troque_em_produção"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 20  #20 minutos

#  OAUTH2 SCHEME
#
#  tokenUrl="/auth/login" diz ao FastAPI onde fica a rota de login.
#  Isso aparece automaticamente na documentação /docs.
#  Quando o usuário clica em "Authorize" no Swagger, ele usa essa URL.
oauth2_schema = OAuth2PasswordBearer(tokenUrl = "/auth/login")

#  FUNÇÃO: verificar_token

def verificar_token(token: str = Depends(oauth2_schema)):
    """
    Valida o token JWT enviado no header da requisição.
    Retorna o username do usuário autenticado.
    Lança HTTPException 401 se o token for inválido ou expirado.
    """

    # cria a exceção que será lançada se algo der errado
    # status.HTTP_401_UNAUTHORIZED = erro padrão para "não autenticado"
    # WWW-Authenticate: Bearer = padrão HTTP para dizer que precisa de token

    credenciais_invalidas = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token inválido ou expirado. Faça login novamente.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # jwt.decode() decodifica e valida o token
        # Se o token foi alterado ou expirou → lança JWTError
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
 
        # O payload é o "recheio" do token — dados que guardamos nele
        # "sub" (subject) = identificador do usuário, padrão JWT
        username: str = payload.get("sub")
 
        # Se não encontrou o username no payload → token inválido
        if username is None:
            raise credenciais_invalidas
 
        return username   # retorna o username para a rota usar
 
    except JWTError:
        # JWTError cobre: token expirado, assinatura inválida, formato errado
        raise credenciais_invalidas