import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Configura as permissões
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
client = gspread.authorize(creds)

# Abre a planilha pelo nome exato
ss = client.open("FINANCEIRO 2026")
sheet = ss.sheet1 # Seleciona a primeira aba

# Exemplo: Lendo tickers da Coluna A e limpando dados
tickers = sheet.col_values(1)[1:] # Pula o cabeçalho

for i, ticker in enumerate(tickers, start=2):
    # Aqui você chamaria sua função de busca (ex: analista_precisao)
    # Supondo que você buscou o LPA real e o DPA real:
    lpa_real = 1.67  # Exemplo Taesa
    dpa_real = 3.24
    payout = (dpa_real / lpa_real) # Python faz a conta sem erro de célula
    
    # Atualiza as colunas L (LPA) e N (Payout)
    sheet.update_cell(i, 12, lpa_real) 
    sheet.update_cell(i, 14, f"{payout:.2%}") # Já envia formatado como %
    print(f"✅ {ticker} atualizado com Payout de {payout:.2%}")
