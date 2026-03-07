import os
import requests
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

# 1. CONFIGURAÇÃO DAS CHAVES (Lendo de forma segura do GitHub Actions)
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# 2. CONFIGURAÇÃO DO MODELO (Gemini 2.5 Flash Lite)
gemini_llm = LLM(
    model="gemini/gemini-2.5-flash-lite",
    api_key=os.environ["GOOGLE_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1",
    temperature=0.1
)

# 3. DEFINIÇÃO DAS FERRAMENTAS
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# 4. AGENTES
jornalista = Agent(
    role='Analista de Sentimento B3',
    goal='Analisar o impacto financeiro das notícias de {ticker}',
    backstory='''Você não apenas lê notícias, você entende o mercado. 
    Sua tarefa é dar um Score de -10 (muito negativo) a +10 (muito positivo) 
    para o conjunto de notícias de cada ativo.''',
    tools=[search_tool],
    llm=gemini_llm,
    verbose=True
)

analista_precisao = Agent(
    role='Analista Fundamentalista',
    goal='Extrair indicadores exatos do site Investsite para {ticker}',
    backstory='Sua fonte é o Investsite. Extrai dados brutos sem inventar.',
    tools=[scrape_tool],
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False
)

estrategista_individual = Agent(
    role='Estrategista de Ativo',
    goal='Dar veredito de COMPRA/VENDA/MANTER para {ticker}',
    backstory='Analista técnico que avalia o ativo isoladamente.',
    llm=gemini_llm,
    verbose=True
)

sintetizador = Agent(
    role='Sintetizador de Dados Financeiros',
    goal='Transformar análises longas em resumos técnicos de 5 linhas cada.',
    backstory='Você é especialista em extrair apenas o essencial: Ticker, Veredito e Indicadores.',
    llm=gemini_llm,
    verbose=True
)

rankeador_master = Agent(
    role='Estrategista de Alocação de Portfólio',
    goal='Comparar as análises recebidas e criar um ranking de 1 a 10.',
    backstory='Gestor de fundo que prioriza margem de segurança e ROE.',
    llm=gemini_llm,
    verbose=True
)

# 5. TAREFAS INDIVIDUAIS
t1 = Task(description='Notícias de hoje sobre {ticker}.', expected_output='3 fatos relevantes.', agent=jornalista)
t2 = Task(
    description='Acesse https://www.investsite.com.br/principais_indicadores.php?cod_negociacao={ticker} e extraia P/L, P/VP, ROE, Dividend Yield e Dívida Líquida/EBITDA.',
    expected_output='Tabela de indicadores.',
    agent=analista_precisao
)
t3 = Task(
    description='Forneça um resumo conciso e um veredito final para {ticker}.',
    expected_output='Resumo incluindo indicadores, fatos e veredito final para {ticker}.',
    agent=estrategista_individual
)

# 6. CONFIGURAÇÃO DA EQUIPE
equipe_analise = Crew(
    agents=[jornalista, analista_precisao, estrategista_individual],
    tasks=[t1, t2, t3],
    verbose=True,
    cache=True 
)

# 7. LISTA DE TICKERS
tickers_para_analise = [
    {'ticker': 'BTLG11'}, {'ticker': 'MXRF11'}, {'ticker': 'BBSE3'},
    {'ticker': 'BBAS3'}, {'ticker': 'ITUB4'}, {'ticker': 'TAEE11'},
    {'ticker': 'EGIE3'}, {'ticker': 'SBSP3'}, {'ticker': 'SAPR11'},
    {'ticker': 'VIVT3'}
]

# 8. EXECUÇÃO
print(f"### ANALISANDO {len(tickers_para_analise)} ATIVOS ###")
resultados_individuais = equipe_analise.kickoff_for_each(inputs=tickers_para_analise)

# 9. SINTETIZAÇÃO
contexto_resumido = ""
print("\n### RESUMINDO ANÁLISES PARA O RANKING ###")
for i, res in enumerate(resultados_individuais):
    ativo = tickers_para_analise[i]['ticker']
    tarefa_resumo = Task(
        description=f"Resuma esta análise de {ativo} em no máximo 5 linhas: {res.raw}",
        expected_output="Ticker, Indicadores e Veredito resumidos.",
        agent=sintetizador
    )
    resumo = sintetizador.execute_task(tarefa_resumo)
    contexto_resumido += f"\n{resumo}\n"

# 10. RANKING FINAL (VERSÃO PROFISSIONAL)
tarefa_ranking = Task(
    description=f'''Analise estas 10 oportunidades resumidas:
    {contexto_resumido}
    
    Crie um relatório estruturado seguindo estes critérios:
    1. Classifique cada ativo com um Score de Sentimento (Baseado nas notícias).
    2. Atribua um Status: 🟢 (Compra Forte), 🟡 (Aguardar), 🔴 (Risco/Venda).
    3. Ordene do 1º ao 10º lugar baseado na Margem de Segurança e ROE.''',
    expected_output='''Um relatório em formato de tabela Markdown contendo:
    Posição | Ticker | Status | Sentimento (-10 a 10) | P/VP | Justificativa Curta.
    
    No final, adicione uma nota de 'Destaque do Mês' para o 1º lugar.''',
    agent=rankeador_master
)

print("\n### GERANDO RELATÓRIO ESTRATÉGICO FINAL ###")
resultado_final = rankeador_master.execute_task(tarefa_ranking)

# 11. EXIBIÇÃO E NOTIFICAÇÃO (FORMATO LIMPO)
print(resultado_final)

def enviar_notificacao_celular(mensagem):
    TOPICO_NTFY = "48998304145"
    url = f"https://ntfy.sh/{TOPICO_NTFY}"
    try:
        # Usamos Markdown para que a tabela fique legível no app ntfy
        requests.post(url,
            data=mensagem.encode('utf-8'),
            headers={
                "Title": "Relatorio Estrategico B3",
                "Priority": "high",
                "Tags": "chart_with_upwards_trend,moneybag",
                "Markdown": "yes" 
            }
        )
        print("✅ Relatório profissional enviado com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao enviar: {e}")

enviar_notificacao_celular(str(resultado_final))
