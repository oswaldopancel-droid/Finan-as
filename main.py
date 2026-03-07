import os
import requests
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

# 1. CONFIGURAÇÃO DAS CHAVES
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# 2. CONFIGURAÇÃO DO MODELO
gemini_llm = LLM(
    model="gemini/gemini-2.5-flash-lite",
    api_key=os.environ["GOOGLE_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1",
    temperature=0.1
)

# 3. DEFINIÇÃO DAS FERRAMENTAS
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# 4. AGENTES (ALTERADOS PARA VALUE INVESTING)
jornalista = Agent(
    role='Analista de Sentimento B3',
    goal='Analisar o impacto financeiro das notícias de {ticker}',
    backstory='''Você identifica fatos que podem afetar a perpetuidade dos negócios. 
    Sua tarefa é dar um Score de -10 a +10 focado em risco de longo prazo.''',
    tools=[search_tool],
    llm=gemini_llm,
    verbose=True
)

analista_precisao = Agent(
    role='Analista de Valor e Dividendos',
    goal='Extrair indicadores de saúde financeira e histórico de proventos para {ticker}',
    backstory='''Você é um discípulo de Luiz Barsi e Benjamin Graham. 
    Foca em "vacas leiteiras" (geradoras de caixa) e margem de segurança. 
    Busca empresas resilientes com histórico de DY consistente.''',
    tools=[scrape_tool],
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False
)

estrategista_individual = Agent(
    role='Estrategista de Valor',
    goal='Dar veredito de Buy & Hold para {ticker}',
    backstory='Analista focado em valor intrínseco e vantagens competitivas (Moat).',
    llm=gemini_llm,
    verbose=True
)

sintetizador = Agent(
    role='Sintetizador de Dados Financeiros',
    goal='Resumir análises focando em indicadores de valor.',
    backstory='Especialista em condensar dados para tomada de decisão estratégica.',
    llm=gemini_llm,
    verbose=True
)

rankeador_master = Agent(
    role='Estrategista de Alocação (Barsi & Buffett)',
    goal='Comparar ativos e criar ranking focado em Dividendos e Preço Justo.',
    backstory='''Você prioriza o método MAPP (Margem, Administração, Preço e Perpetuidade). 
    Prefere empresas do setor BESST e com Preço Atual abaixo do Preço Justo de Graham.''',
    llm=gemini_llm,
    verbose=True
)

# 5. TAREFAS (ALTERADAS PARA CRITÉRIOS DE LONGO PRAZO)
t1 = Task(description='Notícias de hoje sobre {ticker} focando em governança e resultados.', expected_output='3 fatos relevantes.', agent=jornalista)

t2 = Task(
    description='''Acesse o Investsite para {ticker}:
    1. Extraia P/L, P/VP, ROE, DY atual e Dívida Líquida/EBITDA.
    2. Calcule o Preço Justo de Graham: Raiz Quadrada de (22,5 * VPA * LPA).
    3. Identifique se o DY está acima de 6% (Critério Barsi).''',
    expected_output='Tabela de indicadores + Cálculo do Preço Justo de Graham.',
    agent=analista_precisao
)

t3 = Task(
    description='Forneça um veredito de longo prazo para {ticker} considerando a margem de segurança.',
    expected_output='Resumo com indicadores de valor e veredito (COMPRA/AGUARDAR/VENDA).',
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
for i, res in enumerate(resultados_individuais):
    ativo = tickers_para_analise[i]['ticker']
    tarefa_resumo = Task(
        description=f"Resuma em 5 linhas os dados de valor e dividendos de {ativo}: {res.raw}",
        expected_output="Resumo técnico focado em Buy & Hold.",
        agent=sintetizador
    )
    resumo = sintetizador.execute_task(tarefa_resumo)
    contexto_resumido += f"\n{resumo}\n"

# 10. RANKING FINAL (MÉTODO BESST + VALOR)
tarefa_ranking = Task(
    description=f'''Analise as oportunidades:
    {contexto_resumido}
    
    Crie o ranking seguindo:
    1. Prioridade para Setor BESST (Bancos, Energia, Saneamento, Seguros, Telecom).
    2. Melhor relação Preço Atual vs Preço Justo de Graham.
    3. Status: 🟢 (Desconto > 20%), 🟡 (Preço Justo), 🔴 (Caro/Arriscado).''',
    expected_output='''Tabela Markdown:
    Posição | Ticker | Status | DY % | Margem Graham | Setor | Justificativa de Valor''',
    agent=rankeador_master
)

resultado_final = rankeador_master.execute_task(tarefa_ranking)

# 11. NOTIFICAÇÃO
def enviar_notificacao_celular(mensagem):
    TOPICO_NTFY = "48998304145"
    url = f"https://ntfy.sh/{TOPICO_NTFY}"
    try:
        requests.post(url,
            data=mensagem.encode('utf-8'),
            headers={
                "Title": "Carteira Previdentiaria - Analise Mensal",
                "Priority": "high",
                "Tags": "gem,moneybag",
                "Markdown": "yes" 
            }
        )
        print("✅ Relatório de longo prazo enviado!")
    except Exception as e:
        print(f"❌ Erro: {e}")

enviar_notificacao_celular(str(resultado_final))
