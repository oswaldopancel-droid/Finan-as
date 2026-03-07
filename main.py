import os
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

# 1. CONFIGURAÇÃO DAS CHAVES (Lendo dos Secrets do GitHub)
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

# 4. AGENTES (VALUE INVESTING)
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

# 5. TAREFAS
t1 = Task(description='Notícias de hoje sobre {ticker} focando em governança e resultados.', expected_output='3 fatos relevantes.', agent=jornalista)

t2 = Task(
    description='''Busque os indicadores para {ticker} seguindo esta ordem de prioridade:
    1. Tente extrair de: https://www.investsite.com.br/principais_indicadores.php?cod_negociacao={ticker}
    2. Se falhar, tente: https://statusinvest.com.br/acoes/{ticker} ou https://statusinvest.com.br/fundos-imobiliarios/{ticker}
    3. Se ainda falhar, tente: https://www.fundamentus.com.br/detalhes.php?papel={ticker}

    DADOS OBRIGATÓRIOS: P/L, P/VP, ROE, DY atual, Dívida Líquida/EBITDA, VPA e LPA.
    CÁLCULO: Preço Justo de Graham = Raiz Quadrada de (22,5 * VPA * LPA).''',
    expected_output='Tabela completa de indicadores e o Preço Justo calculado.',
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
    {'ticker': 'CPFE3'}, {'ticker': 'PSSA3'}, {'ticker': 'BBSE3'},
    {'ticker': 'BBAS3'}, {'ticker': 'ITUB4'}, {'ticker': 'TAEE11'},
    {'ticker': 'EGIE3'}, {'ticker': 'SBSP3'}, {'ticker': 'SAPR11'},
    {'ticker': 'VIVT3'}
]

# 8. EXECUÇÃO
print(f"### ANALISANDO {len(tickers_para_analise)} ATIVOS ###")
resultados_individuais = equipe_analise.kickoff_for_each(inputs=tickers_para_analise)

# 9. SINTETIZAÇÃO E PREPARAÇÃO DO RELATÓRIO DETALHADO
contexto_resumido = ""
relatorio_detalhado_email = "<h1>Análise Detalhada por Ativo</h1>"

print("\n### PROCESSANDO RESULTADOS PARA CELULAR E E-MAIL ###")
for i, res in enumerate(resultados_individuais):
    ativo = tickers_para_analise[i]['ticker']
    
    # 9a. Criar contexto para o Rankeador (Celular)
    tarefa_resumo = Task(
        description=f"Resuma em 5 linhas os dados de valor e dividendos de {ativo}: {res.raw}",
        expected_output="Resumo técnico focado em Buy & Hold.",
        agent=sintetizador
    )
    resumo = sintetizador.execute_task(tarefa_resumo)
    contexto_resumido += f"\n{resumo}\n"
    
    # 9b. Acumular a análise completa para o E-mail
    relatorio_detalhado_email += f"""
    <div style="border-bottom: 1px solid #ccc; padding: 10px;">
        <h2 style="color: #2c3e50;">{ativo}</h2>
        <div style="white-space: pre-wrap;">{res.raw}</div>
    </div>
    """

# 10. RANKING FINAL (PARA O CELULAR)
tarefa_ranking = Task(
    description=f'''Analise as oportunidades:
    {contexto_resumido}
    
    Crie o ranking seguindo:
    1. Prioridade para Setor BESST.
    2. Melhor relação Preço Atual vs Preço Justo de Graham.
    3. Status: 🟢 (Desconto > 20%), 🟡 (Preço Justo), 🔴 (Caro/Arriscado).''',
    expected_output='''Tabela Markdown:
    Posição | Ticker | Status | DY % | Margem Graham | Setor | Justificativa de Valor''',
    agent=rankeador_master
)

resultado_ranking_celular = str(rankeador_master.execute_task(tarefa_ranking))

# 11. NOTIFICAÇÕES (DIFERENCIADAS)

def enviar_notificacao_celular(mensagem_tabela):
    TOPICO_NTFY = "48998304145"
    url = f"https://ntfy.sh/{TOPICO_NTFY}"
    try:
        requests.post(url,
            data=mensagem_tabela.encode('utf-8'),
            headers={
                "Title": "Ranking Estratégico B3",
                "Priority": "high",
                "Tags": "chart_with_upwards_trend,moneybag",
                "Markdown": "yes" 
            }
        )
        print("✅ Ranking enviado para o celular!")
    except Exception as e:
        print(f"❌ Erro ntfy: {e}")

def enviar_email_relatorio_completo(tabela_ranking, analise_detalhada):
    email_user = os.getenv("EMAIL_USER")
    email_password = os.getenv("EMAIL_PASSWORD")
    
    if not email_user or not email_password:
        print("⚠️ E-mail não enviado: Credenciais ausentes.")
        return

    msg = MIMEMultipart()
    msg['From'] = email_user
    msg['To'] = email_user
    msg['Subject'] = "📊 Relatório Completo de Investimentos - Buy & Hold"

    # Corpo do e-mail unindo a Tabela + Análises Detalhadas
    corpo_html = f"""
    <html>
    <body style="font-family: Calibri, sans-serif; line-height: 1.6;">
        <h1 style="color: #1a5276;">Relatório Estratégico Mensal</h1>
        <p>Abaixo o ranking consolidado e, em seguida, a análise profunda de cada ativo.</p>
        
        <div style="background-color: #f8f9fa; border: 1px solid #dcdcdc; padding: 15px; border-radius: 5px;">
            <h2 style="color: #2c3e50;">Ranking de Alocação</h2>
            <pre style="font-size: 14px;">{tabela_ranking}</pre>
        </div>
        
        <hr style="margin: 30px 0;">
        
        {analise_detalhada}
        
        <br>
        <p style="font-size: 12px; color: #7f8c8d;"><i>Gerado automaticamente via CrewAI & Gemini LLM.</i></p>
    </body>
    </html>
    """
    msg.attach(MIMEText(corpo_html, 'html'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(email_user, email_password)
            server.send_message(msg)
        print("✅ E-mail detalhado enviado com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao enviar e-mail: {e}")

# Execução final
print(resultado_ranking_celular)
enviar_notificacao_celular(resultado_ranking_celular)
enviar_email_relatorio_completo(resultado_ranking_celular, relatorio_detalhado_email)
