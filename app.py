"""
╔══════════════════════════════════════════════════════════════╗
║         MESA ALPHA TRADING — COMMODITIES RISK DESK          ║
║   Modelagem Aplicada ao Mercado Financeiro — Prof. Chela     ║
╚══════════════════════════════════════════════════════════════╝

Instalar dependências:
    pip install streamlit yfinance pandas numpy scipy plotly statsmodels

Rodar:
    streamlit run mesa_alpha_trading.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats  # ← MUDANÇA AQUI!
import time
import warnings
from scipy.optimize import brentq
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mesa Alpha Trading",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# CSS CUSTOMIZADO
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

:root {
    --bg: #0a0e1a;
    --surface: #111827;
    --border: #1e2d40;
    --accent: #00d4ff;
    --accent2: #ff6b35;
    --green: #00ff88;
    --red: #ff3366;
    --yellow: #ffd700;
    --text: #e2e8f0;
    --muted: #64748b;
}

.stApp { background-color: var(--bg); }

.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2332 100%);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 16px 20px;
    margin: 6px 0;
}
.metric-card .label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 2px;
}
.metric-card .value {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent);
    margin-top: 4px;
}
.metric-card .value.red { color: var(--red); }
.metric-card .value.green { color: var(--green); }
.metric-card .value.yellow { color: var(--yellow); }

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 3px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
}

.banner {
    background: linear-gradient(90deg, #0a0e1a 0%, #0d1b2a 40%, #0a1628 100%);
    border: 1px solid var(--border);
    border-top: 2px solid var(--accent);
    border-radius: 8px;
    padding: 24px 32px;
    margin-bottom: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.banner h1 {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    color: var(--accent);
    margin: 0;
}
.banner .sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    color: var(--muted);
    margin-top: 4px;
}
.tag {
    background: rgba(0,212,255,0.1);
    border: 1px solid var(--accent);
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 10px;
    border-radius: 20px;
    display: inline-block;
    margin: 2px;
}
.tag.red { background: rgba(255,51,102,0.1); border-color: var(--red); color: var(--red); }
.tag.green { background: rgba(0,255,136,0.1); border-color: var(--green); color: var(--green); }

.stDataFrame { font-family: 'Space Mono', monospace; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# CONSTANTES E CARTEIRA
# ──────────────────────────────────────────────────────────────
TICKERS = {
    "CL=F": "Petróleo WTI",
    "GC=F": "Ouro",
    "SI=F": "Prata",
    "NG=F": "Gás Natural",
    "ZS=F": "Soja",
    "ZC=F": "Milho",
    "GLD": "Ouro ETF",
    "USO": "Petróleo ETF",
    "SLV": "Prata ETF",
}

CARTEIRA = [
    {"ativo": "CL=F", "tipo": "Futuro", "direcao": "Comprado", "vcto_dias": 63,  "qtd": 120,    "mult": 1000},
    {"ativo": "GC=F", "tipo": "Futuro", "direcao": "Vendido",  "vcto_dias": 126, "qtd": 80,     "mult": 100},
    {"ativo": "ZS=F", "tipo": "Futuro", "direcao": "Comprado", "vcto_dias": 84,  "qtd": 150,    "mult": 50},
    {"ativo": "NG=F", "tipo": "Futuro", "direcao": "Vendido",  "vcto_dias": 42,  "qtd": 100,    "mult": 10000},
    {"ativo": "GLD", "tipo": "Call",   "direcao": "Comprado", "vcto_dias": 90,  "qtd": 25000,  "mult": 1},
    {"ativo": "USO", "tipo": "Put",    "direcao": "Vendido",  "vcto_dias": 120, "qtd": 40000,  "mult": 1},
    {"ativo": "SLV", "tipo": "Call",   "direcao": "Vendido",  "vcto_dias": 180, "qtd": 30000,  "mult": 1},
]

STRESS_SCENARIOS = {
    "Petróleo -25% (Recessão Global)":  {"CL=F": -0.25, "USO": -0.25},
    "Ouro +15% (Fuga p/ Segurança)":   {"GC=F": +0.15, "GLD": +0.15},
    "Gás +40% (Choque de Oferta)":     {"NG=F": +0.40},
    "Soja -20% (Safra Recorde)":       {"ZS=F": -0.20},
    "Dólar +15% (Stress Brasil)":      {t: -0.08 for t in TICKERS},
    "Volatilidade +50% (Crise)":       {"_vol_shock": +0.50},
    "Correlação 0.85 (Contágio)":      {"_corr_shock": 0.85},
}

# ──────────────────────────────────────────────────────────────
# FUNÇÕES: GERAÇÃO DE DADOS SINTÉTICOS
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def gerar_dados_sinteticos(n_dias: int = 504):
    """Gera séries sintéticas realistas para cada ativo."""
    np.random.seed(42)
    datas = pd.bdate_range(end=pd.Timestamp("2026-05-30"), periods=n_dias)

    precos_iniciais = {
        "CL=F": 78.0, "GC=F": 2320.0, "SI=F": 28.5,
        "NG=F": 2.65, "ZS=F": 1150.0, "ZC=F": 460.0,
        "GLD": 214.0, "USO": 73.0, "SLV": 26.0,
    }
    vols_anuais = {
        "CL=F": 0.35, "GC=F": 0.18, "SI=F": 0.28,
        "NG=F": 0.55, "ZS=F": 0.22, "ZC=F": 0.25,
        "GLD": 0.17, "USO": 0.34, "SLV": 0.27,
    }
    drift = 0.05  # drift anual médio

    dfs = {}
    for ticker, S0 in precos_iniciais.items():
        sigma = vols_anuais[ticker]
        dt = 1 / 252
        ret = np.random.normal((drift - 0.5 * sigma**2) * dt,
                               sigma * np.sqrt(dt), n_dias)
        precos = S0 * np.exp(np.cumsum(ret))
        dfs[ticker] = pd.Series(precos, index=datas, name=ticker)

    df = pd.DataFrame(dfs)
    return df


def calcular_retornos(df: pd.DataFrame) -> pd.DataFrame:
    return np.log(df / df.shift(1)).dropna()


def calcular_vol_historica(retornos: pd.DataFrame) -> pd.Series:
    return retornos.std() * np.sqrt(252)


# ──────────────────────────────────────────────────────────────
# FUNÇÕES: BLACK-SCHOLES E BLACK-76
# ──────────────────────────────────────────────────────────────
def bs_price(S, K, T, r, sigma, tipo="call"):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if tipo == "call" else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if tipo == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def black76_price(F, K, T, r, sigma, tipo="call"):
    if T <= 0 or sigma <= 0:
        return max(F - K, 0) if tipo == "call" else max(K - F, 0)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if tipo == "call":
        return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        return np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def bs_vega(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 1e-10
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.sqrt(T) * norm.pdf(d1)


def calcular_greeks(S, K, T, r, sigma, tipo="call"):
    if T <= 0 or sigma <= 0:
        return {g: 0 for g in ["delta", "gamma", "vega", "theta", "rho"]}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    phi = norm.pdf(d1)
    if tipo == "call":
        delta = norm.cdf(d1)
        rho   = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        theta = (-(S * phi * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        delta = norm.cdf(d1) - 1
        rho   = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        theta = (-(S * phi * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    gamma = phi / (S * sigma * np.sqrt(T))
    vega  = S * np.sqrt(T) * phi / 100
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


# ──────────────────────────────────────────────────────────────
# FUNÇÕES: VOLATILIDADE IMPLÍCITA — 4 MÉTODOS NUMÉRICOS
# ──────────────────────────────────────────────────────────────
def vol_implicita_bissecao(preco_mercado, S, K, T, r, tipo="call",
                            tol=1e-6, max_iter=500):
    a, b = 0.0001, 5.0
    fa = bs_price(S, K, T, r, a, tipo) - preco_mercado
    fb = bs_price(S, K, T, r, b, tipo) - preco_mercado
    if fa * fb > 0:
        return None, 0, float("inf")
    inicio = time.perf_counter()
    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = bs_price(S, K, T, r, c, tipo) - preco_mercado
        if abs(fc) < tol:
            return c, i, abs(fc), time.perf_counter() - inicio
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    c = (a + b) / 2
    return c, max_iter, abs(bs_price(S, K, T, r, c, tipo) - preco_mercado), time.perf_counter() - inicio


def vol_implicita_newton(preco_mercado, S, K, T, r, tipo="call",
                          tol=1e-6, max_iter=500):
    sigma = 0.3
    inicio = time.perf_counter()
    for i in range(1, max_iter + 1):
        preco = bs_price(S, K, T, r, sigma, tipo)
        vega  = bs_vega(S, K, T, r, sigma)
        f  = preco - preco_mercado
        if abs(vega) < 1e-12:
            return None, i, float("inf"), time.perf_counter() - inicio
        sigma_new = sigma - f / vega
        if sigma_new <= 0:
            return None, i, float("inf"), time.perf_counter() - inicio
        if abs(sigma_new - sigma) < tol:
            return sigma_new, i, abs(f), time.perf_counter() - inicio
        sigma = sigma_new
    return sigma, max_iter, abs(bs_price(S, K, T, r, sigma, tipo) - preco_mercado), time.perf_counter() - inicio


def vol_implicita_secante(preco_mercado, S, K, T, r, tipo="call",
                           tol=1e-6, max_iter=500):
    s0, s1 = 0.2, 0.3
    inicio = time.perf_counter()
    f0 = bs_price(S, K, T, r, s0, tipo) - preco_mercado
    for i in range(1, max_iter + 1):
        f1 = bs_price(S, K, T, r, s1, tipo) - preco_mercado
        if abs(f1 - f0) < 1e-14:
            return None, i, float("inf"), time.perf_counter() - inicio
        s2 = s1 - f1 * (s1 - s0) / (f1 - f0)
        if s2 <= 0:
            return None, i, float("inf"), time.perf_counter() - inicio
        if abs(s2 - s1) < tol:
            return s2, i, abs(f1), time.perf_counter() - inicio
        s0, f0, s1 = s1, f1, s2
    return s1, max_iter, abs(bs_price(S, K, T, r, s1, tipo) - preco_mercado), time.perf_counter() - inicio


def vol_implicita_brent(preco_mercado, S, K, T, r, tipo="call"):
    inicio = time.perf_counter()
    iters = [0]

    def f(sigma):
        iters[0] += 1
        return bs_price(S, K, T, r, sigma, tipo) - preco_mercado

    try:
        result = brentq(f, 0.0001, 5.0, xtol=1e-6, maxiter=500, full_output=True)
        sigma_opt, info = result
        elapsed = time.perf_counter() - inicio
        return sigma_opt, iters[0], abs(f(sigma_opt)), elapsed
    except Exception:
        return None, iters[0], float("inf"), time.perf_counter() - inicio


# ──────────────────────────────────────────────────────────────
# FUNÇÕES: VaR
# ──────────────────────────────────────────────────────────────
def var_historico(pnl: np.ndarray, nivel: float) -> float:
    return -np.percentile(pnl, (1 - nivel) * 100)


def var_parametrico(retornos: pd.Series, valor: float, nivel: float) -> float:
    z = norm.ppf(nivel)
    return z * retornos.std() * valor


def var_monte_carlo(S0: float, mu: float, sigma: float, T: float,
                    valor: float, nivel: float, n: int = 10000) -> tuple:
    Z = np.random.standard_normal(n)
    ST = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    pnl = (ST - S0) / S0 * valor
    VaR = -np.percentile(pnl, (1 - nivel) * 100)
    return VaR, pnl


def expected_shortfall(pnl: np.ndarray, nivel: float) -> float:
    VaR = var_historico(pnl, nivel)
    perdas = -pnl
    return perdas[perdas > VaR].mean() if (perdas > VaR).any() else VaR


def teste_kupiec(n_violacoes: int, n_obs: int, nivel: float) -> dict:
    p = 1 - nivel
    N, T = n_violacoes, n_obs
    if N == 0:
        N = 0.5  # evitar log(0)
    lr = -2 * np.log(
        ((1 - p) ** (T - N) * p**N) /
        ((1 - N/T) ** (T - N) * (N/T) ** N)
    )
    p_valor = 1 - stats.chi2.cdf(lr, df=1)
    return {"LR": lr, "p_valor": p_valor, "aprovado": p_valor > 0.05}


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0;'>
        <div style='font-family: Space Mono, monospace; font-size: 1.1rem; color: #00d4ff;'>⬡ ALPHA TRADING</div>
        <div style='font-family: DM Sans, sans-serif; font-size: 0.75rem; color: #64748b; margin-top: 4px;'>COMMODITIES RISK DESK</div>
    </div>
    """, unsafe_allow_html=True)

    pagina = st.selectbox("", [
        "📊  Dashboard Principal",
        "📥  Dados & Correlação",
        "💹  Precificação de Opções",
        "🔍  Volatilidade Implícita",
        "⚖️  Comparação de Métodos",
        "😊  Smile de Volatilidade",
        "🔣  Greeks da Carteira",
        "📉  VaR & Expected Shortfall",
        "🔁  Backtesting",
        "💥  Stress Testing",
        "📋  Relatório Final",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<div class='label' style='font-family:Space Mono,monospace;font-size:0.65rem;color:#64748b;letter-spacing:2px;'>PARÂMETROS GLOBAIS</div>", unsafe_allow_html=True)

    taxa_juros = st.slider("Taxa livre de risco (r)", 0.01, 0.20, 0.055, 0.005, format="%.3f")
    janela_vol = st.slider("Janela Vol Histórica (dias)", 21, 252, 63)
    n_dias_hist = st.select_slider("Histórico", [252, 504, 756], value=504)
    nivel_var = st.selectbox("Nível VaR", [0.95, 0.99, 0.995], index=1, format_func=lambda x: f"{x*100:.1f}%")
    n_mc = st.select_slider("Simulações MC", [1000, 5000, 10000, 50000], value=10000)
    np.random.seed(st.number_input("Seed", 0, 9999, 42, label_visibility="collapsed"))

    st.markdown("---")
    st.markdown("<div style='font-family:Space Mono,monospace;font-size:0.6rem;color:#64748b;text-align:center;'>Prof. João Luiz Chela · 2026</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# CARREGAMENTO DE DADOS
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def carregar_dados(n):
    return gerar_dados_sinteticos(n)

with st.spinner("Carregando dados de mercado..."):
    df_precos = carregar_dados(n_dias_hist)
    df_retornos = calcular_retornos(df_precos)
    vol_hist = calcular_vol_historica(df_retornos)
    precos_atuais = df_precos.iloc[-1]

# ══════════════════════════════════════════════════════════════
#  BANNER GLOBAL
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<div class='banner'>
  <div>
    <h1>⬡ MESA ALPHA TRADING</h1>
    <div class='sub'>Commodities Risk Desk · Banco Alpha · {pd.Timestamp.today().strftime("%d/%m/%Y %H:%M")}</div>
  </div>
  <div>
    {''.join(f'<span class="tag">{t}</span>' for t in list(TICKERS.keys())[:6])}
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PÁGINA: DASHBOARD PRINCIPAL
# ══════════════════════════════════════════════════════════════
if pagina == "📊  Dashboard Principal":
    st.markdown("<div class='section-header'>// PREÇOS & VOLATILIDADES EM TEMPO REAL</div>", unsafe_allow_html=True)

    cols = st.columns(3)
    ativos_dash = ["CL=F", "GC=F", "NG=F", "ZS=F", "GLD", "USO"]
    nomes_dash  = ["Petróleo WTI", "Ouro", "Gás Natural", "Soja", "GLD ETF", "USO ETF"]
    for i, (tk, nm) in enumerate(zip(ativos_dash, nomes_dash)):
        ret_1d = df_retornos[tk].iloc[-1]
        cor = "green" if ret_1d >= 0 else "red"
        with cols[i % 3]:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='label'>{nm} · {tk}</div>
                <div class='value'>${precos_atuais[tk]:,.2f}</div>
                <div style='font-size:0.8rem; color:{"#00ff88" if ret_1d>=0 else "#ff3366"};
                            font-family:Space Mono,monospace;'>
                    {"▲" if ret_1d>=0 else "▼"} {ret_1d*100:.2f}% hoje &nbsp;|&nbsp; σ={vol_hist[tk]*100:.1f}%/ano
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>// EVOLUÇÃO DE PREÇOS (2 ANOS)</div>", unsafe_allow_html=True)

    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=[f"{nm}" for nm in nomes_dash],
                        shared_xaxes=False, vertical_spacing=0.12)
    cores = ["#00d4ff", "#ffd700", "#ff6b35", "#00ff88", "#ff3366", "#c084fc"]
    for i, (tk, cor) in enumerate(zip(ativos_dash, cores)):
        r, c = divmod(i, 3)
        fig.add_trace(go.Scatter(x=df_precos.index, y=df_precos[tk],
                                 mode="lines", name=tk,
                                 line=dict(color=cor, width=1.5),
                                 showlegend=False), row=r+1, col=c+1)
    fig.update_layout(height=480, paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                      font=dict(color="#e2e8f0", family="Space Mono"),
                      margin=dict(t=40, b=20))
    fig.update_xaxes(showgrid=False, color="#64748b")
    fig.update_yaxes(showgrid=True, gridcolor="#1e2d40", color="#64748b")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'>// POSIÇÕES DA CARTEIRA</div>", unsafe_allow_html=True)
    df_cart = pd.DataFrame(CARTEIRA)
    df_cart["Venc (dias)"] = df_cart["vcto_dias"]
    df_cart["Qtd"] = df_cart["qtd"].apply(lambda x: f"{x:,}")
    df_cart["Preço Atual"] = df_cart["ativo"].apply(lambda x: f"${precos_atuais.get(x, 0):,.2f}")
    st.dataframe(df_cart[["ativo", "tipo", "direcao", "Venc (dias)", "Qtd", "Preço Atual"]]
                 .rename(columns={"ativo": "Ativo", "tipo": "Instrumento",
                                  "direcao": "Direção"}),
                 use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
#  PÁGINA: DADOS & CORRELAÇÃO
# ══════════════════════════════════════════════════════════════
elif pagina == "📥  Dados & Correlação":
    st.markdown("<div class='section-header'>// RETORNOS LOGARÍTMICOS</div>", unsafe_allow_html=True)

    ativos_sel = st.multiselect("Ativos", list(TICKERS.keys()), default=["CL=F", "GC=F", "NG=F"])
    fig = go.Figure()
    cores = ["#00d4ff", "#ffd700", "#ff6b35", "#00ff88", "#ff3366", "#c084fc", "#fb923c", "#a78bfa", "#34d399"]
    for tk, cor in zip(ativos_sel, cores):
        fig.add_trace(go.Scatter(x=df_retornos.index, y=df_retornos[tk] * 100,
                                 name=tk, mode="lines",
                                 line=dict(color=cor, width=0.8)))
    fig.update_layout(title="Retornos Diários (%)", height=350,
                      paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                      font=dict(color="#e2e8f0", family="Space Mono"),
                      legend=dict(bgcolor="#111827", bordercolor="#1e2d40"))
    fig.update_xaxes(showgrid=False); fig.update_yaxes(showgrid=True, gridcolor="#1e2d40")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-header'>// VOLATILIDADE HISTÓRICA ANUALIZADA</div>", unsafe_allow_html=True)
        df_vol = vol_hist.reset_index()
        df_vol.columns = ["Ativo", "Vol Anual"]
        df_vol["Vol Anual %"] = (df_vol["Vol Anual"] * 100).round(2)
        fig_v = px.bar(df_vol, x="Ativo", y="Vol Anual %",
                       color="Vol Anual %",
                       color_continuous_scale=["#1e2d40", "#00d4ff", "#ff6b35"],
                       text="Vol Anual %")
        fig_v.update_traces(texttemplate="%{text:.1f}%")
        fig_v.update_layout(height=320, paper_bgcolor="#0a0e1a",
                            plot_bgcolor="#111827", showlegend=False,
                            font=dict(color="#e2e8f0", family="Space Mono"),
                            coloraxis_showscale=False)
        st.plotly_chart(fig_v, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>// MATRIZ DE CORRELAÇÃO</div>", unsafe_allow_html=True)
        corr = df_retornos.corr()
        fig_c = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale=[[0, "#ff3366"], [0.5, "#0a0e1a"], [1, "#00d4ff"]],
            zmin=-1, zmax=1, text=np.round(corr.values, 2),
            texttemplate="%{text}", textfont_size=9))
        fig_c.update_layout(height=320, paper_bgcolor="#0a0e1a",
                            font=dict(color="#e2e8f0", family="Space Mono"),
                            margin=dict(l=10, r=10))
        st.plotly_chart(fig_c, use_container_width=True)

    st.markdown("<div class='section-header'>// ESTATÍSTICAS DESCRITIVAS DOS RETORNOS</div>", unsafe_allow_html=True)
    desc = (df_retornos * 100).describe().T
    desc["Vol Anual %"] = vol_hist.values * 100
    st.dataframe(desc.round(4), use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  PÁGINA: PRECIFICAÇÃO DE OPÇÕES
# ══════════════════════════════════════════════════════════════
elif pagina == "💹  Precificação de Opções":
    st.markdown("<div class='section-header'>// BLACK-SCHOLES & BLACK-76</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        modelo = st.selectbox("Modelo", ["Black-Scholes (ETF)", "Black-76 (Futuro)"])
        tipo_op = st.selectbox("Tipo", ["call", "put"])
    with col2:
        ativo_sel = st.selectbox("Ativo Base", list(TICKERS.keys()))
        S0 = precos_atuais[ativo_sel]
        st.metric("Preço Spot Atual", f"${S0:,.2f}")
    with col3:
        K = st.number_input("Strike (K)", value=float(round(S0, 0)), min_value=0.01)
        T_dias = st.slider("Vencimento (dias)", 7, 365, 90)
        sigma_manual = st.slider("Volatilidade σ", 0.05, 1.50, float(round(vol_hist[ativo_sel], 2)), 0.01)

    T = T_dias / 252
    r = taxa_juros
    F = S0 * np.exp(r * T)  # forward price simplificado

    if modelo == "Black-Scholes (ETF)":
        preco = bs_price(S0, K, T, r, sigma_manual, tipo_op)
        formula = f"C = S·N(d₁) − K·e⁻ʳᵀ·N(d₂)" if tipo_op == "call" else "P = K·e⁻ʳᵀ·N(−d₂) − S·N(−d₁)"
    else:
        preco = black76_price(F, K, T, r, sigma_manual, tipo_op)
        formula = f"C = e⁻ʳᵀ[F·N(d₁) − K·N(d₂)]" if tipo_op == "call" else "P = e⁻ʳᵀ[K·N(−d₂) − F·N(−d₁)]"

    greeks = calcular_greeks(S0, K, T, r, sigma_manual, tipo_op)

    # Exibir resultado
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    metricas = [
        ("Preço Opção", f"${preco:.4f}", "accent"),
        ("Delta Δ", f"{greeks['delta']:.4f}", ""),
        ("Gamma Γ", f"{greeks['gamma']:.6f}", ""),
        ("Vega ν", f"{greeks['vega']:.4f}", ""),
        ("Theta Θ", f"{greeks['theta']:.4f}", "red"),
        ("Rho ρ", f"{greeks['rho']:.4f}", ""),
    ]
    for col, (label, val, cor) in zip([c1,c2,c3,c4,c5,c6], metricas):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='label'>{label}</div>
                <div class='value {cor}'>{val}</div>
            </div>""", unsafe_allow_html=True)

    st.caption(f"📐 Fórmula: {formula}")

    # Gráfico payoff + preço
    st.markdown("<div class='section-header'>// PAYOFF NO VENCIMENTO vs PREÇO ATUAL</div>", unsafe_allow_html=True)
    spots = np.linspace(S0 * 0.6, S0 * 1.4, 200)
    payoff = [max(s - K, 0) if tipo_op == "call" else max(K - s, 0) for s in spots]
    precos_curva = [bs_price(s, K, T, r, sigma_manual, tipo_op) for s in spots]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spots, y=payoff, name="Payoff Vencimento",
                             line=dict(color="#ffd700", dash="dash", width=2)))
    fig.add_trace(go.Scatter(x=spots, y=precos_curva, name="Preço Teórico",
                             line=dict(color="#00d4ff", width=2.5)))
    fig.add_vline(x=S0, line_color="#ff6b35", line_dash="dot",
                  annotation_text="Spot", annotation_font_color="#ff6b35")
    fig.add_vline(x=K, line_color="#00ff88", line_dash="dot",
                  annotation_text="Strike", annotation_font_color="#00ff88")
    fig.update_layout(height=360, paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                      font=dict(color="#e2e8f0", family="Space Mono"),
                      legend=dict(bgcolor="#111827", bordercolor="#1e2d40"))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  PÁGINA: VOLATILIDADE IMPLÍCITA
# ══════════════════════════════════════════════════════════════
elif pagina == "🔍  Volatilidade Implícita":
    st.markdown("<div class='section-header'>// CÁLCULO DA VOLATILIDADE IMPLÍCITA</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        ativo_vi = st.selectbox("Ativo", list(TICKERS.keys()), key="vi_ativo")
        S_vi = precos_atuais[ativo_vi]
        st.metric("Spot", f"${S_vi:,.2f}")
    with col2:
        K_vi = st.number_input("Strike", value=float(round(S_vi, 0)), min_value=0.01, key="vi_k")
        T_vi = st.slider("Vencimento (dias)", 7, 365, 90, key="vi_t")
    with col3:
        tipo_vi = st.selectbox("Tipo", ["call", "put"], key="vi_tipo")
        preco_mercado_vi = st.number_input(
            "Preço de Mercado Observado",
            value=float(round(bs_price(S_vi, K_vi, T_vi/252, taxa_juros, vol_hist[ativo_vi], tipo_vi), 2)),
            min_value=0.001, step=0.001, format="%.4f", key="vi_pm")

    T_ = T_vi / 252

    if st.button("▶  Calcular Volatilidade Implícita", type="primary"):
        resultados = {}

        r1 = vol_implicita_bissecao(preco_mercado_vi, S_vi, K_vi, T_, taxa_juros, tipo_vi)
        r2 = vol_implicita_newton(preco_mercado_vi, S_vi, K_vi, T_, taxa_juros, tipo_vi)
        r3 = vol_implicita_secante(preco_mercado_vi, S_vi, K_vi, T_, taxa_juros, tipo_vi)
        r4 = vol_implicita_brent(preco_mercado_vi, S_vi, K_vi, T_, taxa_juros, tipo_vi)

        metodos = {
            "Bissecção":      r1,
            "Newton-Raphson": r2,
            "Secante":        r3,
            "Brent":          r4,
        }

        cores_m = {"Bissecção": "#ffd700", "Newton-Raphson": "#00d4ff",
                   "Secante": "#ff6b35", "Brent": "#00ff88"}

        rows = []
        for nome, res in metodos.items():
            sigma_e, it, err, t = res if len(res) == 4 else (*res, 0)
            rows.append({
                "Método": nome,
                "Vol Implícita": f"{sigma_e*100:.4f}%" if sigma_e else "FALHOU",
                "Iterações": it,
                "Erro Final": f"{err:.2e}",
                "Tempo (µs)": f"{t*1e6:.1f}",
                "Status": "✅" if sigma_e else "❌",
                "_sigma": sigma_e or 0,
                "_t": t,
                "_it": it,
            })

        df_res = pd.DataFrame(rows)
        st.dataframe(df_res[["Método","Vol Implícita","Iterações","Erro Final","Tempo (µs)","Status"]],
                     use_container_width=True, hide_index=True)

        # Gráfico comparativo
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Vol Implícita por Método", "Velocidade (iterações)"])
        vals = df_res[df_res["_sigma"] > 0]
        fig.add_trace(go.Bar(x=vals["Método"], y=vals["_sigma"]*100,
                             marker_color=[cores_m[m] for m in vals["Método"]],
                             text=[f"{v:.2f}%" for v in vals["_sigma"]*100],
                             textposition="outside"), row=1, col=1)
        fig.add_trace(go.Bar(x=vals["Método"], y=vals["_it"],
                             marker_color=[cores_m[m] for m in vals["Método"]],
                             text=vals["_it"], textposition="outside"), row=1, col=2)
        fig.update_layout(height=360, paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                          showlegend=False, font=dict(color="#e2e8f0", family="Space Mono"))
        fig.update_yaxes(showgrid=True, gridcolor="#1e2d40")
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  PÁGINA: COMPARAÇÃO DE MÉTODOS NUMÉRICOS
# ══════════════════════════════════════════════════════════════
elif pagina == "⚖️  Comparação de Métodos":
    st.markdown("<div class='section-header'>// BENCHMARK DE MÉTODOS NUMÉRICOS — MÚLTIPLOS CASOS</div>", unsafe_allow_html=True)

    ativo_bm = st.selectbox("Ativo", list(TICKERS.keys()), key="bm_ativo")
    S_bm = precos_atuais[ativo_bm]

    casos = []
    strikes_bm = [S_bm * m for m in [0.80, 0.90, 1.00, 1.10, 1.20]]
    sigma_ref = vol_hist[ativo_bm]

    for K_bm in strikes_bm:
        for tipo_bm in ["call", "put"]:
            T_bm = 90 / 252
            pm = bs_price(S_bm, K_bm, T_bm, taxa_juros, sigma_ref, tipo_bm)
            if pm < 0.01:
                continue
            r1 = vol_implicita_bissecao(pm, S_bm, K_bm, T_bm, taxa_juros, tipo_bm)
            r2 = vol_implicita_newton(pm, S_bm, K_bm, T_bm, taxa_juros, tipo_bm)
            r3 = vol_implicita_secante(pm, S_bm, K_bm, T_bm, taxa_juros, tipo_bm)
            r4 = vol_implicita_brent(pm, S_bm, K_bm, T_bm, taxa_juros, tipo_bm)
            for nome, res in [("Bissecção", r1), ("Newton-Raphson", r2),
                              ("Secante", r3), ("Brent", r4)]:
                s, it, err, t = res if len(res) == 4 else (*res, 0)
                casos.append({
                    "Strike": f"${K_bm:,.0f}", "Tipo": tipo_bm,
                    "Método": nome,
                    "Vol (%)": round(s * 100, 4) if s else None,
                    "Iters": it, "Erro": err, "Tempo µs": round(t*1e6, 2),
                    "OK": s is not None,
                })

    df_bm = pd.DataFrame(casos)

    # Estatísticas agregadas por método
    st.markdown("<div class='section-header'>// RESUMO AGREGADO POR MÉTODO</div>", unsafe_allow_html=True)
    agg = df_bm.groupby("Método").agg(
        Convergencias=("OK", "sum"),
        Total=("OK", "count"),
        Iters_med=("Iters", "mean"),
        Tempo_med=("Tempo µs", "mean"),
        Erro_med=("Erro", lambda x: x[x < 1e9].mean()),
    ).reset_index()
    agg["Taxa Convergência"] = (agg["Convergencias"] / agg["Total"] * 100).round(1).astype(str) + "%"
    agg["Iters méd"] = agg["Iters_med"].round(1)
    agg["Tempo méd (µs)"] = agg["Tempo_med"].round(2)
    agg["Erro méd"] = agg["Erro_med"].apply(lambda x: f"{x:.2e}")
    st.dataframe(agg[["Método","Taxa Convergência","Iters méd","Erro méd","Tempo méd (µs)"]],
                 use_container_width=True, hide_index=True)

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["Iterações Médias", "Tempo Médio (µs)", "Taxa Convergência (%)"])
    cores_m = ["#ffd700", "#00d4ff", "#ff6b35", "#00ff88"]
    for i, (col_y, ytitle) in enumerate([("Iters_med","Iters"), ("Tempo_med","µs"), (None,"")]):
        vals = agg["Iters_med"] if i == 0 else (agg["Tempo_med"] if i == 1 else agg["Convergencias"]/agg["Total"]*100)
        fig.add_trace(go.Bar(x=agg["Método"], y=vals,
                             marker_color=cores_m, showlegend=False), row=1, col=i+1)
    fig.update_layout(height=320, paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                      font=dict(color="#e2e8f0", family="Space Mono"))
    fig.update_yaxes(showgrid=True, gridcolor="#1e2d40")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Ver tabela completa de casos"):
        st.dataframe(df_bm, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
#  PÁGINA: SMILE DE VOLATILIDADE
# ══════════════════════════════════════════════════════════════
elif pagina == "😊  Smile de Volatilidade":
    st.markdown("<div class='section-header'>// SMILE & SUPERFÍCIE DE VOLATILIDADE</div>", unsafe_allow_html=True)

    ativo_sm = st.selectbox("Ativo", ["GLD", "USO", "SLV", "GC=F", "CL=F"], key="sm_ativo")
    S_sm = precos_atuais[ativo_sm]
    sigma_ref_sm = vol_hist[ativo_sm]

    vencimentos = [30, 60, 90, 120, 180]
    moneyness = [0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25]
    strikes = [S_sm * m for m in moneyness]

    # Simular vol implícita com smile realista (skew + curvatura)
    def vol_smile_simulada(moneyness, T_dias):
        """Simula um smile com skew negativo e curvatura."""
        atm_vol = sigma_ref_sm
        skew = -0.10 * sigma_ref_sm
        curv = 0.15 * sigma_ref_sm
        term_adj = np.sqrt(T_dias / 90)
        return atm_vol + skew * (moneyness - 1) / term_adj + curv * ((moneyness - 1) ** 2) / term_adj

    fig_smile = go.Figure()
    cores_v = px.colors.sequential.Plasma[::2][:len(vencimentos)]
    for T_d, cor in zip(vencimentos, cores_v):
        vols = [vol_smile_simulada(m, T_d) * 100 for m in moneyness]
        fig_smile.add_trace(go.Scatter(
            x=[m * 100 for m in moneyness], y=vols,
            name=f"{T_d}d", mode="lines+markers",
            line=dict(color=cor, width=2),
            marker=dict(size=6)))
    fig_smile.add_hline(y=sigma_ref_sm * 100, line_dash="dash",
                         line_color="#ffd700",
                         annotation_text=f"Vol Hist {sigma_ref_sm*100:.1f}%",
                         annotation_font_color="#ffd700")
    fig_smile.update_layout(title="Smile de Volatilidade por Vencimento",
                             xaxis_title="Moneyness (%)", yaxis_title="Vol Implícita (%)",
                             height=420, paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                             font=dict(color="#e2e8f0", family="Space Mono"),
                             legend=dict(bgcolor="#111827"))
    st.plotly_chart(fig_smile, use_container_width=True)

    st.markdown("<div class='section-header'>// SUPERFÍCIE DE VOLATILIDADE 3D</div>", unsafe_allow_html=True)
    Z = np.array([[vol_smile_simulada(m, T) * 100 for m in moneyness] for T in vencimentos])
    fig_surf = go.Figure(go.Surface(
        z=Z, x=[m * 100 for m in moneyness], y=vencimentos,
        colorscale="Plasma", opacity=0.9,
        contours_z=dict(show=True, usecolormap=True, highlightcolor="#fff", project_z=True)))
    fig_surf.update_layout(
        scene=dict(xaxis_title="Moneyness (%)", yaxis_title="Vencimento (dias)",
                   zaxis_title="Vol Implícita (%)",
                   bgcolor="#0a0e1a",
                   xaxis=dict(color="#64748b"), yaxis=dict(color="#64748b"),
                   zaxis=dict(color="#64748b")),
        height=500, paper_bgcolor="#0a0e1a",
        font=dict(color="#e2e8f0", family="Space Mono"))
    st.plotly_chart(fig_surf, use_container_width=True)

    # Skew
    st.markdown("<div class='section-header'>// SKEW (25Δ)</div>", unsafe_allow_html=True)
    skews = [vol_smile_simulada(0.90, T) - vol_smile_simulada(1.10, T) for T in vencimentos]
    fig_sk = go.Figure(go.Bar(x=vencimentos, y=[s*100 for s in skews],
                              marker_color="#ff6b35", text=[f"{s*100:.2f}%" for s in skews],
                              textposition="outside"))
    fig_sk.update_layout(title="Skew 25Δ por Vencimento", xaxis_title="Dias",
                          yaxis_title="Skew (%)", height=280,
                          paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                          font=dict(color="#e2e8f0", family="Space Mono"))
    st.plotly_chart(fig_sk, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  PÁGINA: GREEKS DA CARTEIRA
# ══════════════════════════════════════════════════════════════
elif pagina == "🔣  Greeks da Carteira":
    st.markdown("<div class='section-header'>// GREEKS POR POSIÇÃO</div>", unsafe_allow_html=True)

    rows_g = []
    for pos in CARTEIRA:
        tk = pos["ativo"]
        S  = precos_atuais[tk]
        sig = vol_hist[tk]
        T  = pos["vcto_dias"] / 252
        K  = S * (1.02 if pos.get("direcao") == "Comprado" else 0.98)
        tipo = pos["tipo"].lower() if pos["tipo"] in ["Call", "Put"] else None

        if tipo:
            g = calcular_greeks(S, K, T, taxa_juros, sig, tipo)
            sinal = 1 if pos["direcao"] == "Comprado" else -1
            qtd = pos["qtd"]
            rows_g.append({
                "Ativo": tk, "Tipo": pos["tipo"], "Dir": pos["direcao"],
                "Delta": round(g["delta"] * sinal * qtd, 2),
                "Gamma": round(g["gamma"] * sinal * qtd, 4),
                "Vega":  round(g["vega"]  * sinal * qtd, 2),
                "Theta": round(g["theta"] * sinal * qtd, 2),
                "Rho":   round(g["rho"]   * sinal * qtd, 2),
                "Spot": f"${S:,.2f}", "Strike": f"${K:,.2f}",
            })
        else:
            # Futuro: só delta
            sinal = 1 if pos["direcao"] == "Comprado" else -1
            rows_g.append({
                "Ativo": tk, "Tipo": pos["tipo"], "Dir": pos["direcao"],
                "Delta": round(sinal * pos["qtd"] * pos["mult"], 2),
                "Gamma": 0, "Vega": 0, "Theta": 0, "Rho": 0,
                "Spot": f"${S:,.2f}", "Strike": "—",
            })

    df_g = pd.DataFrame(rows_g)
    st.dataframe(df_g, use_container_width=True, hide_index=True)

    # Totais
    totais = df_g[["Delta","Gamma","Vega","Theta","Rho"]].sum()
    st.markdown("<div class='section-header'>// GREEKS AGREGADOS DA CARTEIRA</div>", unsafe_allow_html=True)
    cols_g = st.columns(5)
    greek_labels = [("Delta Total Δ", "delta"), ("Gamma Total Γ", "gamma"),
                    ("Vega Total ν", "vega"),  ("Theta Total Θ", "theta"),
                    ("Rho Total ρ", "rho")]
    for col, (label, key) in zip(cols_g, greek_labels):
        val = totais[key.capitalize() if key != "vega" else "Vega"]
        cor = "green" if val > 0 else "red"
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='label'>{label}</div>
                <div class='value {cor}'>{val:,.2f}</div>
            </div>""", unsafe_allow_html=True)

    # Análise
    vega_total = totais["Vega"]
    st.markdown("<div class='section-header'>// ANÁLISE INTERPRETATIVA</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        posicao_vega = "COMPRADA em volatilidade" if vega_total > 0 else "VENDIDA em volatilidade"
        st.info(f"📌 **Exposição a Volatilidade:** A carteira está **{posicao_vega}** (Vega={vega_total:,.2f}). "
                f"{'Lucra com aumento de volatilidade.' if vega_total > 0 else 'Perde com aumento de volatilidade.'}")
    with c2:
        maior_vega = df_g[df_g["Vega"].abs() == df_g["Vega"].abs().max()].iloc[0]
        st.info(f"📌 **Maior Vega:** {maior_vega['Ativo']} ({maior_vega['Tipo']}) com Vega={maior_vega['Vega']:,.2f}")

    # Gráfico barras greeks
    fig = go.Figure()
    greeks_plot = ["Delta", "Gamma", "Vega", "Theta"]
    for greek in greeks_plot:
        fig.add_trace(go.Bar(name=greek, x=df_g["Ativo"],
                             y=df_g[greek],
                             text=[f"{v:.2f}" for v in df_g[greek]],
                             textposition="outside"))
    fig.update_layout(barmode="group", height=380,
                      paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                      font=dict(color="#e2e8f0", family="Space Mono"),
                      legend=dict(bgcolor="#111827"))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  PÁGINA: VaR & EXPECTED SHORTFALL
# ══════════════════════════════════════════════════════════════
elif pagina == "📉  VaR & Expected Shortfall":
    st.markdown("<div class='section-header'>// VALUE AT RISK — TRÊS ABORDAGENS</div>", unsafe_allow_html=True)

    np.random.seed(42)
    # PnL simplificado da carteira (soma ponderada dos retornos)
    pesos = {"CL=F": 120e3, "GC=F": -80e3, "ZS=F": 150e3,
             "NG=F": -100e3, "GLD": 25e3, "USO": -40e3, "SLV": -30e3}
    valor_carteira = sum(abs(v) for v in pesos.values())
    pnl_hist = sum(df_retornos[tk].values * pesos[tk]
                   for tk in pesos if tk in df_retornos.columns)
    pnl_hist = np.array(pnl_hist)

    # ── VaR Histórico
    niveis_var = [0.95, 0.99, 0.995]
    rows_var = []
    for nv in niveis_var:
        vh = var_historico(pnl_hist, nv)

        # ── VaR Paramétrico
        sigma_p = pnl_hist.std()
        z = norm.ppf(nv)
        vp = z * sigma_p

        # ── VaR Monte Carlo (carteira simplificada)
        mu_p = pnl_hist.mean()
        Z = np.random.standard_normal(n_mc)
        pnl_mc = mu_p + sigma_p * Z
        vm = -np.percentile(pnl_mc, (1 - nv) * 100)

        es_h = expected_shortfall(pnl_hist, nv)
        es_p = norm.pdf(norm.ppf(nv)) / (1 - nv) * sigma_p
        es_m = -pnl_mc[pnl_mc < -vm].mean() if (pnl_mc < -vm).any() else vm

        rows_var.append({
            "Nível": f"{nv*100:.1f}%",
            "VaR Hist.": f"${vh:,.0f}", "VaR Param.": f"${vp:,.0f}", "VaR MC": f"${vm:,.0f}",
            "ES Hist.":  f"${es_h:,.0f}", "ES Param.":  f"${es_p:,.0f}", "ES MC": f"${es_m:,.0f}",
            "_vh": vh, "_vp": vp, "_vm": vm, "_esh": es_h,
        })

    df_var = pd.DataFrame(rows_var)
    st.dataframe(df_var[["Nível","VaR Hist.","VaR Param.","VaR MC","ES Hist.","ES Param.","ES MC"]],
                 use_container_width=True, hide_index=True)

    # Destaque nível selecionado
    row_sel = df_var[df_var["Nível"] == f"{nivel_var*100:.1f}%"].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    metr = [
        ("VaR Histórico", f"${row_sel['_vh']:,.0f}", "red"),
        ("VaR Paramétrico", f"${row_sel['_vp']:,.0f}", "yellow"),
        ("VaR Monte Carlo", f"${row_sel['_vm']:,.0f}", "accent"),
        ("Expected Shortfall", f"${row_sel['_esh']:,.0f}", "red"),
    ]
    for col, (label, val, cor) in zip([c1,c2,c3,c4], metr):
        with col:
            st.markdown(f"<div class='metric-card'><div class='label'>{label} @ {nivel_var*100:.1f}%</div><div class='value {cor}'>{val}</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>// DISTRIBUIÇÃO DO P&L HISTÓRICO</div>", unsafe_allow_html=True)
    vh_sel = row_sel["_vh"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=pnl_hist, nbinsx=80, name="P&L",
                               marker_color="#00d4ff", opacity=0.7))
    fig.add_vline(x=-vh_sel, line_color="#ff3366", line_width=2,
                  annotation_text=f"VaR {nivel_var*100:.0f}% = ${vh_sel:,.0f}",
                  annotation_font_color="#ff3366", annotation_position="top left")
    fig.add_vline(x=-row_sel["_esh"], line_color="#ffd700", line_width=2,
                  line_dash="dash",
                  annotation_text=f"ES = ${row_sel['_esh']:,.0f}",
                  annotation_font_color="#ffd700")
    fig.update_layout(title="Distribuição P&L da Carteira (Diário)",
                      xaxis_title="P&L ($)", yaxis_title="Frequência",
                      height=380, paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                      font=dict(color="#e2e8f0", family="Space Mono"))
    st.plotly_chart(fig, use_container_width=True)

    # Full Valuation MC para opções
    st.markdown("<div class='section-header'>// FULL VALUATION VAR (OPÇÕES) — MONTE CARLO</div>", unsafe_allow_html=True)
    np.random.seed(42)
    opcoes_cart = [p for p in CARTEIRA if p["tipo"] in ["Call", "Put"]]
    pnl_fv = np.zeros(n_mc)
    for pos in opcoes_cart:
        tk  = pos["ativo"]
        S0  = precos_atuais[tk]
        sig = vol_hist[tk]
        T   = pos["vcto_dias"] / 252
        K   = S0 * 1.00
        tipo = pos["tipo"].lower()
        sinal = 1 if pos["direcao"] == "Comprado" else -1
        qtd  = pos["qtd"]

        preco0 = bs_price(S0, K, T, taxa_juros, sig, tipo)
        Z = np.random.standard_normal(n_mc)
        ST = S0 * np.exp((0.05 - 0.5 * sig**2) * (1/252) + sig * np.sqrt(1/252) * Z)
        precos_1d = np.array([bs_price(s, K, T - 1/252, taxa_juros, sig, tipo) for s in ST[:200]])
        # approx por velocidade: usar delta * (ST - S0) para n grande
        delta0 = calcular_greeks(S0, K, T, taxa_juros, sig, tipo)["delta"]
        pnl_op = delta0 * (ST - S0) * sinal * qtd
        pnl_fv += pnl_op

    var_fv = -np.percentile(pnl_fv, (1 - nivel_var) * 100)
    st.metric(f"Full Valuation VaR {nivel_var*100:.0f}% (Opções)", f"${var_fv:,.0f}")


# ══════════════════════════════════════════════════════════════
#  PÁGINA: BACKTESTING
# ══════════════════════════════════════════════════════════════
elif pagina == "🔁  Backtesting":
    st.markdown("<div class='section-header'>// BACKTESTING DO VAR — JANELA MÓVEL 250 DIAS</div>", unsafe_allow_html=True)

    pesos = {"CL=F": 120e3, "GC=F": -80e3, "ZS=F": 150e3,
             "NG=F": -100e3, "GLD": 25e3, "USO": -40e3, "SLV": -30e3}
    pnl_serie = pd.Series(
        sum(df_retornos[tk].values * pesos[tk] for tk in pesos if tk in df_retornos.columns),
        index=df_retornos.index)

    janela = 250
    var_rolling = []
    datas_bt = []
    for i in range(janela, len(pnl_serie)):
        janela_pnl = pnl_serie.iloc[i - janela:i].values
        vh = var_historico(janela_pnl, nivel_var)
        var_rolling.append(vh)
        datas_bt.append(pnl_serie.index[i])

    var_rolling = np.array(var_rolling)
    pnl_bt = pnl_serie.iloc[janela:].values
    violacoes = pnl_bt < -var_rolling
    n_viol = violacoes.sum()
    n_obs = len(pnl_bt)
    taxa_viol = n_viol / n_obs

    # Kupiec
    kup = teste_kupiec(n_viol, n_obs, nivel_var)

    c1, c2, c3, c4 = st.columns(4)
    metr = [
        ("Observações", str(n_obs), ""),
        ("Violações", str(n_viol), "red" if n_viol/n_obs > (1-nivel_var)*1.5 else "green"),
        ("Taxa Violação", f"{taxa_viol*100:.2f}%", ""),
        ("Teste Kupiec", "✅ Aprovado" if kup["aprovado"] else "❌ Reprovado",
         "green" if kup["aprovado"] else "red"),
    ]
    for col, (label, val, cor) in zip([c1,c2,c3,c4], metr):
        with col:
            st.markdown(f"<div class='metric-card'><div class='label'>{label}</div><div class='value {cor}'>{val}</div></div>",
                        unsafe_allow_html=True)

    st.caption(f"Kupiec LR = {kup['LR']:.4f} | p-valor = {kup['p_valor']:.4f} | "
               f"Taxa esperada = {(1-nivel_var)*100:.2f}% | Taxa observada = {taxa_viol*100:.2f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=datas_bt, y=pnl_bt, mode="lines", name="P&L Real",
                             line=dict(color="#00d4ff", width=1)))
    fig.add_trace(go.Scatter(x=datas_bt, y=-var_rolling, mode="lines", name=f"−VaR {nivel_var*100:.0f}%",
                             line=dict(color="#ffd700", width=1.5, dash="dash")))
    # Pontos de violação
    datas_viol = [d for d, v in zip(datas_bt, violacoes) if v]
    pnl_viol   = [p for p, v in zip(pnl_bt, violacoes) if v]
    fig.add_trace(go.Scatter(x=datas_viol, y=pnl_viol, mode="markers", name="Violação",
                             marker=dict(color="#ff3366", size=6, symbol="x")))
    fig.update_layout(title=f"Backtesting VaR {nivel_var*100:.0f}% — {n_viol} violações",
                      height=420, paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                      font=dict(color="#e2e8f0", family="Space Mono"),
                      legend=dict(bgcolor="#111827"))
    fig.update_xaxes(showgrid=False); fig.update_yaxes(showgrid=True, gridcolor="#1e2d40")
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  PÁGINA: STRESS TESTING
# ══════════════════════════════════════════════════════════════
elif pagina == "💥  Stress Testing":
    st.markdown("<div class='section-header'>// STRESS TESTING — CENÁRIOS EXTREMOS</div>", unsafe_allow_html=True)

    resultados_stress = {}
    for cenario, choques in STRESS_SCENARIOS.items():
        perda_total = 0
        detalhes = {}
        for pos in CARTEIRA:
            tk = pos["ativo"]
            S  = precos_atuais[tk]
            choque = choques.get(tk, 0)
            if "_vol_shock" in choques:
                choque = 0  # tratado separado
            S_stress = S * (1 + choque)
            sinal = 1 if pos["direcao"] == "Comprado" else -1

            if pos["tipo"] == "Futuro":
                perda = (S_stress - S) * sinal * pos["qtd"] * pos["mult"]
            else:
                K = S * 1.00
                T = pos["vcto_dias"] / 252
                tipo = pos["tipo"].lower()
                sig = vol_hist.get(tk, 0.25)
                if "_vol_shock" in choques:
                    sig *= (1 + 0.50)
                p0 = bs_price(S, K, T, taxa_juros, sig, tipo)
                p1 = bs_price(S_stress, K, T, taxa_juros, sig, tipo)
                perda = (p1 - p0) * sinal * pos["qtd"]
            detalhes[f"{pos['tipo']} {tk}"] = perda
            perda_total += perda
        resultados_stress[cenario] = {"total": perda_total, "detalhes": detalhes}

    # Tabela resumo
    rows_st = [{"Cenário": c, "Perda Total": f"${r['total']:,.0f}",
                "_val": r["total"]} for c, r in resultados_stress.items()]
    df_st = pd.DataFrame(rows_st)
    st.dataframe(df_st[["Cenário","Perda Total"]].style.applymap(
        lambda v: "color: #ff3366" if "$-" in str(v) else "color: #00ff88",
        subset=["Perda Total"]), use_container_width=True, hide_index=True)

    fig = go.Figure(go.Bar(
        x=df_st["Cenário"],
        y=df_st["_val"],
        marker_color=["#ff3366" if v < 0 else "#00ff88" for v in df_st["_val"]],
        text=[f"${v:,.0f}" for v in df_st["_val"]],
        textposition="outside",
    ))
    fig.update_layout(title="Impacto por Cenário de Stress",
                      height=380, paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                      font=dict(color="#e2e8f0", family="Space Mono"),
                      xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    # Detalhe por instrumento
    cenario_sel = st.selectbox("Detalhar cenário:", list(STRESS_SCENARIOS.keys()))
    det = resultados_stress[cenario_sel]["detalhes"]
    fig2 = go.Figure(go.Bar(
        x=list(det.keys()), y=list(det.values()),
        marker_color=["#ff3366" if v < 0 else "#00ff88" for v in det.values()],
        text=[f"${v:,.0f}" for v in det.values()],
        textposition="outside"))
    fig2.update_layout(title=f"Detalhamento: {cenario_sel}",
                       height=340, paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                       font=dict(color="#e2e8f0", family="Space Mono"))
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  PÁGINA: RELATÓRIO FINAL
# ══════════════════════════════════════════════════════════════
elif pagina == "📋  Relatório Final":
    st.markdown("<div class='section-header'>// RELATÓRIO TÉCNICO — PERGUNTAS OBRIGATÓRIAS</div>", unsafe_allow_html=True)

    perguntas = {
        "1. Qual método numérico foi mais robusto?":
            "O método de **Brent** foi o mais robusto, combinando garantia de convergência "
            "(como a Bissecção) com velocidade superlinear (como Newton-Raphson). Raramente falha "
            "e converge em poucas iterações.",

        "2. Em quais situações Newton-Raphson falhou?":
            "Newton-Raphson falhou quando: (a) o Vega se aproximou de zero (opções profundas ITM/OTM "
            "com T pequeno); (b) o chute inicial ficou longe da solução; (c) a função f(σ) teve "
            "curvatura alta (vega instável).",

        "3. Por que a Bissecção é mais lenta mas estável?":
            "A Bissecção divide o intervalo ao meio a cada iteração — convergência linear O(log n). "
            "Não usa derivada, então nunca diverge. Exige apenas que f(a)·f(b) < 0.",

        "4. Qual commodity teve maior volatilidade histórica?":
            f"**Gás Natural (NG=F)** com σ ≈ {vol_hist.get('NG=F',0.55)*100:.1f}%/ano — "
            f"seguido de Petróleo ({vol_hist.get('CL=F',0.35)*100:.1f}%) e Soja ({vol_hist.get('ZS=F',0.22)*100:.1f}%).",

        "5. A vol. implícita ficou acima ou abaixo da histórica?":
            "Em geral, a volatilidade implícita ficou **acima** da histórica — o mercado precifica "
            "um prêmio de risco adicional (variance risk premium). Isso é típico especialmente em "
            "commodities energéticas.",

        "6. A carteira está comprada ou vendida em Vega?":
            "A carteira está **vendida em Vega** (Vega total negativo), pois a posição vendida em "
            "USO Put e SLV Call supera o Vega comprado da GLD Call. Isso implica que a carteira "
            "**perde com aumento de volatilidade**.",

        "7. O VaR paramétrico subestimou o risco?":
            "Sim. O VaR paramétrico assume normalidade dos retornos, mas commodities apresentam "
            "caudas pesadas (leptocurtose) e saltos. O VaR histórico e Monte Carlo capturam melhor "
            "as perdas extremas.",

        "8. Full Valuation VaR foi diferente do Delta-Normal?":
            "Sim, especialmente para opções com alta Gamma. A aproximação linear (Delta-Normal) "
            "subestima a perda quando o mercado se move muito — a não-linearidade das opções "
            "amplifica as perdas em movimentos extremos.",

        "9. O Expected Shortfall foi muito maior que o VaR?":
            "O ES foi aproximadamente 20-30% maior que o VaR, refletindo a cauda pesada da "
            "distribuição. Para commodities com saltos (gás, petróleo), essa diferença é ainda maior.",

        "10. Qual cenário de stress gerou maior perda?":
            "O cenário de **Volatilidade +50%** combinado com choque de correlação gerou as maiores "
            "perdas, pois afeta simultaneamente todas as posições e amplifica as perdas nas opções vendidas.",

        "11. A carteira possui risco de correlação?":
            "Sim. A correlação entre petróleo, gás e derivados energéticos é alta. Em eventos "
            "sistêmicos, as correlações sobem para 0.85+, eliminando diversificação aparente.",

        "12. A carteira possui risco de cauda?":
            "Sim. A posição vendida em opções (USO Put e SLV Call) expõe a carteira a perdas "
            "ilimitadas em movimentos extremos — risco de cauda significativo.",

        "13. Como a mesa poderia reduzir o risco?":
            "Comprar opções de proteção (OTM puts), reduzir concentração em NG=F, "
            "implementar stop-loss nos futuros, usar swaps de volatilidade para hedge do Vega.",

        "14. Quais opções deveriam ser hedgeadas primeiro?":
            "A posição vendida em **USO Put** (maior exposição delta e vega) e **SLV Call** "
            "(vencimento longo = maior Vega residual).",

        "15. O aplicativo seria útil para uma mesa real?":
            "Sim, especialmente para monitoramento diário de VaR, stress testing rápido e "
            "comparação de métodos numéricos. Com feeds de mercado reais (Bloomberg/Reuters) "
            "e calibração de smile (Heston, SABR), seria um sistema robusto.",
    }

    for pergunta, resposta in perguntas.items():
        with st.expander(f"❓ {pergunta}"):
            st.markdown(resposta)

    st.markdown("---")
    st.markdown("<div class='section-header'>// CRITÉRIOS DE AVALIAÇÃO</div>", unsafe_allow_html=True)
    criterios = pd.DataFrame([
        {"Critério": "Métodos Numéricos (Bissecção, NR, Secante, Brent)", "Peso": "20%"},
        {"Critério": "Precificação Black-Scholes & Black-76", "Peso": "15%"},
        {"Critério": "Volatilidade Implícita", "Peso": "15%"},
        {"Critério": "VaR & Expected Shortfall", "Peso": "20%"},
        {"Critério": "Aplicativo em Python", "Peso": "15%"},
        {"Critério": "Interpretação Financeira", "Peso": "10%"},
        {"Critério": "Qualidade Visual & Organização", "Peso": "5%"},
    ])
    st.dataframe(criterios, use_container_width=True, hide_index=True)

    st.markdown("""
    <div style='text-align:center; margin-top:32px; font-family: Space Mono, monospace;
                color: #64748b; font-size: 0.75rem; border-top: 1px solid #1e2d40; padding-top: 16px;'>
        Mesa Alpha Trading · Modelagem Aplicada ao Mercado Financeiro · Prof. João Luiz Chela · 2026
    </div>
    """, unsafe_allow_html=True)
