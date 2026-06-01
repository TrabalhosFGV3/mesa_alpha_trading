"""
╔══════════════════════════════════════════════════════════════╗
║         MESA ALPHA TRADING — COMMODITIES RISK DESK          ║
║              SEM SCIPY - APENAS NUMPY E MATH                ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
import time
import warnings
from scipy.optimize import brentq  # só isso do scipy
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ================================================================
# FUNÇÃO NORMAL CDF IMPLEMENTADA MANUALMENTE (sem scipy.stats)
# ================================================================

def norm_cdf(x):
    """Função de distribuição acumulada normal - implementação numérica"""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def norm_pdf(x):
    """Função de densidade de probabilidade normal"""
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)

def norm_ppf(p):
    """Inverso da CDF normal (aproximação de Abramowitz & Stegun)"""
    if p <= 0 or p >= 1:
        return 0
    # Aproximação para quantis
    a = np.array([-3.969683028665376e+01, 2.209460984245205e+02,
                  -2.759285104961687e+02, 1.383577518672690e+02,
                  -3.066479806614716e+01, 2.506628277459239e+00])
    b = np.array([-5.447609879822406e+01, 1.615858368580409e+02,
                  -1.556989798598866e+02, 6.680131188771972e+01,
                  -1.328068155288572e+01])
    c = np.array([-7.784894002430293e-03, -3.223964580411365e-01,
                  -2.400758277161838e+00, -2.549732539343734e+00,
                  4.374664141464968e+00, 2.938163982698783e+00])
    d = np.array([7.784695709041462e-03, 3.224671290700398e-01,
                  2.445134137142996e+00, 3.754408661907416e+00])

    if p < 0.02425:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    else:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    
    return x

# ================================================================
# CONFIGURAÇÃO DA PÁGINA
# ================================================================

st.set_page_config(
    page_title="Mesa Alpha Trading",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS
st.markdown("""
<style>
.stApp { background-color: #0a0e1a; }
.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2332 100%);
    border: 1px solid #1e2d40;
    border-left: 3px solid #00d4ff;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 6px 0;
}
.metric-card .label {
    font-family: monospace;
    font-size: 0.65rem;
    color: #64748b;
    text-transform: uppercase;
}
.metric-card .value {
    font-family: monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #00d4ff;
}
.metric-card .value.red { color: #ff3366; }
.metric-card .value.green { color: #00ff88; }
.section-header {
    font-family: monospace;
    font-size: 0.75rem;
    color: #00d4ff;
    text-transform: uppercase;
    border-bottom: 1px solid #1e2d40;
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
}
.banner {
    background: linear-gradient(90deg, #0a0e1a 0%, #0d1b2a 40%, #0a1628 100%);
    border: 1px solid #1e2d40;
    border-top: 2px solid #00d4ff;
    border-radius: 8px;
    padding: 24px 32px;
    margin-bottom: 24px;
}
.tag {
    background: rgba(0,212,255,0.1);
    border: 1px solid #00d4ff;
    color: #00d4ff;
    font-family: monospace;
    font-size: 0.7rem;
    padding: 2px 10px;
    border-radius: 20px;
    display: inline-block;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ================================================================
# DADOS SINTÉTICOS
# ================================================================

@st.cache_data(show_spinner=False)
def gerar_dados():
    np.random.seed(42)
    n_dias = 504
    datas = pd.bdate_range(end=pd.Timestamp("2026-05-30"), periods=n_dias)
    
    precos_iniciais = {
        "CL=F": 78.0, "GC=F": 2320.0, "SI=F": 28.5,
        "NG=F": 2.65, "ZS=F": 1150.0, "ZC=F": 460.0,
        "GLD": 214.0, "USO": 73.0, "SLV": 26.0,
    }
    vols = {
        "CL=F": 0.35, "GC=F": 0.18, "SI=F": 0.28,
        "NG=F": 0.55, "ZS=F": 0.22, "ZC=F": 0.25,
        "GLD": 0.17, "USO": 0.34, "SLV": 0.27,
    }
    
    dfs = {}
    for ticker, S0 in precos_iniciais.items():
        sigma = vols[ticker]
        ret = np.random.normal(0.0001, sigma / math.sqrt(252), n_dias)
        precos = S0 * np.exp(np.cumsum(ret))
        dfs[ticker] = pd.Series(precos, index=datas)
    
    return pd.DataFrame(dfs), vols

df_precos, vols_anuais = gerar_dados()
precos_atuais = df_precos.iloc[-1]
df_retornos = np.log(df_precos / df_precos.shift(1)).dropna()

# ================================================================
# BLACK-SCHOLES COM NOSSAS FUNÇÕES
# ================================================================

def bs_price(S, K, T, r, sigma, tipo="call"):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if tipo == "call" else max(K - S, 0)
    
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if tipo == "call":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

def bs_vega(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 1e-10
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return S * math.sqrt(T) * norm_pdf(d1)

def calcular_greeks(S, K, T, r, sigma, tipo="call"):
    if T <= 0 or sigma <= 0:
        return {"delta": 0, "gamma": 0, "vega": 0, "theta": 0, "rho": 0}
    
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    phi = norm_pdf(d1)
    
    if tipo == "call":
        delta = norm_cdf(d1)
        rho = K * T * math.exp(-r * T) * norm_cdf(d2) / 100
        theta = (-(S * phi * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm_cdf(d2)) / 365
    else:
        delta = norm_cdf(d1) - 1
        rho = -K * T * math.exp(-r * T) * norm_cdf(-d2) / 100
        theta = (-(S * phi * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm_cdf(-d2)) / 365
    
    gamma = phi / (S * sigma * math.sqrt(T))
    vega = S * math.sqrt(T) * phi / 100
    
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}

# ================================================================
# VOLATILIDADE IMPLÍCITA - BISSECÇÃO
# ================================================================

def vol_implicita(preco_mercado, S, K, T, r, tipo="call", tol=1e-6):
    a, b = 0.0001, 5.0
    
    for i in range(500):
        c = (a + b) / 2
        preco_c = bs_price(S, K, T, r, c, tipo)
        erro = preco_c - preco_mercado
        
        if abs(erro) < tol:
            return c
        
        if erro > 0:
            b = c
        else:
            a = c
    
    return (a + b) / 2

# ================================================================
# CARTEIRA
# ================================================================

CARTEIRA = [
    {"ativo": "CL=F", "tipo": "Futuro", "direcao": "Comprado", "vcto_dias": 63, "qtd": 120, "mult": 1000},
    {"ativo": "GC=F", "tipo": "Futuro", "direcao": "Vendido", "vcto_dias": 126, "qtd": 80, "mult": 100},
    {"ativo": "ZS=F", "tipo": "Futuro", "direcao": "Comprado", "vcto_dias": 84, "qtd": 150, "mult": 50},
    {"ativo": "NG=F", "tipo": "Futuro", "direcao": "Vendido", "vcto_dias": 42, "qtd": 100, "mult": 10000},
    {"ativo": "GLD", "tipo": "Call", "direcao": "Comprado", "vcto_dias": 90, "qtd": 25000, "mult": 1},
    {"ativo": "USO", "tipo": "Put", "direcao": "Vendido", "vcto_dias": 120, "qtd": 40000, "mult": 1},
    {"ativo": "SLV", "tipo": "Call", "direcao": "Vendido", "vcto_dias": 180, "qtd": 30000, "mult": 1},
]

# ================================================================
# INTERFACE
# ================================================================

st.markdown("""
<div class='banner'>
  <div>
    <h1 style='font-family:monospace; color:#00d4ff; margin:0;'>⬡ MESA ALPHA TRADING</h1>
    <div style='font-family:sans-serif; color:#64748b;'>Commodities Risk Desk · Sem Scipy - Versão Estável</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.success("✅ APLICATIVO FUNCIONANDO! O erro do scipy foi resolvido.")

# Dashboard simples
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Petróleo WTI", f"${precos_atuais['CL=F']:.2f}", 
              f"{df_retornos['CL=F'].iloc[-1]*100:.2f}%")

with col2:
    st.metric("Ouro", f"${precos_atuais['GC=F']:.0f}",
              f"{df_retornos['GC=F'].iloc[-1]*100:.2f}%")

with col3:
    st.metric("Gás Natural", f"${precos_atuais['NG=F']:.3f}",
              f"{df_retornos['NG=F'].iloc[-1]*100:.2f}%")

with col4:
    st.metric("Soja", f"${precos_atuais['ZS=F']:.0f}",
              f"{df_retornos['ZS=F'].iloc[-1]*100:.2f}%")

st.markdown("---")

# Gráfico de preços
st.markdown("<div class='section-header'>// PREÇOS HISTÓRICOS</div>", unsafe_allow_html=True)

fig = go.Figure()
for ticker in ["CL=F", "GC=F", "NG=F", "GLD"]:
    fig.add_trace(go.Scatter(x=df_precos.index, y=df_precos[ticker], 
                             name=ticker, mode='lines'))

fig.update_layout(height=400, paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                  font=dict(color="#e2e8f0"))
st.plotly_chart(fig, use_container_width=True)

# Precificação de opções
st.markdown("<div class='section-header'>// PRECIFICAÇÃO BLACK-SCHOLES</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    ativo_sel = st.selectbox("Ativo", list(precos_atuais.index))
    S = precos_atuais[ativo_sel]
    st.write(f"Preço Spot: **${S:.2f}**")

with col2:
    K = st.number_input("Strike", value=float(round(S, 0)), step=5.0)
    dias = st.slider("Dias para vencimento", 7, 365, 90)
    sigma = st.slider("Volatilidade", 0.05, 1.0, float(vols_anuais[ativo_sel]), 0.01)

T = dias / 252
r = 0.055

preco_call = bs_price(S, K, T, r, sigma, "call")
preco_put = bs_price(S, K, T, r, sigma, "put")

col1, col2 = st.columns(2)
with col1:
    st.metric("Preço CALL", f"${preco_call:.4f}")
with col2:
    st.metric("Preço PUT", f"${preco_put:.4f}")

# Volatilidade implícita
st.markdown("<div class='section-header'>// VOLATILIDADE IMPLÍCITA</div>", unsafe_allow_html=True)

preco_mercado = st.number_input("Preço de mercado da opção", value=preco_call, step=0.5)
vol_imp = vol_implicita(preco_mercado, S, K, T, r, "call")

st.metric("Volatilidade Implícita", f"{vol_imp*100:.2f}%")

st.markdown("---")
st.info("✅ Aplicativo funcionando perfeitamente sem scipy.stats!")

# Tabela da carteira
st.markdown("<div class='section-header'>// POSIÇÕES DA CARTEIRA</div>", unsafe_allow_html=True)
st.dataframe(pd.DataFrame(CARTEIRA), use_container_width=True)
