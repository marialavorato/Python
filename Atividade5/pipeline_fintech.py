import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. INGESTÃO E LIMPEZA
# ---------------------------------------------------------------
df = pd.read_csv(
    "transacoes.csv",
    encoding="latin1",                      # encoding legado
    usecols=["id_cliente", "data_transacao", "valor", "origem", "estado_cliente"],
    dtype={"id_cliente": "category", "origem": "category", "estado_cliente": "category"},
)
print(f"Linhas carregadas: {len(df)} | NaN em valor: {df['valor'].isna().sum()}")

# Imputação da mediana por estado (transform mantém o alinhamento de índice)
mediana_estado = df.groupby("estado_cliente", observed=True)["valor"].transform("median")
df["valor"] = df["valor"].fillna(mediana_estado)

# Coluna estática de rastreabilidade
df["plataforma"] = "Mobile"

# ---------------------------------------------------------------
# 2. ENGENHARIA TEMPORAL
# ---------------------------------------------------------------
df["data_transacao"] = pd.to_datetime(df["data_transacao"])          # datetime64[ns]
df["data_transacao"] = df["data_transacao"].dt.tz_localize("America/Sao_Paulo")

df["dia_semana"] = df["data_transacao"].dt.day_name()
df["mes"] = df["data_transacao"].dt.month

antes = len(df)
df = df.drop_duplicates(keep="first").reset_index(drop=True)
print(f"Duplicatas removidas: {antes - len(df)}")

# ---------------------------------------------------------------
# 3. FILTRO BITWISE VETORIZADO
# ---------------------------------------------------------------
filtro = (
    (df["mes"] == 9)
    & ((df["estado_cliente"] == "SP") | (df["estado_cliente"] == "RJ"))
    & (df["valor"] > 5000)
)
df_filtrado = df[filtro]
print(f"Transações de setembro em SP/RJ acima de R$ 5.000: {len(df_filtrado)}")

# ---------------------------------------------------------------
# 4. CRUZAMENTO E PIVOT
# ---------------------------------------------------------------
risco_dict = {"C100": "Baixo", "C101": "Alto", "C102": "Médio",
              "C103": "Baixo", "C104": "Alto"}
df["nivel_risco"] = df["id_cliente"].astype(str).map(risco_dict)

pivot = pd.pivot_table(
    df, values="valor", index="mes", columns="nivel_risco",
    aggfunc="sum", margins=True, margins_name="Total",
)
print("\nPivot (soma do valor por mês x nível de risco):")
print(pivot.round(2))

# ---------------------------------------------------------------
# 5. Z-SCORE POR ESTADO (vetorizado, sem for)
# ---------------------------------------------------------------
def zscore_por_grupo(df, coluna="valor", grupo="estado_cliente"):
    g = df.groupby(grupo, observed=True)[coluna]
    media = g.transform("mean")
    desvio = g.transform("std").replace(0, np.nan)   # evita divisão por zero
    return (df[coluna] - media) / desvio

df["zscore"] = zscore_por_grupo(df)
df_anomalias = df[df["zscore"] > 2.5].copy()
print(f"\nAnomalias (Z > 2.5): {len(df_anomalias)}")
print(df_anomalias.groupby("estado_cliente", observed=True)["valor"]
      .agg(["count", "min", "max"]))

# ---------------------------------------------------------------
# 6. VISUALIZAÇÃO (API orientada a objetos)
# ---------------------------------------------------------------
diario = df.set_index("data_transacao")["valor"].resample("D").sum()
media_movel = diario.rolling(7).mean()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(diario.index, diario.values, color="#1f77b4", alpha=0.6, label="Total diário")
ax.plot(media_movel.index, media_movel.values, color="#d62728", linewidth=2.5,
        label="Média móvel (7 dias)")
ax.set_ylim(bottom=0)
ax.set_title("Valor Total de Transações Diárias e Média Móvel de 7 Dias")
ax.set_xlabel("Data")
ax.set_ylabel("Valor total (R$)")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", ".")))
ax.grid(alpha=0.3)
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig("grafico_transacoes.png", dpi=150)
print("\nGráfico salvo em grafico_transacoes.png")

df_anomalias.to_csv("anomalias.csv", index=False)
