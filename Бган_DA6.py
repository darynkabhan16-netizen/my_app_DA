import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ── Налаштування сторінки ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Business Dashboard",
    page_icon="📊",
    layout="wide"
)

# ── Завантаження даних ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("streamlit_dataset.csv")
    df["ConversionRate"] = df["ConversionRate"].fillna(df["ConversionRate"].median())
    return df

df = load_data()

# ══════════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Панель фільтрації
# ══════════════════════════════════════════════════════════════════════════════════
st.sidebar.title("📋 Панель фільтрації")
st.sidebar.markdown(
    """
    **Інструкції:**
    Використовуйте фільтри нижче для налаштування відображення даних.
    Усі графіки та таблиця оновлюються автоматично.
    """
)

# 1. Один вибір зі списку (selectbox)
selected_industry = st.sidebar.selectbox(
    "🏭 Галузь (один вибір):",
    options=["Всі"] + sorted(df["Industry"].unique().tolist())
)

# 2. Кілька виборів (multiselect)
selected_regions = st.sidebar.multiselect(
    "🌍 Регіони (кілька виборів):",
    options=sorted(df["Region"].unique().tolist()),
    default=sorted(df["Region"].unique().tolist())
)

# 3. Радіо-кнопки
selected_scenario = st.sidebar.radio(
    "📈 Сценарій:",
    options=["Всі", "Optimistic", "Baseline", "Pessimistic"]
)

# 4. Прапорець
show_profitable_only = st.sidebar.checkbox("✅ Показати лише прибуткові компанії", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**Роки:**")
selected_years = st.sidebar.multiselect(
    "📅 Роки:",
    options=sorted(df["Year"].unique().tolist()),
    default=sorted(df["Year"].unique().tolist())
)

# ── Застосування фільтрів ───────────────────────────────────────────────────────
filtered = df.copy()

if selected_industry != "Всі":
    filtered = filtered[filtered["Industry"] == selected_industry]

if selected_regions:
    filtered = filtered[filtered["Region"].isin(selected_regions)]

if selected_scenario != "Всі":
    filtered = filtered[filtered["Scenario"] == selected_scenario]

if show_profitable_only:
    filtered = filtered[filtered["Profit"] > 0]

if selected_years:
    filtered = filtered[filtered["Year"].isin(selected_years)]

# ══════════════════════════════════════════════════════════════════════════════════
# ГОЛОВНА ЧАСТИНА
# ══════════════════════════════════════════════════════════════════════════════════
st.title("📊 Business Analytics Dashboard")
st.markdown("Інтерактивний дашборд для аналізу фінансових показників компаній")

# ── KPI-метрики ────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("🏢 Компаній", len(filtered))
col2.metric("💰 Середній дохід", f"{filtered['Revenue'].mean():,.0f}" if len(filtered) > 0 else "—")
col3.metric("📈 Середній ROI", f"{filtered['ROI'].mean():.2f}" if len(filtered) > 0 else "—")
col4.metric("👥 Середня к-сть клієнтів", f"{filtered['Customers'].mean():,.0f}" if len(filtered) > 0 else "—")

st.markdown("---")

# ── Таблиця з вибором колонок ──────────────────────────────────────────────────
st.subheader("📋 Таблиця даних")

all_columns = df.columns.tolist()
selected_columns = st.multiselect(
    "Оберіть колонки для відображення:",
    options=all_columns,
    default=["Company", "Year", "Region", "Industry", "Revenue", "Profit", "ROI", "Scenario"]
)

if selected_columns and len(filtered) > 0:
    st.dataframe(
        filtered[selected_columns].reset_index(drop=True),
        use_container_width=True,
        height=300
    )
else:
    st.info("Немає даних для відображення. Змініть фільтри.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════════
# ГРАФІКИ
# ══════════════════════════════════════════════════════════════════════════════════
st.subheader("📈 Графічний аналіз")

if len(filtered) == 0:
    st.warning("Немає даних для побудови графіків. Змініть фільтри.")
else:
    # ── Рядок 1: 3 графіки ─────────────────────────────────────────────────────
    g1, g2, g3 = st.columns(3)

    # Графік 1 — Гістограма (matplotlib)
    with g1:
        st.markdown("**1. Розподіл доходів (Histogram)**")
        fig1, ax1 = plt.subplots(figsize=(5, 3.5))
        ax1.hist(filtered["Revenue"], bins=15, color="#4C72B0", edgecolor="white", alpha=0.85)
        ax1.set_xlabel("Revenue")
        ax1.set_ylabel("Кількість компаній")
        ax1.set_title("Розподіл Revenue")
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    # Графік 2 — Barplot по галузях (seaborn)
    with g2:
        st.markdown("**2. Середній Profit по галузях (Bar)**")
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        industry_profit = filtered.groupby("Industry")["Profit"].mean().reset_index()
        sns.barplot(data=industry_profit, x="Industry", y="Profit", palette="Blues_d", ax=ax2)
        ax2.set_title("Середній Profit")
        ax2.set_xlabel("Галузь")
        ax2.set_ylabel("Profit")
        ax2.tick_params(axis="x", rotation=15)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # Графік 3 — Boxplot ROI по регіонах (seaborn)
    with g3:
        st.markdown("**3. Розподіл ROI по регіонах (Box)**")
        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        sns.boxplot(data=filtered, x="Region", y="ROI", palette="Set2", ax=ax3)
        ax3.set_title("ROI по регіонах")
        ax3.set_xlabel("Регіон")
        ax3.set_ylabel("ROI")
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    st.markdown("")

    # ── Рядок 2: 2 графіки ─────────────────────────────────────────────────────
    g4, g5 = st.columns(2)

    # Графік 4 — Scatter plot (plotly)
    with g4:
        st.markdown("**4. AdBudget vs Revenue (Scatter — інтерактивний)**")
        fig4 = px.scatter(
            filtered,
            x="AdBudget", y="Revenue",
            color="Industry",
            size="Customers",
            hover_data=["Company", "Profit", "ROI"],
            title="AdBudget vs Revenue",
            template="plotly_white"
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Графік 5 — Heatmap кореляцій (seaborn)
    with g5:
        st.markdown("**5. Кореляційна матриця (Heatmap)**")
        num_cols = ["Revenue", "Expenses", "Profit", "ROI", "AdBudget", "Customers"]
        corr = filtered[num_cols].corr()
        fig5, ax5 = plt.subplots(figsize=(5, 4))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax5, linewidths=0.5)
        ax5.set_title("Кореляція між змінними")
        fig5.tight_layout()
        st.pyplot(fig5)
        plt.close(fig5)

    st.markdown("")

    # Графік 6 — Лінійний графік Revenue по роках (plotly)
    st.markdown("**6. Динаміка Revenue по роках та галузях (Line)**")
    line_data = filtered.groupby(["Year", "Industry"])["Revenue"].mean().reset_index()
    fig6 = px.line(
        line_data,
        x="Year", y="Revenue",
        color="Industry",
        markers=True,
        title="Середній Revenue по роках",
        template="plotly_white"
    )
    st.plotly_chart(fig6, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════════
# МОДЕЛІ ML
# ══════════════════════════════════════════════════════════════════════════════════
st.subheader("🤖 Машинне навчання")

if len(filtered) < 5:
    st.warning("Недостатньо даних для моделей. Змініть фільтри.")
else:
    tab1, tab2 = st.tabs(["📉 Лінійна регресія", "🔵 Кластеризація KMeans"])

    # ── Регресія ───────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("**Модель: Лінійна регресія — прогноз ROI на основі AdBudget**")

        reg_data = filtered[["AdBudget", "ROI"]].dropna()
        X = reg_data[["AdBudget"]].values
        y = reg_data["ROI"].values

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = model.score(X, y)

        col_r1, col_r2 = st.columns(2)
        col_r1.metric("Коефіцієнт (slope)", f"{model.coef_[0]:.6f}")
        col_r1.metric("Intercept", f"{model.intercept_:.4f}")
        col_r2.metric("R² Score", f"{r2:.4f}")
        col_r2.metric("К-сть спостережень", len(reg_data))

        fig_r, ax_r = plt.subplots(figsize=(8, 4))
        ax_r.scatter(X, y, alpha=0.6, color="#4C72B0", label="Дані")
        ax_r.plot(X, y_pred, color="red", linewidth=2, label=f"Регресія (R²={r2:.3f})")
        ax_r.set_xlabel("AdBudget")
        ax_r.set_ylabel("ROI")
        ax_r.set_title("Лінійна регресія: AdBudget → ROI")
        ax_r.legend()
        fig_r.tight_layout()
        st.pyplot(fig_r)
        plt.close(fig_r)

        # Прогноз
        st.markdown("**Прогноз ROI:**")
        user_budget = st.slider("Введіть AdBudget:", int(filtered["AdBudget"].min()), int(filtered["AdBudget"].max()), int(filtered["AdBudget"].mean()))
        predicted_roi = model.predict([[user_budget]])[0]
        st.success(f"При AdBudget = {user_budget:,} → прогнозований ROI = **{predicted_roi:.4f}**")

    # ── Кластеризація ──────────────────────────────────────────────────────────
    with tab2:
        st.markdown("**Модель: KMeans кластеризація компаній за Revenue, Profit, ROI**")

        n_clusters = st.slider("Кількість кластерів:", 2, 5, 3)

        cluster_data = filtered[["Revenue", "Profit", "ROI"]].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(cluster_data)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_data = cluster_data.copy()
        cluster_data["Cluster"] = kmeans.fit_predict(X_scaled).astype(str)

        fig_k = px.scatter(
            cluster_data,
            x="Revenue", y="Profit",
            color="Cluster",
            size="ROI",
            title=f"KMeans кластеризація ({n_clusters} кластери)",
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        st.plotly_chart(fig_k, use_container_width=True)

        st.markdown("**Центроїди кластерів (оригінальний масштаб):**")
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        centers_df = pd.DataFrame(centers, columns=["Revenue", "Profit", "ROI"])
        centers_df.index.name = "Кластер"
        st.dataframe(centers_df.round(2), use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("📊 Бган_DA6 | Streamlit Dashboard | Business Analytics")
