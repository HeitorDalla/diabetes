# 🏢 Projeto de Previsão de Diabetes

Este projeto visa construir um modelo de **classificação supervisionada** capaz de prever a **presença de diabetes** com base em características clínicas como **nível de glicose**, **pressão arterial**, **índice de massa corporal (IMC)**, entre outras variáveis.

---

## ⚙️ Tecnologias Utilizadas

- **Python** – Linguagem de programação
- **Pandas** – Manipulação e análise de dados
- **NumPy** – Geração de dados e cálculos numéricos
- **Scikit-learn** – Algoritmos de classificação, avaliação e particionamento dos dados
- **Pickle** – Serialização do modelo final

---

## 📁 Dados Utilizados

- **Fonte**: Pima Indians Diabetes Dataset (UCI Machine Learning Repository / Kaggle)  
- **Tamanho**: 768 amostras  
- **Variáveis**:
  - `Pregnancies` – Número de gravidezes  
  - `Glucose` – Concentração de glicose em jejum (mg/dL)  
  - `BloodPressure` – Pressão arterial diastólica (mm Hg)  
  - `SkinThickness` – Espessura da dobra cutânea do tríceps (mm)  
  - `Insulin` – Nível de insulina (μU/mL)  
  - `BMI` – Índice de Massa Corporal (kg/m²)  
  - `DiabetesPedigreeFunction` – Fator hereditário de diabetes  
  - `Age` – Idade da paciente (anos)  
  - `Outcome` – Diagnóstico de diabetes (0 = Não, 1 = Sim)  

---

## 🎯 Objetivo

Construir um modelo de **classificação** que consiga prever com alta precisão se uma pessoa tem ou não diabetes com base em seus dados clínicos.

---

## 🚀 Como Executar

### 1. Clonar o Repositório

```bash
git clone https://github.com/HeitorDalla/ml_diabetes_regression.git
```

### 2. Instalar Dependências

```bash
pip install pandas numpy seaborn scikit-learn
```

### 3. Executar o Script

```bash
python index.py
```

## 🧠 Possíveis Melhorias Futuras

- Aplicação de escalonamento de dados com StandardScaler
- Inclusão de validação cruzada mais robusta
- Testar outros algoritmos (XGBoost, SVM, LightGBM)
- Criação de um dashboard com Streamlit para visualização interativa

---

## 👨‍💻 Autor

- **Heitor Giussani Dalla Villa**  
- 📧 [heitorvillavilla@gmail.com](mailto:heitorvillavilla@gmail.com)  
- 🔗 [LinkedIn](https://www.linkedin.com/in/heitordallavilla)

---

## 📝 Observações Finais

> Este projeto tem finalidade educacional e demonstra a aplicação de técnicas de Machine Learning em saúde pública.
> Pode ser expandido para uso clínico com dados reais, desde que respeitadas normas éticas e científicas.
