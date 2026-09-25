# Detecção de Fraude em Cartão de Crédito

Modelo de classificação para identificar transações fraudulentas em um dataset real e fortemente desbalanceado (~0,17% de fraudes), comparando Regressão Logística, Random Forest e XGBoost, com ajuste de threshold, tuning de hiperparâmetros e interpretabilidade via SHAP.

## Dataset

- Fonte: [creditcard.csv](https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv) (transações europeias, 2013).
- 30 features: `Time`, `Amount` e `V1`...`V28` (já anonimizadas via PCA).
- Target: `Class` (0 = normal, 1 = fraude).
- Desbalanceamento severo: **99,83% normal vs. 0,17% fraude**.

## Pipeline

1. **Carregamento e exploração** — leitura do CSV, checagem da distribuição de classes.
2. **Feature engineering** — `log1p(Amount)` para reduzir assimetria/outliers; `StandardScaler` em `Amount`.
3. **Split treino/teste** — holdout 70/30 com `stratify=y`.
4. **Baseline** — Regressão Logística, avaliada com `classification_report`, curva ROC/AUC e curva Precision-Recall.
5. **Balanceamento de classes** — UnderSampling e SMOTE (exploratórios).
6. **Modelos** — Random Forest (`class_weight='balanced'`) e XGBoost (`scale_pos_weight`).
7. **Ajuste de threshold** — recalibração do ponto de corte de decisão (0.5 → 0.3).
8. **Tuning** — `GridSearchCV` (max_depth, n_estimators) otimizando recall.
9. **Interpretabilidade** — importância de variáveis do XGBoost e SHAP.

## Resultados (classe 1 = fraude)

| Modelo                          | Precision | Recall | F1-score | AUC    |
|----------------------------------|-----------|--------|----------|--------|
| Regressão Logística (threshold 0.5) | 0.85      | 0.68   | 0.76     | 0.935  |
| Regressão Logística (threshold 0.3) | 0.77      | 0.70   | 0.73     | —      |
| Random Forest (balanced)         | 0.84      | 0.76   | 0.79     | —      |
| **XGBoost**                      | **0.94**  | **0.78** | **0.85** | —      |

XGBoost é o melhor modelo do notebook em todas as métricas de fraude. Baixar o threshold da Regressão Logística aumenta recall, mas derruba a precisão mais do que o ganho justifica.

## Como executar

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost shap matplotlib
jupyter notebook 'Anomaly_Trouble-DIO.ipynb'
```

O dataset é baixado diretamente da URL na primeira célula — não é necessário baixar manualmente.

## Limitações conhecidas / próximos passos

- [ ] `SMOTE` está sendo aplicado em `X, y` (dataset completo) em vez de `X_train, y_train` — padrão incorreto (risco de vazamento de dados), mesmo não sendo usado no treino final.
- [ ] `df_under` (undersampling) e a `Pipeline` de Regressão Logística são gerados mas nunca avaliados — código morto a remover ou completar.
- [ ] `GridSearchCV` encontra os melhores hiperparâmetros mas o modelo (`best_estimator_`) não é reavaliado com `classification_report` — falta confirmar se o tuning realmente melhorou o XGBoost.
- [ ] `Amount`, `Amount_log` e `Amount_scaled` coexistem em `X` — redundante, gera multicolinearidade.
- [ ] Gráfico de importância de variáveis usa índices numéricos em vez dos nomes das colunas.
- [ ] AUC e curva Precision-Recall só foram calculados para a Regressão Logística — faltam para Random Forest e XGBoost, que são os modelos mais fortes.
- [ ] Regressão Logística não convergiu em 2000 iterações (`ConvergenceWarning`) — aumentar `max_iter` ou revisar o pré-processamento.

## Tecnologias

Python · pandas · NumPy · scikit-learn · imbalanced-learn · XGBoost · SHAP · Matplotlib
