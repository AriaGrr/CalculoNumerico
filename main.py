import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Carregar os dados
data = pd.read_csv('dados_banco.csv', header=(0))

# Converter para NumPy array
data = data.to_numpy()

# Variaveis
# Y 
saldocc = data[:,7]
# Y 
devedorcartao = data[:,9]
# Y 
inadimplente = data[:,10]
# X e Y 
salario = data[:,5]
# X 
saldopoupanca = data[:,6]
# X 
idade = data[:,3]

# Montando
# X
Xsalario = np.array(salario).reshape(-1, 1)
Xsaldopoupanca = np.array(saldopoupanca).reshape(-1, 1)
Xidade = np.array(idade).reshape(-1, 1)
# Y
Ysaldocc = np.array(saldocc)
Ydevedorcartao = np.array(devedorcartao)
Yinadimplente = np.array(inadimplente)
Ysalario = np.array(salario)

# | Independente (X) | Dependente (Y) |
# | idade | salario |
# | saldopoupanca | saldocc |
# | salario | devedorcartao |
# | idade | devedorcartao |
# | salario | inadimplente |

# Regressão Linear fit X e Y
modelo1 = LinearRegression().fit(Xidade, Ysalario)
modelo2 = LinearRegression().fit(Xsaldopoupanca, Ysaldocc)
modelo3 = LinearRegression().fit(Xsalario, Ydevedorcartao)
modelo4 = LinearRegression().fit(Xidade, Ydevedorcartao)
modelo5 = LinearRegression().fit(Xsalario, Yinadimplente)

# Print dos modelos
print("Modelos: Intercepto e Coeficiente")
print("Modelo 1: Idade e Salario")
# print(modelo1)
print("Intercepto:", modelo1.intercept_)
print("Coeficiente:", modelo1.coef_)
# print(modelo1.intercept_)
# print (modelo1.coef_)

print("Modelo 2: Saldo Poupança e Saldo CC")
# print(modelo2)
print("Intercepto:", modelo2.intercept_)
print("Coeficiente:", modelo2.coef_)
# print(modelo2.intercept_)
# print (modelo2.coef_)

print("Modelo 3: Salario e Devedor Cartão")
# print(modelo3)
print("Intercepto:", modelo3.intercept_)
print("Coeficiente:", modelo3.coef_)
# print(modelo3.intercept_)
# print (modelo3.coef_)

print("Modelo 4: Idade e Devedor Cartão")
# print(modelo4)
print("Intercepto:", modelo4.intercept_)
print("Coeficiente:", modelo4.coef_)
# print(modelo4.intercept_)
# print (modelo4.coef_)

print("Modelo 5: Salario e Inadimplente")
print(modelo5)
print("Intercepto:", modelo5.intercept_)
print("Coeficiente:", modelo5.coef_)
# print(modelo5.intercept_)
# print (modelo5.coef_)


# Previsão
# Y
Ysalario_pred = modelo1.predict(Xidade)
Ysaldocc_pred = modelo2.predict(Xsaldopoupanca)
Ydevedorcartao_pred = modelo3.predict(Xsalario)
Ydevedorcartao_pred2 = modelo4.predict(Xidade)
Yinadimplente_pred = modelo5.predict(Xsalario)

# EXEMPLO NUNO
# # mostra os dados
# plt.scatter(x, y, color = "b", marker = "o", s = 50) 
# # mostra a reta de regressão
# plt.plot(x, y_pred, color = "r") 
# plt.xlabel('x', fontsize = 15) 
# plt.ylabel('y', fontsize = 15) 
# plt.show(True) 
# plt.scatter(X, Y)
# plt.plot(X, modelo.predict(X), color = 'red')

# Plot
# X e Y
plt.scatter(Xidade, Ysalario, color='black')
plt.plot(Xidade, Ysalario_pred, color='blue', linewidth=3)
plt.xlabel('Idade')
plt.ylabel('Salario')
plt.show()

# X e Y
plt.scatter(Xsaldopoupanca, Ysaldocc, color='black')
plt.plot(Xsaldopoupanca, Ysaldocc_pred, color='blue', linewidth=3)
plt.xlabel('Saldo Poupança')
plt.ylabel('Saldo CC')
plt.show()

# X e Y
plt.scatter(Xsalario, Ydevedorcartao, color='black')
plt.plot(Xsalario, Ydevedorcartao_pred, color='blue', linewidth=3)
plt.xlabel('Salario')
plt.ylabel('Devedor Cartão')
plt.show()

# X e Y
plt.scatter(Xidade, Ydevedorcartao, color='black')
plt.plot(Xidade, Ydevedorcartao_pred2, color='blue', linewidth=3)
plt.xlabel('Idade')
plt.ylabel('Devedor Cartão')
plt.show()

# X e Y
plt.scatter(Xsalario, Yinadimplente, color='black')
plt.plot(Xsalario, Yinadimplente_pred, color='blue', linewidth=3)
plt.xlabel('Salario')
plt.ylabel('Inadimplente')
plt.show()

# R2
# X e Y
print('R2 Idade e Salario:', r2_score(Ysalario, Ysalario_pred))

# X e Y
print('R2 Saldo Poupança e Saldo CC:', r2_score(Ysaldocc, Ysaldocc_pred))

# X e Y
print('R2 Salario e Devedor Cartão:', r2_score(Ydevedorcartao, Ydevedorcartao_pred))

# X e Y
print('R2 Idade e Devedor Cartão:', r2_score(Ydevedorcartao, Ydevedorcartao_pred2))

# X e Y
print('R2 Salario e Inadimplente:', r2_score(Yinadimplente, Yinadimplente_pred))

# Coeficiente de correlação (VERIFICAR)
# X e Y
# print('Coeficiente de correlação Idade e Salario:', np.corrcoef(Xidade.T, Ysalario.T))

# # X e Y
# print('Coeficiente de correlação Saldo Poupança e Saldo CC:', np.corrcoef(Xsaldopoupanca.T, Ysaldocc.T))

# # X e Y
# print('Coeficiente de correlação Salario e Devedor Cartão:', np.corrcoef(Xsalario.T, Ydevedorcartao.T))

# # X e Y
# print('Coeficiente de correlação Idade e Devedor Cartão:', np.corrcoef(Xidade.T, Ydevedorcartao.T))

# # X e Y
# print('Coeficiente de correlação Salario e Inadimplente:', np.corrcoef(Xsalario.T, Yinadimplente.T))

# print('Coeficiente de correlação Idade e Salario:', np.corrcoef(Xidade.flatten(), Ysalario.flatten()))

# print('Coeficiente de correlação Saldo Poupança e Saldo CC:', np.corrcoef(Xsaldopoupanca.flatten(), Ysaldocc.flatten()))

# print('Coeficiente de correlação Salario e Devedor Cartão:', np.corrcoef(Xsalario.flatten(), Ydevedorcartao.flatten()))

# print('Coeficiente de correlação Idade e Devedor Cartão:', np.corrcoef(Xidade.flatten(), Ydevedorcartao.flatten()))

# print('Coeficiente de correlação Salario e Inadimplente:', np.corrcoef(Xsalario.flatten(), Yinadimplente.flatten()))

# Verificar o formato dos arrays
# print("Dar reshape no Y")
# print(Xidade.shape)
# print(Ysalario.shape)
# print(Xsaldopoupanca.shape)
# print(Ysaldocc.shape)
# print(Xsalario.shape)
# print(Ydevedorcartao.shape)
# print(Xidade.shape)
# print(Ydevedorcartao.shape)
# print(Xsalario.shape)
# print(Yinadimplente.shape)

# reshape() para ajustar dimensões antes de calcular correlação

# # Verificando as formas e ajustando
# for arr in [Xidade, Ysalario, Xsaldopoupanca, Ysaldocc, Xsalario, Ydevedorcartao, Yinadimplente]:
#     print(arr.shape)
#     if len(arr.shape) == 1:
#         arr = arr.reshape(-1, 1)

# Ysaldocc = np.array(Ysaldocc).reshape(-1, 1)
# Ydevedorcartao = np.array(Ydevedorcartao).reshape(-1, 1)
# Yinadimplente = np.array(Yinadimplente).reshape(-1, 1)
# Ysalario = np.array(Ysalario).reshape(-1, 1)
# print(Ysaldocc)
# print(Ydevedorcartao)
# print(Yinadimplente)
# print(Ysalario)

# # Calculando as correlações
# corr_coef_idade_salario = np.corrcoef(Xidade.T, Ysalario.T)[0, 1]

# corr_coef_saldopoupanca_saldocc = np.corrcoef(Xsaldopoupanca.T, Ysaldocc.T)[0, 1]

# corr_coef_salario_devedorcartao = np.corrcoef(Xsalario.T, Ydevedorcartao.T)[0, 1]

# corr_coef_idade_devedorcartao = np.corrcoef(Xidade.T, Ydevedorcartao.T)[0, 1]

# corr_coef_salario_inadimplente = np.corrcoef(Xsalario.T, Yinadimplente.T)[0, 1]

# # ... (calcular outras correlações)

# print('Coeficiente de correlação Idade e Salario:', corr_coef_idade_salario)

# print('Coeficiente de correlação Saldo Poupança e Saldo CC:', corr_coef_saldopoupanca_saldocc)

# print('Coeficiente de correlação Salario e Devedor Cartão:', corr_coef_salario_devedorcartao)

# print('Coeficiente de correlação Idade e Devedor Cartão:', corr_coef_idade_devedorcartao)

# print('Coeficiente de correlação Salario e Inadimplente:', corr_coef_salario_inadimplente)

# Ysaldocc = np.array(Ysaldocc).reshape(-1, 1)
# Ydevedorcartao = np.array(Ydevedorcartao).reshape(-1, 1)
# Yinadimplente = np.array(Yinadimplente).reshape(-1, 1)
# Ysalario = np.array(Ysalario).reshape(-1, 1)
# print(Ysaldocc)
# print(Ydevedorcartao)
# print(Yinadimplente)
# print(Ysalario)

# print(Ysaldocc.shape)
# print(Ysalario.shape)
# print(Ydevedorcartao.shape)
# print(Yinadimplente.shape)
# print(Xidade.shape)
# print(Xsalario.shape)
# print(Xsaldopoupanca.shape)

# # print(Ysalario.dtype)
# # print(Xsalario.dtype)

# # print(np.unique(Ysalario))

# # Tentar converter para numérico e lidar com NaN
# try:
#     Ysalario = pd.to_numeric(Ysalario, errors='coerce')
#     Ysalario.dropna(inplace=True)  # Remover linhas com NaN
# except ValueError:
#     print("Não foi possível converter todos os valores para numérico.")
# np.isnan(Ysalario).any()
# np.isinf(Ysalario).any()
# Ysaldocc.dtype
# Calculando as correlações
# corr_coef_idade_salario = np.corrcoef(Xidade.T, Ysalario.T)[0, 1]

# corr_coef_saldopoupanca_saldocc = np.corrcoef(Xsaldopoupanca.T, Ysaldocc.T)[0, 1]

# corr_coef_salario_devedorcartao = np.corrcoef(Xsalario.T, Ydevedorcartao.T)[0, 1]

# corr_coef_idade_devedorcartao = np.corrcoef(Xidade.T, Ydevedorcartao.T)[0, 1]

# corr_coef_salario_inadimplente = np.corrcoef(Xsalario.T, Yinadimplente.T)[0, 1]

# # ... (calcular outras correlações)

# print('Coeficiente de correlação Idade e Salario:', corr_coef_idade_salario)

# print('Coeficiente de correlação Saldo Poupança e Saldo CC:', corr_coef_saldopoupanca_saldocc)

# print('Coeficiente de correlação Salario e Devedor Cartão:', corr_coef_salario_devedorcartao)

# print('Coeficiente de correlação Idade e Devedor Cartão:', corr_coef_idade_devedorcartao)

# print('Coeficiente de correlação Salario e Inadimplente:', corr_coef_salario_inadimplente)


# # print(Ysaldocc.shape)
# print(Ysalario.shape)
# # print(Ydevedorcartao.shape)
# # print(Yinadimplente.shape)
# print(Xidade.shape)
# # print(Xsalario.shape)
# # print(Xsaldopoupanca.shape)

# print(Ysalario.dtype)
# print(Xidade.dtype)
# print(Ysalario)
# print(Xidade)

# Xidade = Xidade.astype(np.float64)
# Ysalario = Ysalario.astype(np.float64)
# print(Ysalario)
# print(Xidade)

# print(Ysalario.dtype)
# print(Xidade.dtype)
# print(Ysalario)
# print(Xidade)

# Converter todos para float
# X
Xsalario = Xsalario.astype(np.float64)
Xsaldopoupanca = Xsaldopoupanca.astype(np.float64)
Xidade = Xidade.astype(np.float64)
# Y
Ysalario = Ysalario.astype(np.float64)
Ysaldocc = Ysaldocc.astype(np.float64)
Ydevedorcartao = Ydevedorcartao.astype(np.float64)
Yinadimplente = Yinadimplente.astype(np.float64)

# print(Xidade)
# print(Xsalario)
# print(Xsaldopoupanca)

# print(Ysaldocc)
# print(Ydevedorcartao)
# print(Yinadimplente)
# print(Ysalario)

print('Coeficiente de correlação Idade e Salario: \n')

# print(np.corrcoef(Xidade, Ysalario))
# print("\n")
print(np.corrcoef(Xidade.T, Ysalario.T))
print("\n")
# print(np.corrcoef(Xidade.T, Ysalario.T)[0, 1])
# print("\n")
# print(np.corrcoef(Xidade.flatten(), Ysalario.flatten()))

print('Coeficiente de correlação Saldo Poupança e Saldo CC: \n')
print(np.corrcoef(Xsaldopoupanca.T, Ysaldocc.T))
print("\n")

print('Coeficiente de correlação Salario e Devedor Cartão: \n')
print(np.corrcoef(Xsalario.T, Ydevedorcartao.T))
print("\n")

print('Coeficiente de correlação Idade e Devedor Cartão: \n')
print(np.corrcoef(Xidade.T, Ydevedorcartao.T))
print("\n")

print('Coeficiente de correlação Salario e Inadimplente: \n')
print(np.corrcoef(Xsalario.T, Yinadimplente.T))
print("\n")