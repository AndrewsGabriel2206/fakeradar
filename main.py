#main para rodar tudo
import os

#Garantir que esta rodando tudo na pasta certa

os.chdir(r'C:\Users\andre\Downloads\trabalho\tcc')

print(" Passo 1: Processando dados....")
exec(open('fake.py', encoding='utf-8').read())

print("\n Passo 2: treinando modelo...")
exec(open('treino.py', encoding ='utf-8').read())


print("\n Tudo pronto!")