# Trabalho-TF-IDF
Sistema de Recomendação Odontológica com TF-IDF

Este projeto aplica o conceito de **TF-IDF (Term Frequency - Inverse Document Frequency)** e **similaridade do cosseno** para criar um sistema que recomenda **procedimentos odontológicos** com base em descrições fornecidas pelo usuário.


Tecnologias utilizadas:
- Python 3
- Scikit-learn
- Pandas
- NumPy
- ReportLab (para gerar o PDF)
- Google Colab (ambiente de desenvolvimento)

Funcionalidades do sistema:
Busca por procedimentos odontológicos usando similaridade de texto
Normaliza textos (remove acentos e converte para minúsculas)
Usa TF-IDF com stop words em português
Retorna os 3 procedimentos mais relevantes com scores de similaridade
Interface interativa no terminal

Exemplo de uso
```bash
Digite o procedimento desejado: clareamento
Recomendações:
1. Clareamento dental
2. Limpeza dental
3. Profilaxia infantil

Digite 'sair' para encerrar.



Instruções para executar o arquivo:
1. Salve o código em um arquivo
2. Instale as dependências necessárias (se ainda não tiver):
pip install scikit-learn
3. Execute o arquivo.
