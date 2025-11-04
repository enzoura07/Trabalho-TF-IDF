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

Como funciona:
1. O sistema carrega um **banco de dados com procedimentos odontológicos**.
2. Aplica a técnica de **TF-IDF** para transformar as descrições em vetores numéricos.
3. Calcula a **similaridade do cosseno** entre o que o usuário digita e os procedimentos cadastrados.
4. Retorna os **procedimentos mais semelhantes** com base no “ângulo” entre os vetores (quanto menor o ângulo, mais próximos os significados).

Exemplo de uso
```bash
Digite o procedimento desejado: clareamento
Recomendações:
1. Clareamento dental
2. Limpeza dental
3. Profilaxia infantil

Digite 'sair' para encerrar.
