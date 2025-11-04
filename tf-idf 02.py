# ==========================================
# Sistema de Recomendação Odontológica (TF-IDF)
# ==========================================
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def normalizar(texto):
    """
    Normaliza o texto: converte para minúsculas e remove acentos (diacríticos).
    """
    texto = texto.lower()
    # Remove acentos (diacríticos)
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto)
                    if unicodedata.category(c) != 'Mn')
    return texto

procedimentos = {
    "Limpeza dental": "Limpeza dental: remoção de placa bacteriana e tártaro, polimento e aplicação de flúor.",
    "Restauração dental": "Restauração dental: reparar dentes danificados por cáries, fraturas ou desgaste.",
    "Extração dental": "Extração dental: remoção de dentes comprometidos por cáries profundas, infecção ou falta de espaço.",
    "Clareamento dental": "Clareamento dental: clarear o tom dos dentes com agentes clareadores.",
    "Canal (endodontia)": "Canal (endodontia): remover a polpa dentária infectada e selar o interior do dente.",
    "Implante dentário": "Implante dentário: colocação de pino de titânio no osso para substituir dentes ausentes.",
    "Aparelho ortodôntico": "Aparelho ortodôntico: dispositivo fixo ou móvel para corrigir o alinhamento dos dentes.",
    "Profilaxia infantil": "Profilaxia infantil: limpeza preventiva em crianças para evitar cáries e gengivite.",
    "Raspagem periodontal": "Raspagem periodontal: remoção de tártaro abaixo da gengiva em casos de periodontite.",
    "Prótese dentária": "Prótese dentária: substituição de dentes perdidos por próteses fixas ou removíveis."
}

nomes = list(procedimentos.keys())
# Aplica a normalização nas descrições para o vetorizador
descricoes = [normalizar(texto) for texto in procedimentos.values()]

# Lista de stop words em Português
portuguese_stop_words = [
    'de', 'a', 'o', 'que', 'e', 'é', 'do', 'da', 'em', 'um', 'uma', 'para', 'com', 'não', 'uma', 'os', 'as', 'dos', 'das', 'pelo', 'pela', 'pelos', 'pelas', 'ao', 'aos', 'à', 'às', 'dele', 'dela', 'deles', 'delas', 'aquele', 'aquela', 'aqueles', 'aquelas', 'isto', 'aquilo', 'este', 'esta', 'estes', 'estas', 'isso', 'esse', 'essa', 'esses', 'essas', 'no', 'na', 'nos', 'nas', 'por', 'mais', 'mas', 'ao', 'tempo', 'se', 'depois', 'quando', 'como', 'qual', 'ser', 'ter', 'ir', 'vir', 'estar', 'fazer', 'dizer', 'poder', 'ver', 'saber', 'querer', 'chegar', 'dar', 'falar', 'comer', 'beber', 'cantar', 'dançar', 'andar', 'correr', 'nadar', 'voar', 'dormir', 'acordar', 'levantar', 'sentar', 'cair', 'subir', 'descer', 'entrar', 'sair', 'abrir', 'fechar', 'ligar', 'desligar', 'começar', 'terminar', 'continuar', 'parar', 'mudar', 'achar', 'pensar', 'sentir', 'ouvir', 'ver', 'olhar', 'gostar', 'amar', 'odiar', 'precisar', 'usar', 'ter', 'haver', 'ser', 'estar', 'ir', 'vir', 'dar', 'fazer', 'dizer', 'poder', 'ver', 'saber', 'querer', 'chegar', 'dar', 'falar', 'comer', 'beber', 'cantar', 'dançar', 'andar', 'correr', 'nadar', 'voar', 'dormir', 'acordar', 'levantar', 'sentar', 'cair', 'subir', 'descer', 'entrar', 'sair', 'abrir', 'fechar', 'ligar', 'desligar', 'começar', 'terminar', 'continuar', 'parar', 'mudar', 'achar', 'pensar', 'sentir', 'ouvir', 'ver', 'olhar', 'gostar', 'amar', 'odiar', 'precisar', 'usar'
]

# Inicializa o vetorizador TF-IDF com as stop words em português e n-gramas de 1 e 2
vetorizador = TfidfVectorizer(stop_words=portuguese_stop_words, ngram_range=(1, 2))
# Ajusta (fit) e transforma (transform) as descrições em uma matriz TF-IDF
matriz_tfidf = vetorizador.fit_transform(descricoes)

print("=== Sistema de Recomendação Odontológica (TF-IDF) ===")
print("Digite o nome ou descrição do procedimento que deseja encontrar.")
print("Quando quiser sair, digite 'sair'.\n")

while True:
    # Coleta a entrada do usuário
    entrada = input("Qual procedimento você procura? ").strip().lower()
    
    if entrada == "sair":
        print("Encerrando o sistema... até logo!")
        break

    # Normaliza a entrada
    entrada = normalizar(entrada)
    
    # Transforma a entrada em vetor TF-IDF (usa apenas transform, pois já foi fitado)
    entrada_tfidf = vetorizador.transform([entrada])
    
    # Calcula a similaridade do cosseno entre a entrada e todos os procedimentos
    similaridades = cosine_similarity(entrada_tfidf, matriz_tfidf)[0]
    
    # Obtém os índices ordenados por similaridade (do maior para o menor)
    indices = similaridades.argsort()[::-1]

    print("\nProcedimentos mais semelhantes:")
    # Exibe os 3 procedimentos mais semelhantes
    for i in indices[:3]:
        print(f"→ {nomes[i]} — Similaridade: {similaridades[i]:.2f}")
        print(f"  Descrição: {procedimentos[nomes[i]]}\n")
    print("-" * 60)