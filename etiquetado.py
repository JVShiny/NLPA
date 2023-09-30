import pandas as pd

data = pd.read_excel('data.xlsx')

def etiquetar_opinion(opinion):
    if "Hermoso" in opinion.lower() or "Excelente" in opinion.lower() or "Me encanta" in opinion.lower():
        return "Excelente"
    elif "bueno" in opinion.lower() or "interesante" in opinion.lower():
        return "bueno"
    elif "malo" in opinion.lower() or "caro" in opinion.lower():
        return "malo"
    elif "Desgraciadamente" in opinion.lower() or "peor" in opinion.lower():
        return "Muy mal"
    else:
        return "Neutral"

data['Categoria'] = data['Opinion'].apply(etiquetar_opinion)
data.to_excel('data_etiquetada.xlsx', index=False)
