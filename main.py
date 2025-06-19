import pandas as pd
import matplotlib.pyplot as plt
import kagglehub
import numpy as np

# Download the dataset
path = kagglehub.dataset_download("yasserh/titanic-dataset")

# Load the CSV file
df = pd.read_csv(f"{path}/Titanic-Dataset.csv")

print("\nInformações sobre o dataset:")
print(f"Número de registros: {len(df)}")
print(f"Número de colunas: {len(df.columns)}")

print("\nColunas e tipos de dados:")
print(df.dtypes)


def survived_by_sex(df):
    survival_by_sex = df.groupby(['Sex', 'Survived']).size().unstack(fill_value=0)

    # Calcular percentuais de sobrevivência
    survival_rates = df.groupby('Sex')['Survived'].mean() * 100

    # Criar o gráfico de pizza
    plt.figure(figsize=(8, 6))
    colors = ['lightcoral', 'lightblue']
    wedges, texts, autotexts = plt.pie(survival_rates.values, 
                                    labels=[f'{sex}\n({rate:.1f}%)' for sex, rate in survival_rates.items()],
                                    colors=colors,
                                    autopct='%1.1f%%',
                                    startangle=90)

    plt.title('Taxa de Sobrevivência por Sexo - Titanic', fontsize=14, fontweight='bold')
    plt.axis('equal')  # Para manter o gráfico circular
    plt.tight_layout()
    plt.show()

def survivors_by_age_and_sex(df):
    # Remover valores nulos de idade
    df_clean = df.dropna(subset=['Age'])
    
    # Criar faixas etárias
    bins = [0, 18, 30, 45, 60, 100]
    labels = ['0-17', '18-29', '30-44', '45-59', '60+']
    df_clean['Age_Group'] = pd.cut(df_clean['Age'], bins=bins, labels=labels)
    
    # Filtrar apenas sobreviventes
    survivors = df_clean[df_clean['Survived'] == 1]
    
    # Agrupar por sexo e faixa etária
    survival_by_age_sex = survivors.groupby(['Sex', 'Age_Group']).size().unstack(fill_value=0)
    
    # Criar gráfico de barras
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(labels))
    width = 0.35
    
    # Dados para homens e mulheres
    men_data = survival_by_age_sex.loc['male'] if 'male' in survival_by_age_sex.index else [0] * len(labels)
    women_data = survival_by_age_sex.loc['female'] if 'female' in survival_by_age_sex.index else [0] * len(labels)
    
    # Criar barras
    bars1 = ax.bar(x - width/2, men_data, width, label='Homens', color='lightblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, women_data, width, label='Mulheres', color='lightcoral', alpha=0.8)
    
    # Adicionar valores nas barras
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Configurar o gráfico
    ax.set_xlabel('Faixa Etária', fontsize=12, fontweight='bold')
    ax.set_ylabel('Número de Sobreviventes', fontsize=12, fontweight='bold')
    ax.set_title('Sobreviventes do Titanic por Sexo e Faixa Etária', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def survival_by_class(df):
    # Calcular taxa de sobrevivência por classe
    survival_rates = df.groupby('Pclass')['Survived'].mean() * 100
    
    # Contar total de passageiros por classe
    total_passengers = df.groupby('Pclass').size()
    
    # Contar sobreviventes por classe
    survivors = df.groupby('Pclass')['Survived'].sum()
    
    # Criar gráfico de barras
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['gold', 'silver', '#CD7F32']  # Ouro, prata, bronze
    bars = ax.bar(survival_rates.index, survival_rates.values, 
                  color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Adicionar valores nas barras
    for i, (bar, rate, survivors_count, total) in enumerate(zip(bars, survival_rates.values, survivors, total_passengers)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%\n({survivors_count}/{total})', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Configurar o gráfico
    ax.set_xlabel('Classe do Passageiro', fontsize=12, fontweight='bold')
    ax.set_ylabel('Taxa de Sobrevivência (%)', fontsize=12, fontweight='bold')
    ax.set_title('Taxa de Sobrevivência por Classe - Titanic', fontsize=14, fontweight='bold')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['1ª Classe', '2ª Classe', '3ª Classe'])
    ax.set_ylim(0, max(survival_rates.values) + 15)
    ax.grid(axis='y', alpha=0.3)
    
    # Adicionar linha de média geral
    overall_survival_rate = df['Survived'].mean() * 100
    ax.axhline(y=overall_survival_rate, color='red', linestyle='--', 
               label=f'Taxa Geral: {overall_survival_rate:.1f}%')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir estatísticas
    print("\nTaxa de sobrevivência por classe:")
    for classe, rate in survival_rates.items():
        print(f"Classe {classe}: {rate:.1f}%")

survived_by_sex(df)
survivors_by_age_and_sex(df)
survival_by_class(df)
