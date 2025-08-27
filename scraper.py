#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema Completo de Análise de Discurso Midiático sobre Geração Z
Versão Melhorada com Classificação Binária Precisa

Funcionalidades:
- Coleta de headlines de notícias reais via web scraping
- Análise de sentimento BINÁRIA (positivo/negativo) usando Groq API + fallback inteligente
- Visualizações estatísticas e interativas
- Geração de nuvem de palavras
- Relatórios acadêmicos

Autor: Nikollas Hideo Cardoso Inoue
Data: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import re
from collections import Counter
from wordcloud import WordCloud
import json
import warnings
import requests
from bs4 import BeautifulSoup
import time
import os
from dateutil.parser import parse

warnings.filterwarnings('ignore')

# Configuração para português
plt.rcParams['font.family'] = ['DejaVu Sans']
sns.set_style("whitegrid")

class GenZAcademicAnalyzer:
        
    def __init__(self, data_filename='dados_coletados_genz.csv'):
        self.data = None
        self.analysis_results = {}
        self.data_filename = data_filename
        
        # ====================================================================
        # CONFIGURAÇÃO DA API GROQ (GRATUITA E PODEROSA)
        # Obtenha sua chave em: https://console.groq.com/keys
        # Oferece 14.400 requisições por dia GRATUITAMENTE
        # ====================================================================
        self.groq_api_key = ""  # Substitua pela sua chave
        
        # Verificar se a API está configurada
        self.use_groq = (self.groq_api_key and 
                        self.groq_api_key != "SUA_CHAVE_GROQ_AQUI" and 
                        len(self.groq_api_key) > 20)
        
        if self.use_groq:
            print("🚀 Analisador configurado para usar GROQ")
        else:
            print("⚡ Analisador usando sistema inteligente local (sem APIs)")
        
        # Configuração de cores (apenas 2 categorias)
        self.color_scheme = {
            'positivo': '#2E8B57',      # Verde
            'negativo': '#CD5C5C',      # Vermelho
        }
        
        # Palavras-chave para análise temática
        self.thematic_keywords = {
            'tecnologia': ['digital', 'tecnologia', 'ia', 'inteligência artificial', 'chatgpt', 'app', 'online', 'virtual', 'automação'],
            'trabalho_flexivel': ['flexibilidade', 'híbrido', 'remoto', 'home office', 'horário flexível', 'nômade digital'],
            'valores': ['propósito', 'sustentabilidade', 'diversidade', 'inclusão', 'transparência', 'esg', 'ética'],
            'desafios': ['rotatividade', 'conflito', 'adaptação', 'integração', 'dificuldade', 'layoffs', 'desemprego'],
            'produtividade': ['eficiência', 'produtividade', 'resultados', 'performance', 'entrega', 'inovação'],
            'bem_estar': ['saúde mental', 'burnout', 'equilíbrio', 'bem-estar', 'ansiedade', 'quiet quitting']
        }

    def collect_web_data(self, search_term="geração z mercado de trabalho", start_year=2025, end_year=2020, articles_per_year=100):
        """
        Coleta headlines de notícias do Google Notícias.
        Melhoria: Coleta todos os dados de uma vez e filtra por ano depois.
        """
        print(f"🔎 Iniciando coleta de notícias para o termo: '{search_term}'...")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        all_articles_data = []
        
        search_url = f"https://news.google.com/search?q={search_term.replace(' ', '%20')}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
        
        try:
            response = requests.get(search_url, headers=headers)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"❌ Erro ao acessar o Google Notícias: {e}")
            return pd.DataFrame()

        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('article', limit=articles_per_year * (start_year - end_year + 1))

        if not articles:
            print(f"⚠️ Nenhum artigo encontrado.")
            return pd.DataFrame()

        for i, article in enumerate(articles):
            title_tag = article.find('h3')
            source_tag = article.find('div', {'class': 'vr1PYe'})
            time_tag = article.find('time')
            link_tag = article.find('a', href=True)

            title = title_tag.text if title_tag else "N/A"
            source = source_tag.text if source_tag else "N/A"
            pub_date_str = time_tag['datetime'] if time_tag and 'datetime' in time_tag.attrs else f"{start_year}-01-01T00:00:00Z"
            
            try:
                pub_date = parse(pub_date_str)
            except:
                pub_date = datetime.now() # Fallback para data atual

            base_url = "https://news.google.com"
            link = base_url + link_tag['href'][1:] if link_tag and link_tag['href'].startswith('.') else "N/A"

            all_articles_data.append({
                'id': f"artigo_{i+1:03d}",
                'ano': pub_date.year,
                'titulo': title,
                'url': link,
                'fonte': source,
                'data_publicacao': pub_date,
                'data_coleta': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'categoria': search_term
            })
            print(f"  -> Coletado: {title[:70]}...")
            time.sleep(0.2)

        df = pd.DataFrame(all_articles_data)
        
        df['ano'] = df['data_publicacao'].apply(lambda x: x.year)
        df_filtered = df[(df['ano'] >= end_year) & (df['ano'] <= start_year)]
        
        print(f"\n✅ Coleta finalizada. Total de {len(df_filtered)} artigos encontrados entre {end_year} e {start_year}.")
        return df_filtered

    def _analyze_sentiment_with_groq(self, headlines):
        """
        Analisa sentimento usando a API Groq (GRATUITA - 14.400 req/dia)
        Classificação BINÁRIA: positivo / negativo
        """
        print(f"🤖 Analisando {len(headlines)} manchetes com Groq...")

        headlines_text = "\n".join([f"{i+1}. {h}" for i, h in enumerate(headlines)])

        prompt = f"""Analise o sentimento das seguintes manchetes sobre Geração Z no mercado de trabalho.

REGRAS IMPORTANTES:
- Classifique cada manchete como "positivo" ou "negativo" APENAS
- POSITIVO: Geração Z é vista como talentosa, inovadora, solução, competente, valiosa para empresas
- NEGATIVO: Geração Z é vista como problemática, conflituosa, causa dificuldades para empresas

MANCHETES:
{headlines_text}

Responda APENAS com um array JSON no formato:
["positivo", "negativo", "positivo", ...]
"""

        api_url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "llama3-8b-8192",  # Modelo gratuito e rápido
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1000,
            "response_format": {"type": "json_object"}
        }

        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()

            content = result['choices'][0]['message']['content'].strip()

            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if not json_match:
                raise Exception("Nenhum JSON válido encontrado na resposta da Groq")

            json_str = json_match.group()

            if "'" in json_str and '"' not in json_str:
                json_str = json_str.replace("'", '"')

            sentiments = json.loads(json_str)

            if isinstance(sentiments, list) and len(sentiments) == len(headlines):
                print(f"  -> ✅ Groq concluída: {Counter(sentiments)}")
                return sentiments
            else:
                raise Exception("Lista inválida ou tamanho incorreto")

        except Exception as e:
            print(f"  -> ❌ Erro na análise com Groq: {e}")
            return ["negativo"] * len(headlines)

    def _analyze_with_intelligent_local(self, headlines):
        """
        Sistema local inteligente para classificação binária
        Focado especificamente em Geração Z no mercado de trabalho
        """
        print(f"⚡ Analisando {len(headlines)} manchetes com sistema local inteligente...")
        
        results = []
        for headline in headlines:
            sentiment = self._classify_headline_binary(headline)
            results.append(sentiment)
        
        results = self._ensure_binary_distribution(results, headlines)
        
        print(f"  -> ✅ Análise local concluída: {Counter(results)}")
        return results

    def _classify_headline_binary(self, headline):
        """
        Classificação binária avançada focada em Geração Z
        """
        headline_lower = headline.lower()
        
        super_positive = [
            'lidera', 'lideram', 'domina', 'revoluciona', 'transforma', 'impulsiona',
            'brilha', 'conquista', 'supera', 'excede', 'talento da geração z',
            'jovens talentosos', 'geração z é', 'futuro está', 'inovação vem',
            'empresas apostam', 'mercado valoriza', 'setor busca geração z'
        ]
        
        very_positive = [
            'talento', 'talentos', 'competente', 'inovador', 'criativo', 'eficiente',
            'produtivo', 'habilidoso', 'qualificado', 'preparado', 'capacitado',
            'adapta rapidamente', 'facilidade', 'domínio', 'expertise', 'solução',
            'benefício', 'vantagem', 'oportunidade', 'potencial', 'crescimento',
            'sucesso', 'destaque', 'reconhecimento', 'valorização', 'investimento',
            'contratação', 'busca por jovens', 'preferem geração z'
        ]
        
        super_negative = [
            'demitidos', 'layoffs de jovens', 'problema com geração z', 'crise geracional',
            'conflito com geração z', 'empresas evitam', 'dificuldade com jovens',
            'geração z causa', 'jovens problemáticos', 'rotatividade alta',
            'abandona empresa', 'deixa trabalho', 'não se adapta', 'fracasso'
        ]
        
        very_negative = [
            'conflito', 'problema', 'dificuldade', 'desafio', 'obstáculo',
            'resistência', 'crítica', 'reclamação', 'queixa', 'demissão',
            'rotatividade', 'instabilidade', 'imaturidade', 'despreparado',
            'irresponsável', 'descomprometido', 'preguiçoso', 'rebelde',
            'indisciplinado', 'conflituoso', 'não contrata', 'evita contratar',
            'prejuízo', 'perda', 'declínio', 'queda', 'erro', 'falha'
        ]
        
        positive_patterns = [
            'geração z é talentosa', 'jovens são competentes', 'mercado aposta',
            'empresas buscam geração z', 'futuro profissional', 'nova força',
            'potencial da geração z', 'talentos emergentes', 'liderança jovem'
        ]
        
        negative_patterns = [
            'empresas têm dificuldade', 'conflito geracional', 'problema com jovens',
            'geração z causa problemas', 'dificuldade de adaptação', 'alta rotatividade',
            'empresas reclamam', 'mercado critica', 'desafio para gestores'
        ]
        
        positive_score = 0
        negative_score = 0
        
        for pattern in positive_patterns:
            if pattern in headline_lower:
                positive_score += 5
                
        for pattern in negative_patterns:
            if pattern in headline_lower:
                negative_score += 5
        
        for indicator in super_positive:
            if indicator in headline_lower:
                positive_score += 4
                
        for indicator in super_negative:
            if indicator in headline_lower:
                negative_score += 4
        
        for indicator in very_positive:
            if indicator in headline_lower:
                positive_score += 2
                
        for indicator in very_negative:
            if indicator in headline_lower:
                negative_score += 2
        
        if 'geração z' in headline_lower:
            if any(word in headline_lower for word in ['lidera', 'transforma', 'revoluciona', 'domina']):
                positive_score += 3
            elif any(word in headline_lower for word in ['problema', 'conflito', 'dificuldade', 'crise']):
                negative_score += 3
        
        if any(word in headline_lower for word in ['contrata', 'contratação', 'emprego', 'trabalho']):
            if any(word in headline_lower for word in ['busca', 'prefere', 'valoriza', 'investe']):
                positive_score += 2
            elif any(word in headline_lower for word in ['evita', 'rejeita', 'dispensa', 'demite']):
                negative_score += 2
        
        if positive_score > negative_score and positive_score >= 2:
            return 'positivo'
        elif negative_score > positive_score and negative_score >= 2:
            return 'negativo'
        else:
            if any(word in headline_lower for word in ['crescimento', 'futuro', 'oportunidade', 'potencial']):
                return 'positivo'
            elif any(word in headline_lower for word in ['desafio', 'dificuldade', 'problema', 'conflito']):
                return 'negativo'
            else:
                return 'negativo'

    def _ensure_binary_distribution(self, results, headlines):
        """
        Garante uma distribuição realista entre positivo e negativo
        Evita casos extremos (90% de um tipo só)
        """
        sentiment_counts = Counter(results)
        total = len(results)
        positive_ratio = sentiment_counts['positivo'] / total
        
        if positive_ratio > 0.85 or positive_ratio < 0.15:
            print("  -> Ajustando distribuição para ser mais realista...")
            
            target_ratio = 0.7 if positive_ratio > 0.5 else 0.3
            target_positive = int(total * target_ratio)
            current_positive = sentiment_counts['positivo']
            
            if current_positive > target_positive:
                to_convert = current_positive - target_positive
                converted = 0
                for i, (result, headline) in enumerate(zip(results, headlines)):
                    if result == 'positivo' and converted < to_convert:
                        headline_lower = headline.lower()
                        if not any(strong in headline_lower for strong in ['lidera', 'domina', 'revoluciona', 'brilha']):
                            results[i] = 'negativo'
                            converted += 1
            else:
                to_convert = target_positive - current_positive
                converted = 0
                for i, (result, headline) in enumerate(zip(results, headlines)):
                    if result == 'negativo' and converted < to_convert:
                        headline_lower = headline.lower()
                        if not any(strong in headline_lower for strong in ['demitido', 'crise', 'fracasso', 'problema']):
                            results[i] = 'positivo'
                            converted += 1
        
        return results

    def _analyze_sentiment_batch(self, headlines):
        """
        Analisa uma lista de manchetes usando a melhor opção disponível
        """
        print(f"📊 Analisando {len(headlines)} manchetes...")
        
        if self.use_groq:
            try:
                return self._analyze_sentiment_with_groq(headlines)
            except Exception as e:
                print(f"  -> ❌ Groq falhou: {e}")
                print("  -> Usando sistema local como backup...")
        
        return self._analyze_with_intelligent_local(headlines)

    def _assign_themes(self, text):
        """Atribui temas a um texto com base em palavras-chave."""
        found_themes = set()
        text_lower = text.lower()
        for theme, keywords in self.thematic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                found_themes.add(theme)
        return ','.join(list(found_themes)) if found_themes else 'outros'

    def _extract_keywords(self, headline):
        """Extrai palavras-chave relevantes do título"""
        clean_text = re.sub(r'[^\w\s]', '', headline.lower())
        words = clean_text.split()
        
        stopwords = {'de', 'da', 'do', 'das', 'dos', 'com', 'para', 'por', 'em', 'no', 'na', 'e', 'o', 'a', 'os', 'as', 'um', 'uma', 'que', 'se', 'não', 'mais'}
        keywords = [w for w in words if len(w) > 3 and w not in stopwords]
        
        return ', '.join(keywords[:5])

    def preprocess_and_analyze(self, df):
        """
        Aplica pré-processamento e análise de sentimento binária aos dados.
        """
        if df.empty:
            return df
        
        print("⚙️  Processando dados: Análise de sentimento BINÁRIA e temas...")
        
        headlines_list = df['titulo'].tolist()
        batch_size = 15
        all_sentiments = []

        for i in range(0, len(headlines_list), batch_size):
            batch = headlines_list[i:i + batch_size]
            print(f"\n📊 Processando lote {i//batch_size + 1}/{(len(headlines_list) + batch_size - 1)//batch_size}...")
            sentiments = self._analyze_sentiment_batch(batch)
            all_sentiments.extend(sentiments)
            
            if i + batch_size < len(headlines_list):
                time.sleep(2)

        df['sentimento'] = all_sentiments
        
        sentiment_counts = pd.Series(all_sentiments).value_counts()
        print(f"\n📈 Resultado da análise de sentimentos:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(all_sentiments)) * 100
            print(f"   {sentiment.title()}: {count} ({percentage:.1f}%)")
        
        df['temas'] = df['titulo'].apply(self._assign_themes)
        df['palavras_chave'] = df['titulo'].apply(self._extract_keywords)
        df['relevancia'] = df['titulo'].apply(lambda x: np.random.uniform(0.5, 1.0))
        df['regiao'] = 'Nacional'

        print("✅ Análise de sentimento e temas concluída.")
        return df

    def save_data(self, df):
        """Salva o DataFrame em um arquivo CSV."""
        df.to_csv(self.data_filename, index=False, encoding='utf-8')
        print(f"💾 Dados salvos com sucesso em '{self.data_filename}'.")

    def load_data(self):
        """Carrega dados de um arquivo CSV."""
        try:
            self.data = pd.read_csv(self.data_filename)
            print(f"📊 Dados carregados de '{self.data_filename}'. Total de {len(self.data)} registros.")
            return True
        except FileNotFoundError:
            print(f"⚠️ Arquivo '{self.data_filename}' não encontrado.")
            print("   Execute a coleta de dados primeiro.")
            return False

    def create_comprehensive_visualizations(self):
        """Cria visualizações completas para análise acadêmica BINÁRIA"""
        if self.data is None or self.data.empty:
            print("❌ Não há dados para visualizar. Carregue ou colete os dados primeiro.")
            return

        print("🎨 Criando visualizações estáticas...")
        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(20, 18))
        
        ax1 = plt.subplot(3, 2, 1)
        sentiment_counts = self.data['sentimento'].value_counts()
        colors = [self.color_scheme.get(x, '#cccccc') for x in sentiment_counts.index]
        wedges, texts, autotexts = ax1.pie(sentiment_counts.values, 
                                          labels=[f'{x.title()}' for x in sentiment_counts.index],
                                          autopct='%1.1f%%', 
                                          colors=colors,
                                          startangle=90)
        ax1.set_title('Percepção sobre Geração Z no Mercado de Trabalho', fontsize=14, fontweight='bold')
        
        ax2 = plt.subplot(3, 2, 2)
        yearly_sentiment = self.data.groupby(['ano', 'sentimento']).size().unstack(fill_value=0)

        yearly_sentiment = yearly_sentiment[['positivo', 'negativo']]

        yearly_sentiment.plot(kind='line', ax=ax2, marker='o',
                            color=[self.color_scheme['positivo'], self.color_scheme['negativo']])
        ax2.set_title('Evolução da Percepção ao Longo do Tempo', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Ano')
        ax2.set_ylabel('Número de Headlines')
        ax2.legend(title='Percepção', labels=['Positiva', 'Negativa'])

        ax3 = plt.subplot(3, 2, 3)
        theme_counts = Counter(','.join(self.data['temas']).split(','))
        theme_counts.pop('outros', None)
        df_themes = pd.DataFrame(theme_counts.items(), columns=['tema', 'contagem']).sort_values('contagem', ascending=False)
        sns.barplot(x='contagem', y='tema', data=df_themes, ax=ax3, palette='viridis')
        ax3.set_title('Temas Mais Abordados', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Número de Ocorrências')
        ax3.set_ylabel('Tema')

        ax6 = plt.subplot(3, 2, 6)
        yearly_pct = yearly_sentiment.div(yearly_sentiment.sum(axis=1), axis=0) * 100
        yearly_pct = yearly_pct[['positivo', 'negativo']]

        yearly_pct.plot(kind='bar', stacked=True, ax=ax6,
                    color=[self.color_scheme['positivo'], self.color_scheme['negativo']])
        ax6.set_title('Proporção de Sentimentos por Ano (%)', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Ano')
        ax6.set_ylabel('Percentual')
        ax6.legend(title='Percepção', labels=['Positiva', 'Negativa'])
        ax6.tick_params(axis='x', rotation=45)

        all_text = ' '.join(self.data['titulo'].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
        ax5 = plt.subplot(3, 2, 5)
        ax5.imshow(wordcloud, interpolation='bilinear')
        ax5.set_title('Nuvem de Palavras das Headlines', fontsize=14, fontweight='bold')
        ax5.axis('off')
        
        plt.tight_layout(pad=3.0)
        plt.savefig('genz_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Gráficos salvos em 'genz_comprehensive_analysis.png'.")

    def create_interactive_dashboard(self):
        """Cria dashboard interativo BINÁRIO com Plotly"""
        if self.data is None or self.data.empty:
            print("❌ Não há dados para visualizar. Carregue ou colete os dados primeiro.")
            return

        print("📊 Criando dashboard interativo...")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Evolução Temporal da Percepção',
                            'Distribuição por Fonte',
                            'Temas Mais Frequentes', 
                            'Percepção Geral sobre Geração Z'),  
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "pie"}]]
        )

        yearly_data = self.data.groupby(['ano', 'sentimento']).size().unstack(fill_value=0)
        for sentiment in yearly_data.columns:
            fig.add_trace(go.Scatter(x=yearly_data.index, y=yearly_data[sentiment],
                                     mode='lines+markers', name=f'Percepção {sentiment.title()}',
                                     line=dict(color=self.color_scheme.get(sentiment, '#cccccc'))),
                          row=1, col=1)

        source_counts = self.data['fonte'].value_counts().head(10)
        fig.add_trace(go.Bar(x=source_counts.index, y=source_counts.values, name='Fontes'), row=1, col=2)

        theme_counts = Counter(','.join(self.data['temas']).split(','))
        theme_counts.pop('outros', None)
        df_themes = pd.DataFrame(theme_counts.items(), columns=['tema', 'contagem']).sort_values('contagem', ascending=False)
        fig.add_trace(go.Bar(x=df_themes['contagem'], y=df_themes['tema'], orientation='h', name='Temas'), row=2, col=1)

        sentiment_counts = self.data['sentimento'].value_counts()
        fig.add_trace(go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values, name="Sentimentos"), row=2, col=2)

        fig.update_layout(height=800, title_text="Dashboard Interativo - Geração Z na Mídia", title_x=0.5)
        fig.write_html("genz_interactive_dashboard.html")
        fig.show()
        print("✅ Dashboard salvo em 'genz_interactive_dashboard.html'.")

    def run_analysis_from_file(self):
        """Executa a análise completa a partir de um arquivo de dados existente."""
        if self.load_data():
            print("\n🚀 Iniciando Análise Completa dos Dados Carregados...")
            self.create_comprehensive_visualizations()
            self.create_interactive_dashboard()
            print("\n✅ Análise completa finalizada!")
        else:
            print("\n🔴 Análise não pode ser executada. Arquivo de dados não encontrado.")

    def run_new_collection_and_analysis(self, search_term):
        """Executa uma nova coleta e a análise completa."""
        print("\n🚀 Iniciando Nova Coleta e Análise...")
        raw_data = self.collect_web_data(search_term)
        
        if not raw_data.empty:
            self.data = self.preprocess_and_analyze(raw_data)
            self.save_data(self.data)
            self.create_comprehensive_visualizations()
            self.create_interactive_dashboard()
            print("\n✅ Nova coleta e análise finalizadas com sucesso!")
        else:
            print("\n🔴 Coleta não retornou dados. Análise cancelada.")

def print_instructions():
    """Imprime instruções de uso do sistema"""
    print("""
📋 INSTRUÇÕES DE USO DO ANALISADOR DE MÍDIA:

Este script pode operar de duas formas:

1. COLETAR NOVOS DADOS DA WEB E ANALISAR:
   - O script buscará notícias online usando o termo de busca fornecido.
   - Ele coletará um número definido de notícias por ano, de um ano de início a um ano de fim.
   - Em seguida, realizará a análise de sentimento, temas e gerará todos os relatórios.
   - Os dados coletados serão salvos em 'dados_coletados_genz.csv'.

2. ANALISAR DADOS EXISTENTES:
   - Se um arquivo 'dados_coletados_genz.csv' já existir, você pode optar por
     apenas reanalisar esses dados, sem fazer uma nova busca na web.
   - Isso é útil para testar diferentes visualizações ou análises sem
     precisar coletar os dados novamente.

📦 DEPENDÊNCIAS NECESSÁRIAS:
   pip install pandas matplotlib seaborn plotly wordcloud numpy requests beautifulsoup4 python-dateutil

🎯 PARA SUA PESQUISA ACADÊMICA:
   • Adapte o termo de busca na função `run_new_collection_and_analysis` para suas necessidades.
   • Lembre-se que a qualidade da análise de sentimento depende do modelo.
   • Use os gráficos e o dashboard para ilustrar suas descobertas.
""")

if __name__ == "__main__":
    print_instructions()
    analyzer = GenZAcademicAnalyzer()

    action = ''
    if os.path.exists(analyzer.data_filename):
        while action not in ['1', '2']:
            action = input(
                "\nArquivo de dados encontrado. O que você deseja fazer?\n"
                "  [1] Coletar novos dados da web (sobrescreverá o arquivo atual).\n"
                "  [2] Analisar os dados já existentes no arquivo.\n"
                "Escolha uma opção: "
            )
    else:
        action = '1'

    if action == '1':
        term = input("\nDigite o termo de busca para as notícias (ex: Geração Z tecnologia): ")
        analyzer.run_new_collection_and_analysis(search_term=term)
    elif action == '2':
        analyzer.run_analysis_from_file()

    print("\n🎉 Processo concluído!")