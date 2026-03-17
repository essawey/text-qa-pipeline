import pandas as pd
import re
from bs4 import BeautifulSoup
import language_tool_python
from collections import Counter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DataCleaner:
    def __init__(self):
        self.tool = language_tool_python.LanguageTool('en-US')

    def strip_html(self, text):
        soup = BeautifulSoup(str(text), "html.parser")
        return soup.get_text(separator=" ")

    def fix_spacing(self, text):
        text = str(text)
        text = re.sub(r'\s+([?.!,:;])', r'\1', text)
        text = re.sub(r'\(\s+', '(', text)
        text = re.sub(r'\s+\)', ')', text)
        text = re.sub(r'``\s+', '`` ', text)
        text = re.sub(r'\s+\'\'', " ''", text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def grammar_correction(self, text):
        text = str(text)
        matches = self.tool.check(text)
        return language_tool_python.utils.correct(text, matches)

    def clean_text(self, text):
        text = str(text)
        if not text.strip(): return ""
        text = self.strip_html(text)
        text = self.fix_spacing(text)
        text = self.grammar_correction(text)
        return text

    def close(self):
        self.tool.close()

def categorize_question_type(question, answer):
    q_lower = str(question).lower().strip()
    if q_lower.startswith('who'): return 'Person/Entity'
    elif q_lower.startswith('when'): return 'Date/Time'
    elif q_lower.startswith('where'): return 'Location'
    elif q_lower.startswith('why'): return 'Reason/Explanation'
    elif q_lower.startswith('how'): return 'Method/Quantity'
    elif q_lower.startswith('what'): return 'Object/Concept'
    elif q_lower.startswith('which'): return 'Choice/Identification'
    else: return 'Other'

def estimate_difficulty(question, answer):
    words = str(question).split()
    if len(words) <= 8:
        return 'Easy'
    elif len(words) <= 10:
        return 'Medium'
    else:
        return 'Hard'

def extract_domain(question, answer) -> str:
    if not hasattr(extract_domain, "_patterns"):
        domains = {
            'Pop Culture & Entertainment': [
                'movie', 'song', 'season', 'sings', 'sang', 'voice', 'show', 'series',
                'star', 'book', 'episode', 'episodes', 'film', 'music', 'story', 'actor',
                'actress', 'thrones', 'theme', 'singer', 'album', 'lyrics', 'potter', 'video', 'band'
            ],
            'Geography & Places': [
                'world', 'located', 'india', 'states', 'united', 'america', 'american',
                'city', 'south', 'country', 'earth', 'river', 'indian', 'england', 'north',
                'british', 'york', 'map', 'australia', 'canada', 'texas', 'capital', 'france',
                'island', 'china', 'africa', 'population', 'largest', 'sea', 'california', 'germany'
            ],
            'Sports & Competitions': [
                'won', 'cup', 'game', 'league', 'nba', 'win', 'football', 'games', 'team',
                'olympics', 'bowl', 'nfl', 'championship', 'baseball', 'player', 'wins',
                'basketball', 'scored', 'tournament', 'coach'
            ],
            'History & Government': [
                'war', 'president', 'state', 'national', 'battle', 'king', 'built',
                'government', 'law', 'court', 'constitution', 'act', 'rights', 'civil',
                'supreme', 'flag', 'minister', 'union', 'empire', 'bill', 'independence',
                'congress', 'federal', 'army', 'revolution', 'amendment', 'senate'
            ],
            'Science & Nature': [
                'body', 'water', 'air', 'system', 'human', 'anatomy', 'blood',
                'heart', 'cell', 'theory', 'sun', 'space', 'light', 'moon', 'cells', 'fire',
                'physics', 'chemistry', 'biology', 'scientist', 'equation', 'calculate', 'planet'
            ]
        }
        extract_domain._patterns = {
            domain: re.compile(r'\b(?:' + '|'.join(re.escape(kw) for kw in keywords) + r')\b', re.IGNORECASE)
            for domain, keywords in domains.items()
        }

    scores = Counter()
    for domain, pattern in extract_domain._patterns.items():
        matches = pattern.findall(str(question))
        if matches:
            scores[domain] = len(matches)

    if scores:
        return scores.most_common(1)[0][0]

    return 'General'

def process_and_chunk_data(df_base):
    cleaner = DataCleaner()
    # Assuming standard tqdm mapping
    df_base['long_answers_clean'] = df_base['long_answers'].apply(lambda x: cleaner.clean_text(x))
    df_base['short_answers_clean'] = df_base['short_answers'].apply(lambda x: cleaner.clean_text(x))

    df_base['question_type'] = df_base.apply(
        lambda row: categorize_question_type(row['question'], row['long_answers_clean']), axis=1
    )
    df_base['question_difficulty'] = df_base.apply(
        lambda row: estimate_difficulty(row['question'], row['long_answers_clean']), axis=1
    )
    df_base['question_domain'] = df_base.apply(
        lambda row: extract_domain(row['question'], row['long_answers_clean']), axis=1
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, length_function=len, is_separator_regex=False
    )
    
    all_chunks_data = []
    for index, row in df_base.iterrows():
        question = row['question']
        q_type = row['question_type']
        difficulty = row['question_difficulty']
        domain = row['question_domain']
        long_answer = row['long_answers_clean']
        short_answer = row['short_answers_clean']

        combined_text = f"[QUESTION]\n{question}\n-----\n[SHORT ANSWER]\n{short_answer}\n-----\n[LONG ANSWER]\n{long_answer}"
        custom_metadata = {
            "source_row_index": index,
            "question_type": q_type,
            "question_difficulty": difficulty,
            "question_domain": domain
        }

        doc = Document(page_content=combined_text, metadata=custom_metadata)
        chunk_docs = splitter.split_documents([doc])
        for i, chunk_doc in enumerate(chunk_docs):
            all_chunks_data.append({
                'chunk_id': f"{index}_{i}",
                'chunk_text': chunk_doc.page_content,
                'metadata': chunk_doc.metadata
            })

    cleaner.close()
    return pd.DataFrame(all_chunks_data), df_base
