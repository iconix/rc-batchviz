import argparse
import csv
from dataclasses import dataclass
import datasets
from datasets import load_dataset
import hashlib
import hdbscan
from keybert import KeyBERT
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import random
from scipy.spatial import ConvexHull
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, Generator, List, Optional
import umap.umap_ as umap

DEFAULT_SEED = 42
TEXT_MODEL_NAME = 'nomic-ai/nomic-embed-text-v1.5'


def set_global_seed(seed):
    """Set seed for reproducibility across all random number generators"""
    if not seed:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@dataclass
class ScriptConfig:
    """Configuration for embeddings processing and visualization"""
    input_file: str
    output: str
    preserve_names: List[str]
    pseudonymize: bool
    neighbors: int
    min_dist: float
    components: int
    metric: str
    gpu: bool
    topic_method: str

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'ScriptConfig':
        """Create Config from argparse namespace"""
        preserve_names = [name.strip() for name in args.preserve_names.split(',')] if args.preserve_names else []

        return cls(
            input_file=args.input_file,
            output=args.output,
            preserve_names=preserve_names,
            pseudonymize=args.pseudonymize,
            neighbors=args.neighbors,
            min_dist=args.min_dist,
            components=args.components,
            metric=args.metric,
            gpu=args.gpu,
            topic_method=args.topic_method
        )


class DataFields:
    # map from CSV field names to internal names
    MAPPING = {
        'Name': 'name',
        'Location': 'location',
        'Admission Pseudonym': 'admission_pseudonym',
        'Intro Call: Interests': 'intro_call_interests',
        'Intro Call: Background': 'intro_call_background',
        'Return Recurser': 'return_recurser',
        'Welcome Post': 'welcome_post'
    }
    # fields that are derived/generated later
    EXTRA = ['document', 'text_embedding']

    @classmethod
    def csv_names(cls):
        return list(cls.MAPPING.keys())

    @classmethod
    def dataset_names(cls):
        return list(cls.MAPPING.values()) + cls.EXTRA


class Pseudonymizer:
    """Handles consistent pseudonym generation for names"""
    def __init__(self, seed: int = DEFAULT_SEED):
        self.animals = ['Penguin', 'Platypus', 'Narwhal', 'Capybara', 'Axolotl',
                       'Wombat', 'Pangolin', 'Quokka', 'Lemur', 'Octopus', 'Owl']
        self.adjectives = ['Cosmic', 'Quantum', 'Dancing', 'Sleepy', 'Jazzy', 'Snazzy', 'Mystical',
                        'Fanciful', 'Wobbly', 'Zigzag', 'Whimsical', 'Ethereal', 'Flamboyant', 'Glowing',
                        'Breezy', 'Eccentric', 'Gleaming', 'Radiant', 'Twinkling', 'Magical', 'Celestial',
                        'Peculiar', 'Otherworldly', 'Curious', 'Enchanting', 'Dreamy', 'Charming',
                        'Exuberant', 'Spirited', 'Zany', 'Trippy', 'Dazzling', 'Surreal', 'Mysterious',
                        'Playful', 'Imaginative', 'Funky', 'Swaying', 'Sizzling', 'Bouncy', 'Luminous',
                        'Mellow', 'Jolly', 'Frolicking', 'Giddy', 'Delightful', 'Vibrant', 'Kaleidoscopic',
                        'Unconventional', 'Oddball', 'Lively', 'Whirlwind', 'Glimmering', 'Mischievous',
                        'Witty', 'Sparkling', 'Vivid', 'Shifting', 'Wiggly', 'Bizarre', 'Mellow', 'Zesty',
                        'Oddly', 'Jumpy', 'Groovy', 'Surprising', 'Ticklish', 'Frothy', 'Electric', 'Bubbly',
                        'Witty', 'Fuzzy', 'Sublime', 'Fantastical', 'Jumpy', 'Lively', 'Blissful', 'Breezy'
                    ]
        self.colors = ['AliceBlue', 'AntiqueWhite', 'Aquamarine', 'Azure', 'Bisque', 'BlanchedAlmond',
                       'BurlyWood', 'CadetBlue', 'Chartreuse', 'Chocolate', 'Coral', 'CornflowerBlue',
                       'Cornsilk', 'Crimson', 'Cyan', 'DarkOrchid', 'DimGray', 'DodgerBlue', 'FireBrick',
                       'FloralWhite', 'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'GoldenRod',
                       'HoneyDew', 'HotPink', 'Lavender', 'LemonChiffon', 'Lime', 'Linen', 'MidnightBlue',
                       'MintCream', 'MistyRose', 'Moccasin', 'OldLace', 'OliveDrab', 'Orchid', 'PapayaWhip',
                       'PeachPuff', 'Plum', 'PowderBlue', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Salmon',
                       'SeaGreen', 'SeaShell', 'Sienna', 'SkyBlue', 'SlateBlue', 'SlateGray', 'Snow', 'SteelBlue',
                       'Tan', 'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'WhiteSmoke']
        self.nature = ['Forest', 'Mountain', 'River', 'Ocean', 'Meadow',
                      'Crystal', 'Boulder', 'Cloud', 'Storm', 'Star']
        self.objects = ['Pickle', 'Robot', 'Teapot', 'Kazoo', 'Banjo',
                       'Compass', 'Cactus', 'Bicycle', 'Mushroom', 'Unicorn']
        self.name_map = {}

    def generate_whimsical_name(self) -> str:
        """Generate an whimsical name using custom word lists"""
        # different styles of absurd names - randomly pick one
        styles = [
            # color + nature
            lambda: f"{random.choice(self.colors)} {random.choice(self.nature)}",
            # adjective + animal
            lambda: f"{random.choice(self.adjectives)} {random.choice(self.animals)}",
            # adjective + object
            lambda: f"{random.choice(self.adjectives)} {random.choice(self.objects)}",
        ]

        return random.choice(styles)()

    def get_pseudonym(self, real_name: str, preserve_names: Optional[List[str]] = None) -> str:
        """Returns consistent pseudonym for a given real name

        Args:
            real_name: The real name to pseudonymize
            preserve_names: Optional list of names to preserve unchanged
        """
        if preserve_names and real_name in preserve_names:
            return real_name

        if real_name not in self.name_map:
            # use hash of name to ensure consistent seeds per name
            name_hash = int(hashlib.md5(real_name.encode()).hexdigest(), 16)
            random.seed(name_hash)
            self.name_map[real_name] = self.generate_whimsical_name()
            # go back to default seed after this process
            random.seed(DEFAULT_SEED)

        return self.name_map[real_name]


def read_csv_data(datafile: str) -> Generator[Dict, None, None]:
    """Read raw data from CSV file, yielding one row at a time"""
    with open(datafile, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=DataFields.csv_names())

        # skip header row
        next(reader, None)

        for row in reader:
            if not row['Location']:
                continue
            yield row


def create_document(row: dict) -> str:
    """Create document text from row data"""
    # if row['Intro Call: Background']:
    #     return f"---\ninterests: {row['Intro Call: Interests']}\nbackground: {row['Intro Call: Background']}\n---\n\n{row['Welcome Post']}\n"
    # else:
    return f"---\ninterests: {row['Intro Call: Interests']}\n---\n\n{row['Welcome Post']}\n"
    # return f"interests: {row['Intro Call: Interests']}"


def prepare_data(datafile: str, config: Optional[ScriptConfig] = None) -> list[dict]:
    """Prepare data from CSV, optionally with pseudonymization"""
    data = []
    pseudonymizer = Pseudonymizer() if config and config.pseudonymize else None

    for row in read_csv_data(datafile):
        # convert CSV field names to internal names
        processed = {DataFields.MAPPING[k]: v for k, v in row.items()}

        if pseudonymizer:
            processed['name'] = pseudonymizer.get_pseudonym(
                processed['name'],
                config.preserve_names if config else None
            )

        processed['document'] = create_document(row)  # note: uses original CSV field names
        data.append(processed)

    return data


class TextModel:
    def __init__(self, gpu: bool):
        self.tokenizer = AutoTokenizer.from_pretrained(
            TEXT_MODEL_NAME, model_max_length=8192
        )
        self.text_model = AutoModel.from_pretrained(
            TEXT_MODEL_NAME, trust_remote_code=True, rotary_scaling_factor=2
        )
        self.gpu = gpu

        if self.gpu:
            self.text_model.to('cuda')

        self.text_model.eval()

    def get_embeddings(self, text: str):
        encoded_input = self.tokenizer(
            text, padding=True, truncation=True, return_tensors='pt')

        if self.gpu:
            encoded_input = encoded_input.to('cuda')

        with torch.no_grad():
            model_output = self.text_model(**encoded_input)

        text_embeddings = TextModel.mean_pooling(
            model_output, encoded_input['attention_mask'])
        # TODO: "popular pipeline innovation in finding similar embeddings efficiently" - how? why?
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

        return text_embeddings

    @staticmethod
    def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor):
        # TODO: what layers is this averaging over, and why?
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(
                token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )


def get_hf_features(embed_len: int) -> datasets.Features:
    """Get dataset features schema"""
    features = {
        name: datasets.Value(dtype='string')
        for name in DataFields.dataset_names()
        if name != 'text_embedding'
    }
    features['text_embedding'] = datasets.Sequence(
        feature=datasets.Value(dtype='float32'),
        length=embed_len
    )
    return datasets.Features(features)

def build_hf_dataset(model: TextModel, data: list):
    rows = []
    embed_len = None

    for d in tqdm(data):
        embed = model.get_embeddings(f"clustering: {d['document']}")[0].cpu().detach().numpy()

        if embed_len is None:
            embed_len = embed.size

        d['text_embedding'] = embed
        rows.append(d)

    recurser_df = pd.DataFrame(rows, columns=DataFields.dataset_names())

    dataset = datasets.Dataset.from_pandas(recurser_df, features=get_hf_features(embed_len))
    dataset.set_format(
        type='numpy',
        columns=['text_embedding'],
        output_all_columns=True,
    )

    return dataset


def get_similar_names(dataset, name, embed_type='text'):
    """ given a name, sort by most similar names in dataset """
    assert embed_type in ['text', 'umap']

    embed_col = f"{embed_type}_embedding"

    embedding = dataset.filter(lambda recurser: recurser['name'] == name)[embed_col].squeeze()

    similarities = []
    for data in dataset:
        if embed_type == 'text':
            # cosine
            similarity = embedding @ data[embed_col]
        elif embed_type == 'umap':
            # euclidian distance
            a_min_b = embedding - data[embed_col]
            # similarity = np.sqrt(np.einsum("ij,ij->i", a_min_b, a_min_b))
            similarity = np.sqrt(np.sum(a_min_b**2))

        similarities.append((data['name'], similarity))

    similarities.sort(key=lambda x: x[1], reverse=True if embed_type == 'text' else False)

    return similarities


class TopicModeler:
    def __init__(self, docs, clusters):
        self.docs = docs
        self.clusters = clusters
        self.docs_df = pd.DataFrame({
            'Doc': docs,
            'Topic': clusters,
            'Doc_ID': range(len(docs))
        })
        self.docs_per_topic = self.docs_df.groupby(['Topic'], as_index=False).agg({'Doc': ' '.join})

    def tf_idf_topics(self, top_n=20, ngram_range=(1, 3)):
        # calculate c-TF-IDF
        count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(self.docs_per_topic.Doc.values)
        t = count.transform(self.docs_per_topic.Doc.values).toarray()
        w = t.sum(axis=1)
        tf = np.divide(t.T, w)
        sum_t = t.sum(axis=0)
        idf = np.log(np.divide(len(self.docs), sum_t)).reshape(-1, 1)
        tf_idf = np.multiply(tf, idf)

        # extract top words per topic
        words = count.get_feature_names_out()
        labels = list(self.docs_per_topic.Topic)
        tf_idf_transposed = tf_idf.T
        indices = tf_idf_transposed.argsort()[:, -top_n:]
        top_n_words = {
            label: [(words[j], float(tf_idf_transposed[i][j]))
                   for j in indices[i]][::-1]
            for i, label in enumerate(labels)
        }

        return top_n_words

    def keyword_bert_topics(self, top_n=20):
        """Use KeyBERT for contextual keyword extraction"""
        kw_model = KeyBERT(model='all-MiniLM-L6-v2')

        topics = {}
        for _, row in self.docs_per_topic.iterrows():
            keywords = kw_model.extract_keywords(
                row.Doc,
                keyphrase_ngram_range=(1, 3),
                stop_words='english',
                top_n=top_n,
                use_mmr=True,
                diversity=0.7
            )
            topics[row.Topic] = keywords

        return topics

    def extract_representative_sentences(self, top_n=3):
        """Extract most representative sentences per topic using embeddings"""
        model = SentenceTransformer('all-MiniLM-L6-v2')

        topics = {}
        for _, row in self.docs_per_topic.iterrows():
            # Split into sentences
            sentences = TextBlob(row.Doc).sentences
            if not sentences:
                continue

            # Get embeddings for all sentences
            sentence_embeddings = model.encode([str(s) for s in sentences])

            # Get centroid embedding
            centroid = np.mean(sentence_embeddings, axis=0)

            # Calculate distances to centroid
            distances = np.linalg.norm(sentence_embeddings - centroid, axis=1)

            # Get top N closest sentences
            top_indices = distances.argsort()[:top_n]
            topics[row.Topic] = [
                (str(sentences[i]), float(distances[i]))
                for i in top_indices
            ]

        return topics

    def get_topic_summaries(self, method='all', **kwargs):
        """Get topic summaries using multiple methods"""
        summaries = {}

        if method in ['tfidf', 'all']:
            summaries['tfidf'] = self.tf_idf_topics(**kwargs)

        if method in ['keybert', 'all']:
            summaries['keybert'] = self.keyword_bert_topics(**kwargs)

        if method in ['representative_sentences', 'all']:
            summaries['representative_sentences'] = self.extract_representative_sentences(**kwargs)

        return summaries if method == 'all' else summaries[method]


def compute_umap_clustering(data, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean',
                            use_hdbscan=True, hdbscan_dim_reduction=5,
                            hdbscan_min_cluster_size=2, hdbscan_min_samples=None, random_state=None):
    """Compute UMAP embeddings and HDBSCAN clustering without visualization"""

    # set global seed for reproducibility
    set_global_seed(random_state)

    umap_data = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components if not use_hdbscan else hdbscan_dim_reduction,
        metric=metric,
        random_state=random_state,  # fix UMAP seed
    ).fit_transform(data)

    init_umap = umap_data

    clusters = None
    if use_hdbscan:
        # perform clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        clusters = clusterer.fit_predict(umap_data)

        # further reduce to visualization dimensions if needed
        if hdbscan_dim_reduction != n_components:
            umap_data = umap.UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=min_dist,
                metric=metric,
                random_state=random_state,  # fix UMAP seed
            ).fit_transform(data)

    # create result dataframe
    columns = ['x', 'y', 'z'][:n_components]
    result = pd.DataFrame(umap_data, columns=columns)
    # note: to get back to a numpy array: np.array(df['umap_embedding'].tolist())
    result['umap_embedding'] = [coord for coord in init_umap]
    if clusters is not None:
        result['clusters'] = clusters

    return result


def create_interactive_umap(result, labels, topics=None, n_top_words=5, title='',
                            point_size=10, label_size=12, hover_width=140):
    """
    Create an interactive 2D or 3D UMAP visualization using Plotly.

    Args:
        result (pd.DataFrame): DataFrame containing UMAP coordinates ('x', 'y', and optionally 'z')
                                and cluster assignments
        labels (list): List of point labels
        topics (dict): Dictionary mapping cluster IDs to lists of top words
        n_top_words (int): Number of top words to display per topic
        title (str): Title for the visualization
        point_size (int): Size of the scatter points
        label_size (int): Size of the labels
        hover_width (int): Maximum width in characters for hover text wrapping

    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()

    # convert labels to numpy array for easier indexing
    labels = np.array(labels)
    is_3d = 'z' in result.columns

    def wrap_text(text, width):
        """Wrap text to specified width in characters"""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            word_length = len(word)
            if current_length + word_length + 1 <= width:
                current_line.append(word)
                current_length += word_length + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_length

        if current_line:
            lines.append(' '.join(current_line))

        return '<br>'.join(lines)

    if 'clusters' in result.columns:
        unique_clusters = np.unique(result.clusters)
        n_clusters = len(unique_clusters)
        # generate a color sequence
        colors = px.colors.qualitative.Set3[:n_clusters]

        # plot clustered points and their (2d-only) regions
        # (treating outliers as their own cluster)
        for idx, cluster_id in enumerate(unique_clusters):
            mask = result.clusters == cluster_id
            cluster_points = result[mask]
            cluster_labels = labels[mask]

            # create hover text
            hover_text = []
            for label in cluster_labels:
                if topics and cluster_id in topics:
                    # format topic words
                    if isinstance(topics[cluster_id], list):
                        topic_text = ', '.join(topics[cluster_id][:n_top_words])
                    else:
                        topic_text = str(topics[cluster_id])

                    # wrap topic text
                    wrapped_topics = wrap_text(topic_text, hover_width)

                    hover_text.append(
                        f"<b>Name:</b> {label}<br>" +
                        f"<b>Cluster:</b> {cluster_id + 1}<br>" +
                        f"<b>Topics:</b><br>{wrapped_topics}"
                    )
                else:
                    hover_text.append(
                        f"<b>Name:</b> {label}<br>" +
                        f"<b>Cluster:</b> {cluster_id + 1}"
                    )

            # create convex hull for 2D plots
            if not is_3d and len(cluster_points) >= 3:  # need at least 3 points for a hull
                try:
                    points = cluster_points[['x', 'y']].values
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices]

                    # close the polygon by repeating the first point
                    hull_points = np.vstack((hull_points, hull_points[0]))

                    # add shaded region
                    fig.add_trace(go.Scatter(
                        x=hull_points[:, 0],
                        y=hull_points[:, 1],
                        fill='toself',
                        mode='lines',
                        name=f"Cluster {cluster_id + 1} Region",
                        line=dict(width=0),
                        fillcolor=colors[idx],
                        opacity=0.2,
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                except:
                    pass  # skip if hull calculation fails

            # add scatter trace for this cluster
            scatter_kwargs = dict(
                mode='markers+text',
                name=f"Cluster {cluster_id + 1}",
                text=cluster_labels,
                textposition='top center',
                hovertext=hover_text,
                hovertemplate='%{hovertext}<extra></extra>',
                marker=dict(
                    size=point_size,
                    color=colors[idx] if idx != -1 else 'grey',
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                textfont=dict(
                    size=label_size,
                    color='rgba(0,0,0,0.5)'
                )
            )

            if is_3d:
                fig.add_trace(go.Scatter3d(
                    x=cluster_points.x,
                    y=cluster_points.y,
                    z=cluster_points.z,
                    **scatter_kwargs
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=cluster_points.x,
                    y=cluster_points.y,
                    **scatter_kwargs
                ))

    else:
        # if no clusters, plot all points with a single color
        scatter_kwargs = dict(
            mode='markers+text',
            name='Points',
            text=labels,
            textposition='top center',
            hovertext=[f"<b>Name:</b> {label}" for label in labels],
            hovertemplate='%{hovertext}<extra></extra>',
            marker=dict(
                size=point_size,
                color='blue',
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            textfont=dict(
                size=label_size,
                color='rgba(0,0,0,0.5)'
            )
        )

        if is_3d:
            fig.add_trace(go.Scatter3d(
                x=result.x,
                y=result.y,
                z=result.z,
                **scatter_kwargs
            ))
        else:
            fig.add_trace(go.Scatter(
                x=result.x,
                y=result.y,
                **scatter_kwargs
            ))

    # update layout
    layout_args = dict(
        title=dict(
            text=title,
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=20)
        ),
        showlegend=True,
        hovermode='closest',
        plot_bgcolor='rgb(240,240,240)',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(
            yanchor='top',
            y=0.99,
            xanchor='right',
            x=0.99,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )

    if is_3d:
        layout_args.update(
            scene=dict(
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False),
                zaxis=dict(showticklabels=False),
            )
        )
    else:
        layout_args.update(
            xaxis=dict(
                showgrid=True,
                gridcolor='white',
                gridwidth=1,
                zeroline=False,
                showticklabels=False
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='white',
                gridwidth=1,
                zeroline=False,
                showticklabels=False
            ),
        )

    fig.update_layout(**layout_args)

    return fig


def parse_arguments() -> ScriptConfig:
    parser = argparse.ArgumentParser(description='Visualize text embeddings data on my RC batch using UMAP, HDBSCAN, and topic modeling')

    parser.add_argument('input_file',
                       help='Input file path (.csv or .parquet)')
    parser.add_argument('--output', '-o',
                       help='Output file path for visualization (.html)',
                       default='interactive_viz.html')
    parser.add_argument('--preserve-names',
                       help='Comma-separated list of names to preserve (not pseudonymize)',
                       default='')
    parser.add_argument('--pseudonymize',
                       action='store_true',
                       help='Enable name pseudonymization')
    parser.add_argument('--neighbors', '-n',
                       type=int, default=2,
                       help='UMAP n_neighbors parameter')
    parser.add_argument('--min-dist', '-d',
                       type=float, default=0.99,
                       help='UMAP min_dist parameter')
    parser.add_argument('--components', '-c',
                       type=int, default=2,
                       help='Number of UMAP components')
    parser.add_argument('--metric', '-m',
                       type=str, default='cosine',
                       help='UMAP metric parameter')
    parser.add_argument('--gpu',
                       action='store_true',
                       help='Use GPU for text model if available')
    parser.add_argument('--topic-method', '-t',
                       type=str,
                       choices=['tfidf', 'keybert', 'representative_sentences'],
                       default='tfidf',
                       help='Topic modeling method to use')
    return ScriptConfig.from_args(parser.parse_args())


def process_data(config: ScriptConfig) -> datasets.Dataset:
    """Load and process data according to config settings"""
    if config.input_file.endswith('.csv'):
        recursers = prepare_data(config.input_file, config)

        ## for r in recursers[:3]:
        ##     print(r['document'])

        # generate embeddings
        model = TextModel(gpu=config.gpu)
        dataset = build_hf_dataset(model, recursers)
        dataset.to_parquet(f"{config.input_file.replace('.csv', '')}.parquet", compression='gzip')
    elif config.input_file.endswith('.parquet'):
        dataset = load_dataset('parquet', data_files=config.input_file)['train']

        # apply pseudonymization to loaded dataset if requested
        if config.pseudonymize:
            pseudonymizer = Pseudonymizer()
            dataset = dataset.map(lambda x: {
                'name': pseudonymizer.get_pseudonym(x['name'], config.preserve_names)
            })
    else:
        raise ValueError("Input file must be .csv or .parquet")

    return dataset


def main():
    config = parse_arguments()

    # process data according to config
    dataset = process_data(config)

    embeddings = dataset['text_embedding']
    names = dataset['name']

    ## print(get_similar_names(dataset, 'Nadja Rhodes', 'text'))
    ## print(get_similar_names(dataset, 'Nadja Rhodes', 'umap'))

    result = compute_umap_clustering(
        embeddings,
        n_neighbors=config.neighbors,
        min_dist=config.min_dist,
        n_components=config.components,
        metric=config.metric,
        use_hdbscan=True,
        hdbscan_dim_reduction=config.components,  # 5
        hdbscan_min_cluster_size=2,
        random_state=DEFAULT_SEED,
    )

    # fig = create_interactive_umap(
    #     result,
    #     names,
    #     title='Interactive UMAP clustered with HDBSCAN'
    # )
    # fig.show()

    # generate topic models using the clusters
    modeler = TopicModeler(dataset['document'], result['clusters'])
    summaries = modeler.get_topic_summaries(method=config.topic_method)
    # available: tfidf, keybert, representative_sentences
    if config.topic_method == 'representative_sentences':
        topics = {k: [sent for sent, score in v] for k, v in summaries.items()}
    else:
        topics = {k: [word for word, weight in v] for k, v in summaries.items()}

    # visualize again with topics
    fig = create_interactive_umap(
        result,
        names,
        topics=topics,
        title=f"batches of batch: a UMAP Projection with HDBSCAN Clustering and `{config.topic_method}` Topics"
    )

    # save as HTML file
    fig.write_html(config.output)
    print(f"visualization saved to {config.output}")

    # display in browser
    fig.show()

    # TODO:
    #   - add sliders to html output?


if __name__ == '__main__':
    main()
