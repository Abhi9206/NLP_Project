import os
import gc
import spacy
import torch
import torch.nn as nn
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.data import Data, Batch
from sentence_transformers import SentenceTransformer
import evaluation_metric as eval_model
import pandas as pd


#--------------------------------------------------------------------------------------------------------------------
# Class StoryGraphBuilder
#--------------------------------------------------------------------------------------------------------------------
#%%



class StoryGraphBuilder:
    """
    Complete graph construction pipeline for keyword-constrained story generation.
    Handles entity extraction, graph building, feature creation, and dataset processing.
    """

    # Node type constants
    NODE_TYPES = {
        'ROOT': 0,
        'ENTITY': 1,
        'KEYWORD': 2,
        'PAD': 3
    }

    # Edge type constants
    EDGE_TYPES = {
        'CONSTRAINT': 0,
        'RELATION': 1,
        'SEQUENCE': 2
    }

    def __init__(self,
                 spacy_model='en_core_web_sm',
                 bert_model='all-MiniLM-L6-v2',
                 embedding_dim=384,
                 target_nodes=15,
                 max_entities=10):
        """
        Initialize the graph builder with models and parameters.

        Args:
            spacy_model: SpaCy model name for entity extraction
            bert_model: Sentence transformer model for embeddings
            embedding_dim: Dimension of BERT embeddings
            target_nodes: Fixed size for all graphs (padding applied)
            max_entities: Maximum entities to extract from partial sentence
        """
        self.embedding_dim = embedding_dim
        self.target_nodes = target_nodes
        self.max_entities = max_entities

        # Load models
        print("Loading models...")
        self.nlp = spacy.load(spacy_model)
        self.bert_model = SentenceTransformer(bert_model)
        print("Models loaded successfully!")

    def extract_entities(self, partial_sentence):
        """
        Extract entities from partial sentence using SpaCy POS tagging.

        Args:
            partial_sentence: First few words of the story

        Returns:
            List of unique entity strings
        """
        doc = self.nlp(partial_sentence)
        entities = []

        for token in doc:
            if token.is_punct or token.is_stop:
                continue
            if token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ']:
                entities.append(token.text.lower())

        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)

        return unique_entities[:self.max_entities]

    def build_graph(self, partial_sentence, keywords):
        """
        Build NetworkX graph with fixed number of nodes.

        Args:
            partial_sentence: Beginning of the story
            keywords: List of required keywords

        Returns:
            G: NetworkX DiGraph
            node_mapping: Dict mapping text to node_id
        """
        G = nx.DiGraph()
        node_id = 0
        node_mapping = {}

        # 1. Add ROOT node
        G.add_node(node_id,
                   text='ROOT',
                   node_type=self.NODE_TYPES['ROOT'],
                   is_constraint=False)
        root_id = node_id
        node_mapping['ROOT'] = node_id
        node_id += 1

        # 2. Add ENTITY nodes
        entities = self.extract_entities(partial_sentence)
        entity_ids = []

        for entity in entities:
            G.add_node(node_id,
                       text=entity,
                       node_type=self.NODE_TYPES['ENTITY'],
                       is_constraint=False)
            node_mapping[entity] = node_id
            entity_ids.append(node_id)
            node_id += 1

        # 3. Add KEYWORD nodes (check for duplicates with entities)
        keyword_ids = []

        for keyword in keywords:
            keyword_lower = keyword.lower()

            if keyword_lower in node_mapping:
                # Mark existing entity as constraint
                existing_id = node_mapping[keyword_lower]
                G.nodes[existing_id]['is_constraint'] = True
                keyword_ids.append(existing_id)
            else:
                # Add new keyword node
                G.add_node(node_id,
                           text=keyword_lower,
                           node_type=self.NODE_TYPES['KEYWORD'],
                           is_constraint=True)
                node_mapping[keyword_lower] = node_id
                keyword_ids.append(node_id)
                node_id += 1

        # 4. Pad to target size
        while node_id < self.target_nodes:
            G.add_node(node_id,
                       text='PAD',
                       node_type=self.NODE_TYPES['PAD'],
                       is_constraint=False)
            node_id += 1

        # 5. Add edges
        # ROOT → keywords (CONSTRAINT)
        for kw_id in keyword_ids:
            G.add_edge(root_id, kw_id, edge_type=self.EDGE_TYPES['CONSTRAINT'])

        # Entities ↔ Keywords (RELATION, bidirectional)
        for ent_id in entity_ids:
            for kw_id in keyword_ids:
                G.add_edge(ent_id, kw_id, edge_type=self.EDGE_TYPES['RELATION'])
                G.add_edge(kw_id, ent_id, edge_type=self.EDGE_TYPES['RELATION'])

        # Entities in sequence (SEQUENCE)
        for i in range(len(entity_ids) - 1):
            G.add_edge(entity_ids[i], entity_ids[i + 1],
                       edge_type=self.EDGE_TYPES['SEQUENCE'])

        # ROOT → entities (RELATION)
        for ent_id in entity_ids:
            G.add_edge(root_id, ent_id, edge_type=self.EDGE_TYPES['RELATION'])

        return G, node_mapping

    def create_node_features(self, G):
        """
        Create feature matrix for all nodes using BERT embeddings + metadata.

        Args:
            G: NetworkX graph

        Returns:
            Tensor of shape [num_nodes, feature_dim]
        """
        num_nodes = G.number_of_nodes()

        # Feature dimensions: BERT(384) + node_type(4) + is_constraint(1) + is_root(1) = 390
        feature_dim = self.embedding_dim + 4 + 1 + 1
        node_features = np.zeros((num_nodes, feature_dim))

        # Batch encode all node texts
        node_texts = []
        for node_id in range(num_nodes):
            text = G.nodes[node_id]['text']
            node_texts.append('' if text == 'PAD' else text)

        embeddings = self.bert_model.encode(node_texts, show_progress_bar=False)

        # Fill feature matrix
        for node_id in range(num_nodes):
            idx = 0

            # BERT embedding (384 dims)
            node_features[node_id, idx:idx + self.embedding_dim] = embeddings[node_id]
            idx += self.embedding_dim

            # Node type (one-hot, 4 dims)
            node_type = G.nodes[node_id]['node_type']
            node_features[node_id, idx + node_type] = 1.0
            idx += 4

            # Is constraint (1 dim)
            node_features[node_id, idx] = float(G.nodes[node_id]['is_constraint'])
            idx += 1

            # Is root (1 dim)
            node_features[node_id, idx] = 1.0 if node_type == self.NODE_TYPES['ROOT'] else 0.0

        return torch.FloatTensor(node_features)

    def create_edge_index(self, G):
        """
        Create edge index and attributes for PyTorch Geometric.

        Args:
            G: NetworkX graph

        Returns:
            edge_index: Tensor [2, num_edges]
            edge_attr: Tensor [num_edges, 1]
        """
        edges = list(G.edges())
        if len(edges) == 0:
            return (torch.zeros((2, 0), dtype=torch.long),
                    torch.zeros((0, 1), dtype=torch.long))

        # Edge connectivity
        edge_index = torch.tensor([[e[0] for e in edges],
                                   [e[1] for e in edges]], dtype=torch.long)

        # Edge types
        edge_types = [G[u][v]['edge_type'] for u, v in edges]
        edge_attr = torch.tensor(edge_types, dtype=torch.long).unsqueeze(1)

        return edge_index, edge_attr

    def create_pyg_graph(self, partial_sentence, keywords,
                         target_story=None, target_tokens=None):
        """
        Complete pipeline: create PyTorch Geometric Data object.

        Args:
            partial_sentence: Beginning of story
            keywords: List of required keywords
            target_story: Full story text (optional)
            target_tokens: Tokenized target (optional)

        Returns:
            PyTorch Geometric Data object
        """
        # Build graph structure
        G, node_map = self.build_graph(partial_sentence, keywords)

        # Create features
        x = self.create_node_features(G)
        edge_index, edge_attr = self.create_edge_index(G)

        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=G.number_of_nodes()
        )

        # Add metadata
        data.partial_sentence = partial_sentence
        data.keywords = keywords

        if target_story is not None:
            data.target_story = target_story

        if target_tokens is not None:
            data.target_tokens = torch.LongTensor(target_tokens)

        return data

    def process_dataset(self, tokenized_data, split_name, output_dir='graph_data'):
        """
        Process entire dataset and save PyG graphs.

        Args:
            tokenized_data: List of tokenized samples
            split_name: 'train', 'val', or 'test'
            output_dir: Directory to save processed graphs

        Returns:
            List of PyG Data objects
        """
        print(f"\n{'=' * 60}")
        print(f"Processing {split_name} dataset")
        print(f"{'=' * 60}")
        print(f"Total samples: {len(tokenized_data)}")

        graph_list = []
        failed_count = 0

        for idx, sample in enumerate(tqdm(tokenized_data,
                                          desc=f"Building {split_name} graphs")):
            try:
                # Extract data
                partial = sample['input_text'].split('Story: ')[1] \
                    if 'Story: ' in sample['input_text'] \
                    else sample['input_text']
                keywords = sample['keywords']
                target_story = sample['target_text']
                target_tokens = sample['target_ids']

                # Create graph
                data = self.create_pyg_graph(
                    partial_sentence=partial,
                    keywords=keywords,
                    target_story=target_story,
                    target_tokens=target_tokens
                )

                data.sample_id = idx
                graph_list.append(data)

            except Exception as e:
                failed_count += 1
                if failed_count <= 5:
                    print(f"\nWarning: Failed to process sample {idx}: {str(e)}")

        print(f"\nProcessing complete!")
        print(f"  Success: {len(graph_list)}")
        print(f"  Failed: {failed_count}")

        # Save graphs
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{split_name}_graphs.pt')
        torch.save(graph_list, output_path)
        print(f"  Saved to: {output_path}")

        return graph_list

    def visualize_graph(self, G, title="Story Graph", figsize=(14, 10)):
        """
        Create a visualization of the story graph.

        Args:
            G: NetworkX graph
            title: Plot title
            figsize: Figure size tuple

        Returns:
            matplotlib figure
        """
        plt.figure(figsize=figsize)

        # Color mapping
        color_map = {
            self.NODE_TYPES['ROOT']: '#FF6B6B',
            self.NODE_TYPES['ENTITY']: '#4ECDC4',
            self.NODE_TYPES['KEYWORD']: '#FFD93D',
            self.NODE_TYPES['PAD']: '#E8E8E8'
        }

        node_colors = [color_map[G.nodes[node]['node_type']] for node in G.nodes()]

        # Labels (skip PAD nodes)
        labels = {}
        for node in G.nodes():
            if G.nodes[node]['text'] != 'PAD':
                text = G.nodes[node]['text']
                labels[node] = f"*{text}*" if G.nodes[node]['is_constraint'] else text

        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                               node_size=1500, alpha=0.9)

        # Draw edges with different styles
        edge_colors = []
        edge_styles = []
        for u, v in G.edges():
            edge_type = G[u][v]['edge_type']
            if edge_type == self.EDGE_TYPES['CONSTRAINT']:
                edge_colors.append('red')
                edge_styles.append('solid')
            elif edge_type == self.EDGE_TYPES['SEQUENCE']:
                edge_colors.append('blue')
                edge_styles.append('dashed')
            else:
                edge_colors.append('gray')
                edge_styles.append('dotted')

        nx.draw_networkx_edges(G, pos, edge_color=edge_colors,
                               style=edge_styles, alpha=0.5,
                               arrows=True, arrowsize=20, width=2)

        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')

        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B',
                       markersize=12, label='ROOT'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ECDC4',
                       markersize=12, label='ENTITY'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFD93D',
                       markersize=12, label='KEYWORD (*=constraint)'),
            plt.Line2D([0], [0], color='red', linewidth=2, label='CONSTRAINT edge'),
            plt.Line2D([0], [0], color='blue', linewidth=2, linestyle='--',
                       label='SEQUENCE edge'),
            plt.Line2D([0], [0], color='gray', linewidth=2, linestyle=':',
                       label='RELATION edge')
        ]
        plt.legend(handles=legend_elements, loc='upper left', fontsize=10)

        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()

        return plt





#--------------------------------------------------------------------------------------------------------------------
# Training and Evaluation Utilities
#--------------------------------------------------------------------------------------------------------------------

def collate_fn(batch):
    batched = Batch.from_data_list(batch)
    # Don't move here - let it happen in compute_loss
    return batched


def compute_loss(model, batch, device):
    """Compute loss with keyword penalty"""
    batch = batch.to(device)
    if hasattr(batch, 'x'):
        batch.x = batch.x.to(device)
    if hasattr(batch, 'edge_index'):
        batch.edge_index = batch.edge_index.to(device)

    batch_size = batch.num_graphs
    total_loss = 0.0
    keyword_penalty = 0.0

    for i in range(batch_size):
        # Extract graph
        node_mask = (batch.batch == i)
        graph_x = batch.x[node_mask].to(device)

        edge_mask = (batch.batch[batch.edge_index[0]] == i)
        graph_edge_index = batch.edge_index[:, edge_mask].to(device)

        if graph_edge_index.shape[1] == 0:
            continue

        # Remap edges
        unique_nodes = graph_edge_index.unique()
        node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(unique_nodes)}
        remapped_edges = torch.tensor(
            [[node_mapping[n.item()] for n in graph_edge_index[0]],
             [node_mapping[n.item()] for n in graph_edge_index[1]]],
            device=device
        )

        # Get data
        target_tokens = batch[i].target_tokens.to(device)
        partial_text = batch[i].partial_sentence
        keywords = batch[i].keywords

        # GNN encode
        graph_embedding = model.gnn(graph_x, remapped_edges)
        if torch.isnan(graph_embedding).any():
            continue

        graph_context = model.fusion(graph_embedding)
        if torch.isnan(graph_context).any():
            continue

        # Prepare input with KEYWORDS
        if keywords and len(keywords) > 0:
            keywords_str = ", ".join(keywords)
            input_text = f"REQUIRED: {keywords_str}\nStory: {partial_text}"
        else:
            input_text = f"Story: {partial_text}"

        inputs = model.tokenizer(
            input_text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)

        # Get embeddings
        if hasattr(model.slm, 'model') and hasattr(model.slm.model, 'embed_tokens'):
            input_embeds = model.slm.model.embed_tokens(inputs['input_ids'])
        else:
            input_embeds = model.slm.get_input_embeddings()(inputs['input_ids'])

        # Match dtype
        graph_context = graph_context.to(input_embeds.dtype).to(device)
        graph_context = graph_context.view(1, model.n_prefix_tokens, -1)  # [1, 8, 4096]

        # Prepare target
        max_target_len = 80
        if target_tokens.shape[0] > max_target_len:
            target_tokens = target_tokens[:max_target_len]

        # Get target embeddings
        if hasattr(model.slm, 'model') and hasattr(model.slm.model, 'embed_tokens'):
            target_embeds = model.slm.model.embed_tokens(target_tokens.unsqueeze(0))
        else:
            target_embeds = model.slm.get_input_embeddings()(target_tokens.unsqueeze(0))

        # Combine: [graph_context | input_embeds | target_embeds]
        combined_embeds = torch.cat([graph_context, input_embeds, target_embeds], dim=1)

        # Create attention mask
        input_attention = inputs.get('attention_mask', torch.ones_like(inputs['input_ids']))
        graph_attention = torch.ones((1, model.n_prefix_tokens), device=device, dtype=input_attention.dtype)
        target_attention = torch.ones((1, target_tokens.shape[0]), device=device, dtype=input_attention.dtype)
        combined_attention = torch.cat([graph_attention, input_attention, target_attention], dim=1)

        # Create labels: [-100 for graph and input] + [target tokens]
        num_prefix = model.n_prefix_tokens + inputs['input_ids'].shape[1]
        labels = torch.cat([
            torch.full((1, num_prefix), -100, dtype=torch.long, device=device),
            target_tokens.unsqueeze(0)
        ], dim=1)

        # Truncate if too long
        max_seq_len = 256
        if combined_embeds.shape[1] > max_seq_len:
            combined_embeds = combined_embeds[:, :max_seq_len, :]
            combined_attention = combined_attention[:, :max_seq_len]
            labels = labels[:, :max_seq_len]

        # Forward pass
        try:
            outputs = model.slm(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention,
                labels=labels
            )
            lm_loss = outputs.loss

            if torch.isnan(lm_loss) or torch.isinf(lm_loss):
                continue

        except Exception as e:
            continue

        total_loss += lm_loss

        # Keyword penalty
        if keywords and len(keywords) > 0:
            target_text = model.tokenizer.decode(target_tokens, skip_special_tokens=True).lower()

            for keyword in keywords:
                keyword_lower = keyword.lower().strip()
                variants = [keyword_lower, keyword_lower + 's', keyword_lower + 'ed', keyword_lower + 'ing']
                found = any(v in target_text for v in variants)

                if not found:
                    keyword_penalty += 10.0

    if batch_size == 0:
        return None

    avg_loss = total_loss / batch_size
    avg_penalty = keyword_penalty / batch_size
    combined_loss = avg_loss + 10.0 * avg_penalty

    return combined_loss




def train_epoch(model, dataloader, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0.0
    valid_batches = 0
    skip_reasons = {'loss_invalid': 0, 'grad_invalid': 0}  # ADD THIS

    pbar = tqdm(dataloader, desc="Training")

    for batch_idx, batch in enumerate(pbar):

        loss = compute_loss(model, batch, device)

        # Skip bad batch
        if loss is None:
            skip_reasons['loss_invalid'] += 1
            pbar.write(f"[batch {batch_idx}] skipping invalid batch - loss: None")
            continue

        # Ensure loss is a tensor
        if not isinstance(loss, torch.Tensor):
            continue

        if torch.isnan(loss) or torch.isinf(loss):
            skip_reasons['loss_invalid'] += 1
            pbar.write(f"[batch {batch_idx}] skipping invalid batch - loss: {loss.item()}")
            continue

        optimizer.zero_grad()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        if not torch.isfinite(grad_norm):
            skip_reasons['grad_invalid'] += 1  # ADD THIS
            pbar.write(f"[batch {batch_idx}] skipping due to non-finite grad_norm: {grad_norm}")  # ENHANCED
            optimizer.zero_grad()
            continue

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_value = float(loss.item())
        total_loss += loss_value
        valid_batches += 1

        lr = optimizer.param_groups[0]['lr']

        pbar.set_postfix(
            loss=f"{loss_value:.4f}",
            grad_norm=f"{grad_norm:.2f}",
            lr=f"{lr:.2e}"
        )

    # ADD THIS AT END:
    total_batches = len(dataloader)
    skipped_batches = total_batches - valid_batches
    skip_rate = (skipped_batches / total_batches) * 100

    print(f"\n📊 Batch Statistics:")
    print(f"  Total batches: {total_batches}")
    print(f"  Valid batches: {valid_batches}")
    print(f"  Skipped batches: {skipped_batches} ({skip_rate:.2f}%)")
    print(f"    - Due to invalid loss: {skip_reasons['loss_invalid']}")
    print(f"    - Due to invalid gradients: {skip_reasons['grad_invalid']}")

    return total_loss / max(1, valid_batches)


def generate_story(model, graph_data, device, max_new_tokens=100):
    """Generate story from graph data using the model"""
    model.eval()

    with torch.no_grad():
        # Get single graph data
        graph_x = graph_data.x.to(device)
        graph_edge_index = graph_data.edge_index.to(device)
        partial_text = graph_data.partial_sentence
        keywords = graph_data.keywords if hasattr(graph_data, 'keywords') else None

        # 1. Encode graph with GNN
        graph_embedding = model.gnn(graph_x, graph_edge_index)

        # 2. Project to LM embedding space - ADD THIS
        graph_context = model.fusion(graph_embedding)

        # 3. Prepare input for LM WITH KEYWORDS
        if keywords and len(keywords) > 0:
            keywords_str = ", ".join(keywords)
            input_text = f"YOU MUST USE THESE WORDS: {keywords_str}\n\nStory beginning: {partial_text}\n\nContinue the story and USE ALL THE REQUIRED WORDS:"
        else:
            input_text = f"Story: {partial_text}"

        inputs = model.tokenizer(input_text, return_tensors='pt', padding=True).to(device)

        # Get input embeddings
        if hasattr(model.slm, 'model') and hasattr(model.slm.model, 'embed_tokens'):
            input_embeds = model.slm.model.embed_tokens(inputs['input_ids'])
        elif hasattr(model.slm, 'transformer') and hasattr(model.slm.transformer, 'wte'):
            input_embeds = model.slm.transformer.wte(inputs['input_ids'])
        else:
            input_embeds = model.slm.get_input_embeddings()(inputs['input_ids'])

        # Match dtype and reshape
        graph_context = graph_context.to(input_embeds.dtype).to(device)
        graph_context = graph_context.view(1, model.n_prefix_tokens, -1)

        # Combine embeddings
        combined_embeds = torch.cat([graph_context, input_embeds], dim=1)

        # Create attention masks - FIXED ORDER
        original_attention = inputs.get('attention_mask', torch.ones_like(inputs['input_ids']))
        graph_attention = torch.ones((1, model.n_prefix_tokens), device=device, dtype=original_attention.dtype)
        combined_attention = torch.cat([graph_attention, original_attention], dim=1)

        # Generate
        output_ids = model.slm.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.85,
            do_sample=True,
            pad_token_id=model.tokenizer.eos_token_id,
            eos_token_id=model.tokenizer.eos_token_id
        )

        # Decode
        generated_text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Remove input prompt
        if input_text in generated_text:
            generated_text = generated_text[len(input_text):].strip()

        return generated_text



def evaluate_with_metrics(model, dataloader, device, num_samples=50):
    """Evaluate model with comprehensive metrics"""
    model.eval()

    generated_stories = []
    reference_stories = []
    keywords_list = []

    sample_count = 0

    with torch.no_grad():
        for batch in dataloader:
            if sample_count >= num_samples:
                break

            batch = batch.to(device)
            batch_size = batch.num_graphs

            for i in range(min(batch_size, num_samples - sample_count)):
                # Extract single graph
                node_mask = (batch.batch == i)
                graph_x = batch.x[node_mask].to(device)

                # Get edges for this graph
                edge_mask = (batch.batch[batch.edge_index[0]] == i)
                graph_edge_index = batch.edge_index[:, edge_mask]

                # Remap node indices
                unique_nodes = graph_edge_index.unique()
                node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(unique_nodes)}
                remapped_edges = torch.tensor([[node_mapping[n.item()] for n in graph_edge_index[0]],
                                             [node_mapping[n.item()] for n in graph_edge_index[1]]],
                                            device=device)

                # Create single graph data
                single_graph = Data(x=graph_x, edge_index=remapped_edges)
                single_graph.partial_sentence = batch[i].partial_sentence
                single_graph.keywords = batch[i].keywords

                # Generate story
                generated_story = generate_story(model, single_graph, device)
                reference_story = batch[i].target_story
                keywords = batch[i].keywords

                generated_stories.append(generated_story)
                reference_stories.append(reference_story)
                keywords_list.append(keywords)

                sample_count += 1

            if sample_count >= num_samples:
                break

    # Create evaluation dataframe
    eval_df = pd.DataFrame({
        'id': range(len(generated_stories)),
        'story': reference_stories,
        'generated_story': generated_stories,
        'words': keywords_list
    })

    # Compute comprehensive evaluation metrics using evaluation_metric functions
    print("  Computing BLEU scores...")
    corpus_bleu, _ = eval_model.compute_bleu_scores(eval_df, add_column=True)

    print("  Computing Perplexity...")
    per_text_ppl, mean_ppl = eval_model.compute_perplexity(generated_stories)
    eval_df["perplexity"] = per_text_ppl

    print("  Computing BERTScore...")
    bert_p, bert_r, bert_f1 = eval_model.compute_bert_scores(eval_df, add_columns=True)

    print("  Computing ROUGE scores...")
    rouge1, rouge2, rougeL = eval_model.compute_rouge_scores(eval_df, add_columns=True)

    print("  Computing Keyword metrics...")
    # Since each story has different keywords, process individually and aggregate
    all_strict_accuracies = []
    all_avg_percentages = []

    for i, (generated_story, keywords) in enumerate(zip(generated_stories, keywords_list)):
        if not keywords:
            all_strict_accuracies.append(0.0)
            all_avg_percentages.append(0.0)
            continue

        # Create single-row dataframe for this story
        single_story_df = pd.DataFrame({'generated_story': [generated_story]})

        # Use evaluation_metric function
        strict_acc, avg_pct = eval_model.compute_keyword_metrics(
            df=single_story_df,
            hyp_col='generated_story',
            keywords=keywords,
            add_columns=False,
            case_sensitive=False
        )

        all_strict_accuracies.append(strict_acc)
        all_avg_percentages.append(avg_pct)

    # Calculate overall keyword metrics
    keyword_strict_accuracy = sum(all_strict_accuracies) / len(all_strict_accuracies) * 100.0 if all_strict_accuracies else 0.0
    keyword_average_percentage = sum(all_avg_percentages) / len(all_avg_percentages) if all_avg_percentages else 0.0

    # Create comprehensive metrics summary
    metrics_summary = {
        'BLEU': corpus_bleu,
        'Perplexity': mean_ppl,
        'BERTScore_Precision': bert_p,
        'BERTScore_Recall': bert_r,
        'BERTScore_F1': bert_f1,
        'ROUGE-1': rouge1,
        'ROUGE-2': rouge2,
        'ROUGE-L': rougeL,
        'Keyword_Strict_Accuracy': keyword_strict_accuracy,
        'Keyword_Average_Percentage': keyword_average_percentage
    }

    return {
        'metrics_summary': metrics_summary,
        'eval_df': eval_df
    }


def validate(model, dataloader, device):
    """Validation pass"""
    model.eval()
    total_loss = 0
    valid_batches = 0  # ADD THIS

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating")

        for batch in pbar:
            try:
                loss = compute_loss(model, batch, device)

                # ADD THIS CHECK:
                if loss is None or torch.isnan(loss) or torch.isinf(loss):
                    continue

                total_loss += loss.item()
                valid_batches += 1  # ADD THIS
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            except Exception as e:
                continue

    return total_loss / max(1, valid_batches)


def display_metrics_table(metrics):
    """Display evaluation metrics in a formatted table"""
    print("\nEVALUATION RESULTS:")
    print("-" * 50)
    print(f"{'Metric':<25} {'Value':<15}")
    print("-" * 50)
    print(f"{'BLEU Score':<25} {metrics['BLEU']:<15.4f}")
    print(f"{'Perplexity':<25} {metrics['Perplexity']:<15.2f}")
    print(f"{'BERTScore Precision':<25} {metrics['BERTScore_Precision']:<15.4f}")
    print(f"{'BERTScore Recall':<25} {metrics['BERTScore_Recall']:<15.4f}")
    print(f"{'BERTScore F1':<25} {metrics['BERTScore_F1']:<15.4f}")
    print(f"{'ROUGE-1':<25} {metrics['ROUGE-1']:<15.4f}")
    print(f"{'ROUGE-2':<25} {metrics['ROUGE-2']:<15.4f}")
    print(f"{'ROUGE-L':<25} {metrics['ROUGE-L']:<15.4f}")
    print(f"{'Keyword Strict Acc.':<25} {metrics['Keyword_Strict_Accuracy']:<15.2f}%")
    print(f"{'Keyword Avg. Pct.':<25} {metrics['Keyword_Average_Percentage']:<15.2f}%")
    print("-" * 50)


def save_evaluation_results(results, model_directory, epoch):
    """Save evaluation results and metrics to CSV files"""
    # Save evaluation results
    eval_save_path = f'{model_directory}/checkpoints/eval_epoch_{epoch + 1}.csv'
    results['eval_df'].to_csv(eval_save_path, index=False)

    # Save metrics summary
    metrics_save_path = f'{model_directory}/checkpoints/metrics_epoch_{epoch + 1}.csv'
    metrics_df = pd.DataFrame([results['metrics_summary']])
    metrics_df.to_csv(metrics_save_path, index=False)

    print(f"\nDetailed results saved to: {eval_save_path}")
    print(f"Metrics summary saved to: {metrics_save_path}")

    return eval_save_path, metrics_save_path



