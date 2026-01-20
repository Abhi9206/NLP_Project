import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import os


class GNNRetrievalSLM(nn.Module):
    """
    Complete architecture: GNN + Retrieval + Small Language Model
    """

    def __init__(self,
                 gnn_hidden_dim=256,
                 slm_name='gpt2',
                 retriever=None,
                 hf_token=None):
        super().__init__()

        # 1. GNN Encoder
        from .gnn_encoder import GraphEncoder
        self.gnn = GraphEncoder(
            node_feature_dim=390,
            hidden_dim=gnn_hidden_dim,
            num_layers=3,
            num_heads=4
        )

        # 2. Determine dtype (use float16/bfloat16 for efficiency)
        # GPT-2 works well with float32, but bfloat16 saves memory
        if 'gpt2' in slm_name.lower():
            dtype = torch.float32  # GPT-2 small enough for FP32
        else:
            dtype = torch.bfloat16  # Larger models use bfloat16

        # 3. Load Language Model
        print(f"Loading {slm_name}...")
        load_kwargs = {
            'torch_dtype': dtype,
            'device_map': "auto",
            'trust_remote_code': True,
        }

        # Only add token if provided (not needed for public models like GPT-2)
        if hf_token is not None:
            load_kwargs['token'] = hf_token

        self.slm = AutoModelForCausalLM.from_pretrained(
            slm_name,
            **load_kwargs
        )
        print(f"✓ Loaded {slm_name}")

        # 4. Load Tokenizer
        print(f"Loading tokenizer for {slm_name}...")
        try:
            tokenizer_kwargs = {'trust_remote_code': True}
            if hf_token is not None:
                tokenizer_kwargs['token'] = hf_token

            self.tokenizer = AutoTokenizer.from_pretrained(
                slm_name,
                **tokenizer_kwargs
            )
            print(f"✓ Loaded tokenizer")
        except Exception as e:
            print(f"Warning: Could not load tokenizer for {slm_name}, falling back to GPT2: {e}")
            from transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 5. Define fusion layer (projects GNN output to LM embedding space)
        self.n_prefix_tokens = 16
        self.fusion = nn.Linear(
            gnn_hidden_dim,
            self.n_prefix_tokens * self.slm.config.hidden_size
        )

        # 6. Store retriever
        self.retriever = retriever

        # Print model info
        print(f"\n✓ Model initialized successfully!")
        print(f"  - GNN hidden dim: {gnn_hidden_dim}")
        print(f"  - SLM: {slm_name}")
        print(f"  - SLM hidden size: {self.slm.config.hidden_size}")
        print(f"  - Prefix tokens: {self.n_prefix_tokens}")
        print(f"  - Fusion output size: {self.fusion.out_features}")
        print(f"  - Dtype: {dtype}")

    def forward(self, graph_data, partial_text, keywords=None, max_length=200):
        """Generate story continuation."""

        # 1. Get device
        device = next(self.slm.parameters()).device

        # 2. Encode graph with GNN
        graph_embedding = self.gnn(
            graph_data.x.to(device),
            graph_data.edge_index.to(device)
        )  # [1, gnn_hidden_dim]

        # 3. Project to LM embedding space
        graph_context = self.fusion(graph_embedding)  # [1, n_prefix_tokens * hidden_size]
        graph_context = graph_context.view(1, self.n_prefix_tokens, -1)  # [1, n_prefix, hidden_size]

        # 4. Retrieve similar stories (optional)
        retrieved_context = ""
        if self.retriever is not None:
            retrieved_stories = self.retriever.retrieve(partial_text, keywords)
            if retrieved_stories:  # Check if list is not empty
                retrieved_context = " ".join(retrieved_stories[:1])

        # 5. Prepare input text with keywords
        if keywords and len(keywords) > 0:
            keywords_str = ", ".join(keywords)
            input_text = f"REQUIRED WORDS: {keywords_str}\n{retrieved_context}\nStory: {partial_text}"
        else:
            input_text = f"{retrieved_context} Story: {partial_text}"

        # 6. Tokenize and move to device
        inputs = self.tokenizer(input_text, return_tensors='pt', padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 7. Get input embeddings
        if hasattr(self.slm, 'model') and hasattr(self.slm.model, 'embed_tokens'):
            input_embeds = self.slm.model.embed_tokens(inputs['input_ids'])
        elif hasattr(self.slm, 'transformer') and hasattr(self.slm.transformer, 'wte'):
            input_embeds = self.slm.transformer.wte(inputs['input_ids'])
        else:
            input_embeds = self.slm.get_input_embeddings()(inputs['input_ids'])

        # 8. Ensure dtype compatibility
        graph_context = graph_context.to(input_embeds.dtype).to(device)

        # 9. Combine graph context with input embeddings
        combined_embeds = torch.cat([graph_context, input_embeds], dim=1)

        # 10. Create attention mask for combined embeddings
        original_attention = inputs.get('attention_mask', torch.ones_like(inputs['input_ids']))
        graph_attention = torch.ones((1, self.n_prefix_tokens), device=device, dtype=original_attention.dtype)
        combined_attention = torch.cat([graph_attention, original_attention], dim=1)

        # 11. Generate
        with torch.no_grad():
            outputs = self.slm.generate(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.3,
                top_p=0.85,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return outputs[0], generated_text