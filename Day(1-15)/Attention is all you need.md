# Attention is all you need
---
* Aman singh(USAR) - (Coordinator)
* Asif Iqbal khan(USAR)
* Md.Wajid(USAR)
* Yash Gupta(USICT)
* Ankit(USAR)

---

# Introduction

Recurrent neural networks (RNNs), particularly LSTMs and gated recurrent networks, have been the dominant approach for sequence modeling tasks like language modeling and machine translation. However, their sequential nature limits parallelization, making training inefficient for long sequences. While advancements like factorization tricks and conditional computation have improved efficiency, the fundamental constraint of sequential computation remains. Attention mechanisms help model dependencies without regard to distance but are typically combined with recurrent networks.


# Transformer model

<img width="1013" alt="Image" src="https://github.com/user-attachments/assets/c0439df2-f27e-467e-949a-04758ae73384" />

The Transformer is a deep learning model introduced in 2017 that eliminates recurrence and relies entirely on self-attention and feedforward layers to process sequential data. It is designed for parallelization, making it more efficient than RNNs and LSTMs for tasks like machine translation and language modeling.
The model consists of encoder-decoder architecture:
•	The encoder processes input sequences using multi-head self-attention and feedforward layers, capturing contextual relationships between words.
•	The decoder generates output sequences by using both self-attention (to understand generated words) and encoder-decoder attention (to incorporate input information).


# Encoder
<img width="202" style="margin-left: 500px" alt="Image" src="https://github.com/user-attachments/assets/356c76cc-6a34-4bb2-9c04-767e11ede9df" /> The encoder in a Transformer model consists of multiple stacked layers that process input tokens to generate rich contextual representations. First, input tokens are converted into embeddings, with positional encodings added to retain word order. Each encoder layer includes a multi-head self-attention mechanism, allowing tokens to attend to different parts of the sequence, followed by a feed-forward network (FFN) that applies non-linearity to enhance feature extraction. Residual connections and layer normalization are used after both self-attention and FFN layers to stabilize training. By stacking multiple encoder layers, the model captures deep contextual relationships, making the encoded representations highly meaningful for downstream tasks like translation or text generation.

# Calculation of Positional embedding

---
  PE(pos 2i)  = sin (pos) / (2i / (10000d  * model))
---
  PE(pos 2i + 1)  = cos (pos) / (2i / (10000d  * model))
---

# Decoder
The decoder in a Transformer model generates output sequences by processing encoded representations and previously generated tokens. Like the encoder, it consists of multiple layers, each containing key components. First, input tokens (such as previously generated words in translation tasks) are embedded with positional encodings. The decoder has a masked multi-head self-attention mechanism that prevents future tokens from being seen during training, ensuring autoregressive generation. Next, a multi-head attention layer attends to the encoder’s output, helping the decoder focus on relevant input information. A feed-forward network (FFN) follows, applying transformations to enhance feature extraction. Residual connections and layer normalization stabilize learning at each step. By stacking multiple decoder layers, the model refines token predictions, ultimately producing coherent and contextually appropriate sequences.

<img width="300" height="600" alt="Image" src="https://github.com/user-attachments/assets/eb6b94e8-ad3c-4190-ac3e-a8756ff77238" />

# Self Attention

Self-attention is a mechanism that enables a model to weigh different parts of an input sequence when processing each token, capturing long-range dependencies efficiently. It works by transforming input embeddings into Query (Q), Key (K), and Value (V) vectors, computing attention scores using a scaled dot product, applying softmax to get attention weights, and then using these weights to compute a weighted sum of values. This allows for parallelization, better handling of long-range dependencies, and improved context sensitivity compared to RNNs. Self-attention is the core of multi-head attention in Transformer models, making them highly effective for NLP tasks.


# Scaled dot-Product Attention

Scaled dot-product attention is the core mechanism behind self-attention in Transformers. It determines the importance of different tokens in a sequence by computing attention scores between Query (Q), Key (K), and Value (V) vectors.

---
The process involves:

* Computing attention scores using the dot product of Q and K:

---
                               Score = QKT
---
* Scaling the scores by dividing by √dk (the dimension of k) to stabilize gradients:

---
                           Scaled Score = QKT / √dk

---
  
* Applying softmax to convert score into attention weights.
* Multiplying the weights with V to get the final output.

 ``` bash
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    return (lambda d_k, scores, attention_weights: (torch.matmul(attention_weights, V), attention_weights))(
        Q.shape[-1],
        torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(Q.shape[-1], dtype=torch.float32)),
        F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(Q.shape[-1], dtype=torch.float32)), dim=-1)
    )
```


# Multi-Head Attention

Multi-head attention is a key component of the Transformer model that enhances the self-attention mechanism by applying multiple attention layers in parallel. Instead of using a single attention function, the input is projected into multiple sets of Query (Q), Key (K), and Value (V) matrices, allowing the model to capture different aspects of relationships between words.
Each attention head independently computes attention scores and produces an output. These outputs are then concatenated and passed through a final projection layer. This approach enables the model to focus on multiple positions in the sequence simultaneously, improving its ability to capture complex dependencies.

# Why LSTMs and RNNs are not used

LSTMs and RNNs are not used in Transformer models because they rely on sequential processing, making them inefficient for handling long-range dependencies and parallel computation. Transformers, on the other hand, use self-attention, allowing them to process entire sequences simultaneously rather than step-by-step, significantly improving training speed and scalability. Additionally, RNNs suffer from the vanishing gradient problem, making it difficult to capture long-term dependencies, whereas Transformers use positional encodings and attention mechanisms to effectively model relationships between distant words. The lack of recurrence in Transformers also enables better utilization of modern hardware, such as GPUs and TPUs, making them more efficient for large-scale natural language processing tasks.


# Challenges and Ongoing Research  

Computational and Environmental Costs  
- Training BERT-Large: ~1,024 TPU v3 days, ~1,400 kg CO2 emissions.  
- Memory Bottlenecks: Long sequences (e.g., 4k tokens) require optimizations like sparse attention (Longformer) or memory-efficient kernels (FlashAttention).  


# Model Interpretability  

- Attention Visualization: Tools like BertViz map attention heads but lack semantic clarity.  
- Probing Studies: Reveal that lower layers capture syntax, while higher layers handle semantics.  

# Efficiency Improvements 

- Quantization: Reducing precision (e.g., 16-bit to 8-bit) for faster inference.  
- Pruning: Removing redundant attention heads/weights.  
- Hybrid Models: Combining Transformers with RNNs (e.g., Transformer-XH).  

# Future Directions  

1. Efficient Architectures: Models like Linformer (linear attention) and Performer (kernel approximations).  
2. Multimodal Transformers: CLIP (text-image), Wav2Vec 2.0 (speech).  
3. Explainability: Developing tools to decode self-attention’s decision-making.  
4. General-Purpose AI: Towards models that reason across text, vision, and robotics (e.g., Gato).  

# Conclusion  

The Transformer and BERT redefined NLP by prioritizing attention over recurrence, enabling bidirectional context capture, and democratizing transfer learning. While challenges like computational costs and interpretability persist, their legacy lies in enabling adaptable, high-performance language systems. Future work will focus on efficiency, ethical AI, and expanding Transformers beyond language into multimodal reasoning.  

# References  

1. Vaswani, A. et al. (2017). Attention Is All You Need. 
2. Devlin, J. et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. 

