**Introduction**

Recurrent neural networks (RNNs), particularly LSTMs and gated recurrent networks, have been the dominant approach for sequence modeling tasks like language modeling and machine translation. However, their sequential nature limits parallelization, making training inefficient for long sequences. While advancements like factorization tricks and conditional computation have improved efficiency, the fundamental constraint of sequential computation remains. Attention mechanisms help model dependencies without regard to distance but are typically combined with recurrent networks.

