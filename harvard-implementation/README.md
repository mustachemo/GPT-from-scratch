# Transformer Model Implementation

## Overview

This repository contains the implementation of the Transformer model as described in the paper "Attention Is All You Need" by Vaswani et al., and based on the annotated guide provided by the Harvard NLP group. The Transformer model revolutionizes sequence-to-sequence learning by replacing traditional recurrent layers with multi-head self-attention mechanisms and positionally-encoded inputs. This architecture allows for significantly more parallelization during training and better handling of long-range dependencies across sequences.

## Model Description

The Transformer model is structured around the encoder-decoder architecture:

- **Encoder**: Composed of a stack of N identical layers, each containing two sub-layers: a multi-head self-attention mechanism, and a simple, position-wise fully connected feed-forward network. A residual connection is employed around each of the two sub-layers, followed by layer normalization.

- **Decoder**: Also comprises N identical layers, with an additional third sub-layer that performs multi-head attention over the encoder's output. Similar to the encoder, each sub-layer in the decoder is equipped with a residual connection, followed by layer normalization.

The attention mechanism used in the Transformer allows each output element to attend to every input element dynamically, making it highly effective for tasks such as machine translation.
