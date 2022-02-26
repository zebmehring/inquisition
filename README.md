# Inquisition: A Reformed Question-Answering System



The current state for QANet:
- Input embedding layer is done and I'm confident it's correct
- PositionalEncoding is done and I'm confident it's correct
- There's an implementation for EncoderBlock, but I think it's NOT correct. See 09fb3ae982301ccc8dca60a4562feebf0439db09 for the initial version without multihead attention (although it's missing self.ff2 -- which you can see in the newest version)
        I'm confident it doesn't work because, when I tested it with very small data, the NLL went haywire (whereas it didn't with BiDAF or with the input char embeddings or a positional encoding on top of the char embeddings)
- Because of the Encoder Block issue, I'm going to use the BiDAF implementation to test the context-query attention and output layer. Once we find the EncoderBlock issue, we should be good. I reverted back to using the RNN encoders and will now test C2Q attention and then we just need the output layer.
