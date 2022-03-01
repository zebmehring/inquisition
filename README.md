# Inquisition: A Reformed Question-Answering System

Things we should still add:
- Test the ipmlementation with our written version of multihead attention (that's already written and exists in a branch; I'll find it) (Easy, somewhat important)
- Check over the C2Q Attention! (Difficult, potentially VERY important)
- CHange to depthwise separable convolutions (Easy-ish, not that imporant)
- Add in more dropout throughout the network (Easy-ish, potentially important)
- Add in layer dropout (have to also learn about what this is) (difficult, potentially important)
- We may want to change how we project the embedding down to hiddenSize before putting it through the highway network; the paper actually uses  charEmbDim + wordEmbDim dimension and feeds that to the highway network, then probably projects it down to hiddenSize afterwards (although they may never project it back down; it's not entirely clear) (Easy-ish, not sure of importance)
