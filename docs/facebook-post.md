Just built something I'm really excited about.

MidnightStar is a platform that uses deep learning to discover hidden connections between genes and diseases. It downloads real data from 5 major genomic databases (13+ million protein interactions, 1 million genetic associations, expression data across 54 human tissues, 20K protein structures from AlphaFold) — and trains graph neural networks to find patterns no human could spot manually.

The model scores 88% accuracy at predicting real biological connections. Meaning: it can suggest which genes might be linked to which diseases, even when that link isn't in any database yet.

Built it with Python, PyTorch, and Streamlit. 3,800 lines of code. Runs on a laptop. No cloud needed.

The craziest part? The simplest model (GNN with basic neighbor averaging) beat the fancy Transformer. Sometimes the straightforward approach wins.

If you're into AI, biology, or just like seeing what happens when you throw neural networks at real scientific data — happy to share more.
