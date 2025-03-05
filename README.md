# TokenVisor

Try it here: https://gduteaud.github.io/tokenvisor/

This is a simple little tool to visually explore how large language models tokenize and embed input text. It is in part inspired by [The Tokenizer Playground](https://huggingface.co/spaces/Xenova/the-tokenizer-playground), but this version shows the outputs of multiple models concurrently for easier comparison and also includes an embedding step.

## What's the point of doing this?

I find LLMs fascinating, I think it's important to understand how they work, and I'm a very visual person so I mainly undertook this to satisfy my own curiosity.

For example I think it's interesting to look at the differences in how models break down the following text. We can see that all three models assign similar meaning to "forest" and "beach" but beyond that they each group tokens/words slightly differently (should we go to/the/forest/or/the/beach/today, should/we go/to the forest/or/the beach today, should we go to the/forest/or the/beach/today). We can also see how a given word/token can be given a different embedding, thanks to positional encoding and attention*. This is true in all 3 models for the two instances of "the", but is particularly apparent in Llama.

*I'm also working on some other tools to help visualize these mechanisms

![Screen capture of the tool's colour-coded textual output for the input text "Hello! Should we go to the forest or the beach today?"](/src/assets/example_tokens.png)

## How does this actually work under the hood?

Just like my [Embedding Explorer](https://github.com/gduteaud/embedding-explorer), this uses the fantastic [transformers.js](https://github.com/huggingface/transformers.js) library to run entirely client-side. 

This tool takes the input text and runs it through each model's tokenizer to display the corresponding tokens. It then uses a feature extraction pipeline to obtain the model's hidden states and extract the per-token representations from the final layer. These hidden states represent each token's contextual embedding - how the model interprets each piece of text in the context of the full input.

These embeddings are high-dimensional (768+) vectors. We can't easily visualize those directly so we apply principal component analysis (PCA), a dimensionality reduction technique, to project those vectors to 3 dimensions. We then map that to the RGB colour space and display the resulting colour as a background to each token. This is a convenient, if crude, way for us to visualize the kinds of semantic relationships these models "understand" in text.

We also display each of these tokens as a point in a 3D scatter plot. We plot the outputs of all 3 models in the same graph for convenience but it's important to note that **relationships between the outputs of different models on that graph are not meaningful**. Because we apply PCA separately to the outputs of each model, the axes effectively correspond to different features ("meanings") for each model. This is why the outputs of each model cluster so neatly. Still, it's interesting to look at the relationships between tokens for the same model in this 3D space (you can toggle each model on and off by clicking it in the legend).

![Screen capture of the tool's colour-coded scatter plot output for the input text "Hello! Should we go to the forest or the beach today?"](/src/assets/example_plot.png)

## Why no embeddings for T5?

Out of the 4 model families this tool supports, T5 is the only one that uses the encoder-decoder architecture (BERT is encoder-only, and GPT and Llama are decoder-only). This isn't inherently a problem for our purposes as we can simply focus on the encoder outputs. The issue is that the transformers.js library we use in order to run this entirely in the browser is optimized for inference, and as such doesn't expose intermediate states the way the original Python library does. I could re-write this in Python (and I probably will at some point) but then it would no longer run fully client-side and that's a whole other can of worms.

## Why doesn't this use TypeScript?

TypeScript is fantastic (dare I say indispensable?) for larger-scale, team projects but for a quick and dirty solo project like this I felt the cons would outweigh the pros.

## What's next?

- Add user-facing explanations of what the tool is doing
- Specify the kind of tokenizer each model uses and their differences
- Extend to more model types
- Add alpha channel to RGB representation to visualize one more dimension
- "Rolling window" visualization to display raw embeddings instead of using dimensionality reduction