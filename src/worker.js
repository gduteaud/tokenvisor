import { pipeline, env, AutoTokenizer } from "@huggingface/transformers";
import { PCA } from "ml-pca";

console.log("[Worker] Initializing worker...");

env.allowLocalModels = false;

const TOKENIZER_MAPPINGS = new Map();
const MODEL_MAPPINGS = new Map();

self.addEventListener("message", async (event) => {
  console.log(`[Worker: ${event.data.model_id}] Received message:`, event.data);

  const model_id = event.data.model_id;
  const text = event.data.text;

  if (!text) {
    console.log(
      `[Worker: ${model_id}] No text provided, only loading tokenizer.`
    );
    // Just load the tokenizer and exit
    try {
      let tokenizerPromise = TOKENIZER_MAPPINGS.get(model_id);
      if (!tokenizerPromise) {
        console.log(`[Worker: ${model_id}] Loading tokenizer`);
        tokenizerPromise = AutoTokenizer.from_pretrained(`Xenova/${model_id}`);
        TOKENIZER_MAPPINGS.set(model_id, tokenizerPromise);
      }
      await tokenizerPromise; // Wait for it to load
      console.log(`[Worker: ${model_id}] Successfully pre-loaded tokenizer`);
    } catch (error) {
      console.error(
        `[Worker: ${model_id}] Error pre-loading tokenizer: ${error}`
      );
      self.postMessage({ error: error.message, model_id });
    }
    return;
  }

  try {
    let tokenizerPromise = TOKENIZER_MAPPINGS.get(model_id);
    if (!tokenizerPromise) {
      console.log(`[Worker: ${model_id}] Loading tokenizer`);
      tokenizerPromise = AutoTokenizer.from_pretrained(`Xenova/${model_id}`);
      TOKENIZER_MAPPINGS.set(model_id, tokenizerPromise);
    }

    const tokenizer = await tokenizerPromise;
    console.log(`[Worker: ${model_id}] Tokenizer loaded`);

    // Tokenize input text
    console.log(`[Worker: ${model_id}] Tokenizing text:`, text);
    const token_ids = tokenizer.encode(text);
    let decoded_tokens = token_ids.map((x) => tokenizer.decode([x]));

    let margins = [];
    console.log(
      `[Worker: ${model_id}] Tokenizer type:`,
      tokenizer.constructor.name
    );
    console.log(
      `[Worker: ${model_id}] Decoded tokens before processing:`,
      decoded_tokens
    );

    switch (tokenizer.constructor.name) {
      case "BertTokenizer":
      case "DebertaV2Tokenizer":
        margins = decoded_tokens.map((x, i) =>
          i === 0 || x.startsWith("##") ? 0 : 8
        );
        decoded_tokens = decoded_tokens.map((x) => x.replace("##", ""));
        break;
      case "T5Tokenizer":
        console.log(
          `[Worker: ${model_id}] T5 processing - first token:`,
          decoded_tokens[0]
        );
        if (decoded_tokens.length > 0 && decoded_tokens[0] !== " ") {
          decoded_tokens[0] = decoded_tokens[0].replace(/^ /, "");
        }
        break;
      case "LlamaTokenizer":
      case "PhiTokenizer":
        decoded_tokens = decoded_tokens.map(token => 
          token.replace(/^▁/, '')
        );
        margins = decoded_tokens.map((_, i) => 
          i === 0 || !tokenizer.decode([token_ids[i]]).startsWith('▁') ? 0 : 8
        );
        break;
    }
    console.log(
      `[Worker: ${model_id}] Decoded tokens after processing:`,
      decoded_tokens
    );

    let response = {
      model_id,
      decoded: decoded_tokens,
      margins,
      ids: token_ids,
    };

    // Send initial tokenization results
    self.postMessage({
      model_id,
      decoded: decoded_tokens,
      margins,
      ids: token_ids,
      partial: true, // Flag to indicate this is just tokenization
    });

    // Compute embeddings
    let output;
    // Skip embedding computation for T5 models
    if (tokenizer.constructor.name === "T5Tokenizer") {
      console.log(`[Worker: ${model_id}] T5 model - skipping embedding computation`);
      response.complete = true; // Mark as complete even without embeddings
      response.embeddings = null;
      response.tokenColors = decoded_tokens.map(() => [128, 128, 128]); // Default gray colors
      console.log(`[Worker: ${model_id}] Sending tokenization-only results back.`);
      self.postMessage(response);
      return;
    }

    // Existing pipeline-based approach for other models
    let modelPromise = MODEL_MAPPINGS.get(model_id);
    if (!modelPromise) {
      console.log(
        `[Worker: ${model_id}] Loading feature extraction pipeline`
      );

      // Configure pipeline options based on model
      const pipelineOptions = {
        pooling: false, // Disable pooling to get per-token embeddings
        normalize: false, // Typically not needed for per-token embeddings
        tokenize: false, // Disable internal tokenization
      };

      // Add specific model file names for different architectures
      if (model_id === "gpt2") {
        pipelineOptions.model_file_name = "decoder_model_merged";
      } else if (model_id === "llama-68m") {
        pipelineOptions.model_file_name = "decoder_model_merged";
      }

      modelPromise = pipeline(
        "feature-extraction",
        `Xenova/${model_id}`,
        pipelineOptions
      );
      MODEL_MAPPINGS.set(model_id, modelPromise);
    }
    const extractor = await modelPromise;
    output = await extractor(text);

    // Get the last layer's embeddings
    const lastLayerEmbeddings = Array.from(output.data).slice(
      -token_ids.length * output.dims[2]
    );

    // Reshape into [sequence_length, hidden_size]
    const reshapedEmbeddings = [];
    const hiddenSize = output.dims[2];
    for (let i = 0; i < token_ids.length; i++) {
      reshapedEmbeddings.push(
        lastLayerEmbeddings.slice(i * hiddenSize, (i + 1) * hiddenSize)
      );
    }

    // Handle PCA and color computation
    let rgbColors;
    
    // Add debug logging
    console.log(`[Worker: ${model_id}] Embeddings before PCA:`, {
      length: reshapedEmbeddings.length,
      sampleEmbedding: reshapedEmbeddings[0],
      dimensions: reshapedEmbeddings[0]?.length
    });

    // Check for valid embeddings
    if (!reshapedEmbeddings.length || !reshapedEmbeddings[0]?.length) {
      console.error(`[Worker: ${model_id}] Invalid embeddings detected`);
      rgbColors = token_ids.map(() => [128, 128, 128]); // Default gray color
    } else if (reshapedEmbeddings.length < 3) {
      // For very short sequences, assign a default color
      rgbColors = reshapedEmbeddings.map(() => [128, 128, 128]); // Default gray color
    } else {
      try {
        // Compute PCA for sequences with sufficient length
        const pca = new PCA(reshapedEmbeddings);
        const pcaOutput = pca.predict(reshapedEmbeddings, { nComponents: 3 });

        // Project to RGB space (0-255) with safer scaling
        rgbColors = pcaOutput.to2DArray().map((point) =>
          point.map((val) => {
            const min = Math.min(...pcaOutput.to2DArray().flat());
            const max = Math.max(...pcaOutput.to2DArray().flat());
            const scaled = Math.round(((val - min) / (max - min)) * 255);
            return Math.max(0, Math.min(255, scaled));
          })
        );
      } catch (error) {
        console.error(`[Worker: ${model_id}] Error computing PCA:`, error);
        rgbColors = token_ids.map(() => [128, 128, 128]); // Default gray color
      }
    }

    console.log(
      `[Worker: ${model_id}] Embeddings and colors computed:`,
      reshapedEmbeddings,
      rgbColors
    );
    response.embeddings = reshapedEmbeddings;
    response.tokenColors = rgbColors;
    response.complete = true; // Flag to indicate this includes embeddings

    console.log(`[Worker: ${model_id}] Sending results back.`);
    self.postMessage(response);
  } catch (error) {
    console.error(`[Worker: ${model_id}] Error processing message:`, error);
    self.postMessage({ error: error.message, model_id });
  }
});
