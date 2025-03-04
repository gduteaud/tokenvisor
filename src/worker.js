import { env, AutoTokenizer, pipeline } from '@huggingface/transformers';
import { PCA } from 'ml-pca';

console.log('[Worker] Initializing worker...');

env.allowLocalModels = false;

const TOKENIZER_MAPPINGS = new Map();
const MODEL_MAPPINGS = new Map();

self.addEventListener('message', async (event) => {
    console.log('[Worker] Received message:', event.data);
    
    const model_id = event.data.model_id;
    const text = event.data.text;
    
    if (!text) {
        console.log('[Worker] No text provided, skipping processing.');
        return;
    }

    try {
        let tokenizerPromise = TOKENIZER_MAPPINGS.get(model_id);
        if (!tokenizerPromise) {
            console.log(`[Worker] Loading tokenizer for: ${model_id}`);
            tokenizerPromise = AutoTokenizer.from_pretrained(model_id);
            TOKENIZER_MAPPINGS.set(model_id, tokenizerPromise);
        }
        
        const tokenizer = await tokenizerPromise;
        console.log(`[Worker] Tokenizer loaded for: ${model_id}`);

        // Tokenize input text
        console.log('[Worker] Tokenizing text:', text);
        const token_ids = tokenizer.encode(text);
        const decoded_tokens = token_ids.map(x => tokenizer.decode([x]));

        let margins = [];
        switch (tokenizer.constructor.name) {
            case 'BertTokenizer':
                margins = decoded_tokens.map((x, i) => i === 0 || x.startsWith('##') ? 0 : 8);
                decoded_tokens = decoded_tokens.map(x => x.replace('##', ''));
                break;
            case 'T5Tokenizer':
                if (decoded_tokens.length > 0 && decoded_tokens[0] !== ' ') {
                    decoded_tokens[0] = decoded_tokens[0].replace(/^ /, '');
                }
                break;
        }

        let response = {
            model_id,
            decoded: decoded_tokens,
            margins,
            ids: token_ids
        };

        // If DistilBERT, also compute embeddings
        if (model_id === "distilbert-base-uncased") {
            let modelPromise = MODEL_MAPPINGS.get(model_id);
            if (!modelPromise) {
                console.log(`[Worker] Loading feature extraction pipeline for: ${model_id}`);
                modelPromise = pipeline('feature-extraction', 'Xenova/distilbert-base-uncased', {
                    pooling: false, // Disable pooling to get per-token embeddings
                    normalize: false // Typically not needed for per-token embeddings
                });
                MODEL_MAPPINGS.set(model_id, modelPromise);
            }

            const extractor = await modelPromise;
            console.log('[Worker] Pipeline loaded, computing embeddings...');

            const output = await extractor(text);
            // Get the last layer's embeddings (shape: [batch_size=1, sequence_length, hidden_size=768])
            const lastLayerEmbeddings = Array.from(output.data).slice(-token_ids.length * 768);
            
            // Reshape into [sequence_length, hidden_size]
            const reshapedEmbeddings = [];
            for (let i = 0; i < token_ids.length; i++) {
                reshapedEmbeddings.push(lastLayerEmbeddings.slice(i * 768, (i + 1) * 768));
            }

            // Compute PCA
            const pca = new PCA(reshapedEmbeddings);
            const pcaOutput = pca.predict(reshapedEmbeddings, { nComponents: 3 });

            // Project to RGB space (0-255) with safer scaling
            const rgbColors = pcaOutput.to2DArray().map(point => 
                point.map(val => {
                    // Use a simpler min-max scaling approach
                    const min = Math.min(...pcaOutput.to2DArray().flat());
                    const max = Math.max(...pcaOutput.to2DArray().flat());
                    const scaled = Math.round(((val - min) / (max - min)) * 255);
                    return Math.max(0, Math.min(255, scaled));
                })
            );

            console.log('[Worker] Embeddings and colors computed:', reshapedEmbeddings, rgbColors);
            response.embeddings = reshapedEmbeddings;
            response.tokenColors = rgbColors;
        }

        console.log('[Worker] Sending results back.');
        self.postMessage(response);

    } catch (error) {
        console.error('[Worker] Error processing message:', error);
        self.postMessage({ error: error.message, model_id });
    }
});