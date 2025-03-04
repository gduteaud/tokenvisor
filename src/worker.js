import { env, AutoTokenizer, AutoModel } from '@huggingface/transformers';

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
                console.log(`[Worker] Loading model for: ${model_id}`);
                modelPromise = AutoModel.from_pretrained("Xenova/distilbert-base-uncased", { dtype: 'q8' });
                MODEL_MAPPINGS.set(model_id, modelPromise);
            }

            const model = await modelPromise;
            console.log('[Worker] Model loaded, computing embeddings...');

            const tokenized = tokenizer(text);
            const outputs = await model(tokenized);
            console.log('[Worker] Model outputs:', outputs);
            response.embeddings = outputs.logits.tolist();
            console.log('[Worker] Embeddings:', response.embeddings);
        }

        console.log('[Worker] Sending results back.');
        self.postMessage(response);

    } catch (error) {
        console.error('[Worker] Error processing message:', error);
        self.postMessage({ error: error.message, model_id });
    }
});