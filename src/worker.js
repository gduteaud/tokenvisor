import { env, AutoTokenizer } from '@huggingface/transformers'

console.log('[Worker] Initializing worker...');

env.allowLocalModels = false;

// This is a map of all the tokenizer instances that we have loaded.
// model_id -> promise that resolves to tokenizer
const TOKENIZER_MAPPINGS = new Map();

// Listen for messages from the main thread
self.addEventListener('message', async (event) => {
    console.log('[Worker] Received message:', event.data);
    
    try {
        let tokenizerPromise = TOKENIZER_MAPPINGS.get(event.data.model_id);
        console.log('[Worker] Existing tokenizer promise:', !!tokenizerPromise);
        
        if (!tokenizerPromise) {
            console.log('[Worker] Loading new tokenizer for:', event.data.model_id);
            tokenizerPromise = AutoTokenizer.from_pretrained(event.data.model_id)
                .then(tokenizer => {
                    console.log('[Worker] Tokenizer loaded:', tokenizer.constructor.name);
                    
                    switch (tokenizer.constructor.name) {
                        case 'T5Tokenizer':
                            console.log('[Worker] Configuring T5Tokenizer');
                            tokenizer.decoder.addPrefixSpace = false;
                            break;
                    }
                    return tokenizer;
                })
                .catch(error => {
                    console.error('[Worker] Failed to load tokenizer:', error);
                    throw error;
                });
            
            TOKENIZER_MAPPINGS.set(event.data.model_id, tokenizerPromise);
        }

        const tokenizer = await tokenizerPromise;
        console.log('[Worker] Using tokenizer:', tokenizer.constructor.name);

        const text = event.data.text;
        if (!text) {
            console.log('[Worker] No text provided, skipping tokenization');
            return;
        }

        console.log('[Worker] Tokenizing text:', text.substring(0, 50) + '...');
        const start = performance.now();
        const token_ids = tokenizer.encode(text);
        const end = performance.now();
        console.log('[Worker] Token IDs:', token_ids);
        console.log('[Worker]', `Tokenized ${text.length} characters in ${(end - start).toFixed(2)}ms`);

        let decoded = token_ids.map(x => tokenizer.decode([x]));
        console.log('[Worker] Decoded tokens:', decoded);

        let margins = [];

        console.log('[Worker] Post-processing for tokenizer:', tokenizer.constructor.name);
        switch (tokenizer.constructor.name) {
            case 'BertTokenizer':
                margins = decoded.map((x, i) => i === 0 || x.startsWith('##') ? 0 : 8);
                decoded = decoded.map(x => x.replace('##', ''));
                break;
            case 'T5Tokenizer':
                if (decoded.length > 0 && decoded[0] !== ' ') {
                    decoded[0] = decoded[0].replace(/^ /, '');
                }
                break;
        }

        console.log('[Worker] Sending results back:', {
            model_id: event.data.model_id,
            token_ids: token_ids.length,
            decoded: decoded.length,
            margins: margins.length
        });

        self.postMessage({
            model_id: event.data.model_id,
            token_ids,
            decoded,
            margins
        });
    } catch (error) {
        console.error('[Worker] Error processing message:', error);
        self.postMessage({
            error: error.message,
            model_id: event.data.model_id
        });
    }
});