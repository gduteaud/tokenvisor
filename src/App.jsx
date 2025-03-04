import { useState, useRef, useEffect } from 'react'
import './App.css'

const TOKENIZERS = [
  { id: 'distilbert-base-uncased', name: 'DistilBERT Base Uncased' },
  { id: 'gpt2', name: 'GPT-2' },
  { id: 't5-small', name: 'T5-Small' }
];

// To start: GPT type (GPT2Tokenizer), BERT type (DistilBertTokenizer), T5 type (T5Tokenizer)
// Specify tokenizer type (WordPiece, SentencePiece, BytePairEncoding)

function App() {
  const [input, setInput] = useState('')
  const [tokenResults, setTokenResults] = useState({})
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [loadingProgress, setLoadingProgress] = useState({})
  
  const worker = useRef(null)

  useEffect(() => {
    console.log('[App] Starting useEffect, worker exists:', !!worker.current);
    
    if (!worker.current) {
        console.log('[App] Initializing new worker');
        try {
            const workerUrl = new URL('./worker.js', import.meta.url);
            console.log('[App] Created worker URL:', workerUrl.href);
            
            worker.current = new Worker(workerUrl, {
                type: 'module'
            });
            
            console.log('[App] Worker instance created');

            worker.current.onerror = (error) => {
                console.error('[App] Worker error event:', error);
                setError(`Worker error: ${error.message}`);
                setIsLoading(false);
            };

            worker.current.addEventListener('message', (e) => {
                console.log('[App] Received worker response:', e.data);
                
                if (e.data.error) {
                    console.error('[App] Worker reported error:', e.data.error);
                    setError(e.data.error);
                    setIsLoading(false);
                    return;
                }

                const currentTokenizer = e.data.model_id;
                console.log('[App] Updating results for tokenizer:', currentTokenizer);
                setTokenResults(prevResults => {
                    const newResults = {
                        ...prevResults,
                        [currentTokenizer]: {
                            decoded: e.data.decoded,
                            margins: e.data.margins,
                            ids: e.data.ids,
                            embeddings: e.data.embeddings,
                            tokenColors: e.data.tokenColors
                        }
                    };
                    console.log('[App] New token results:', newResults);
                    return newResults;
                });
                setIsLoading(false);
            });

            // Initial tokenizer loading
            console.log('[App] Loading initial tokenizers:', TOKENIZERS);
            TOKENIZERS.forEach(tokenizer => {
                console.log('[App] Requesting tokenizer load:', tokenizer.id);
                worker.current.postMessage({ 
                    model_id: tokenizer.id
                });
            });

        } catch (error) {
            console.error('[App] Worker initialization failed:', error);
            setError('Failed to initialize worker: ' + error.message);
        }
    }
}, []);

  const handleTokenize = async () => {
    console.log('[App] handleTokenize called, input length:', input.length);
    
    if (!worker.current || !input.trim()) {
        console.log('[App] Tokenization cancelled:', !worker.current ? 'No worker' : 'Empty input');
        return;
    }
    
    console.log('[App] Starting tokenization');
    setIsLoading(true);
    setError(null);
    setTokenResults({});
    
    TOKENIZERS.forEach(tokenizer => {
        console.log('[App] Sending tokenization request for:', tokenizer.id);
        worker.current.postMessage({
            model_id: tokenizer.id,
            text: input
        });
    });
  }

  return (
    <div className="container">
      <h1>TokenVisor</h1>
      <h2>A simple LLM tokenization visualizer</h2>
      
      <div className="tokenizer-controls">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Enter text to tokenize..."
          rows={5}
        />
        
        <button 
          onClick={handleTokenize}
          disabled={isLoading || !input.trim()}
        >
          {isLoading ? 'Tokenizing...' : 'Tokenize'}
        </button>
      </div>

      {error && (
        <div className="error">
          {error}
        </div>
      )}

      {TOKENIZERS.map(({ id, name }) => (
        <div key={id} className="tokenizer-section">
          <h3 className="model-name">{name}</h3>
            {tokenResults[id] && (
              <div className="tokens-list">
                {tokenResults[id].decoded.map((token, index) => (
                  <div 
                    key={index} 
                    className="token-container"
                    style={{ marginLeft: tokenResults[id].margins[index] }}
                  >
                    <span 
                      className="token"
                      style={{
                        background: tokenResults[id].tokenColors ? 
                          `rgb(${tokenResults[id].tokenColors[index].join(',')})` : 
                          'rebeccapurple'
                      }}
                    >{token}</span>
                    <span className="token-id">{tokenResults[id].ids[index]}</span>
                  </div>
                ))}
              </div>
            )}
        </div>
      ))}
    </div>
  )
}

export default App