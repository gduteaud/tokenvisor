import { useState, useRef, useEffect } from 'react'
import Plot from 'react-plotly.js'
import './App.css'

const TOKENIZERS = [
  { id: 'distilbert-base-uncased', name: 'DistilBERT Base Uncased' },
  { id: 'gpt2', name: 'GPT-2' },
  { id: 'llama-68m', name: 'Llama 68M'},
  { id: 't5-small', name: 'T5-Small' },
];

// To start: GPT type (GPT2Tokenizer), BERT type (DistilBertTokenizer), T5 type (T5Tokenizer)
// Specify tokenizer type (WordPiece, SentencePiece, BytePairEncoding)

const MODEL_COLORS = {
  'distilbert-base-uncased': '#ff7f0e',  // orange
  'gpt2': '#1f77b4',  // blue
  'llama-68m': '#d62728',  // red
};

function App() {
  const [input, setInput] = useState('')
  const [tokenResults, setTokenResults] = useState({})
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  
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
                            ...prevResults[currentTokenizer],  // Preserve any existing data
                            decoded: e.data.decoded,
                            margins: e.data.margins,
                            ids: e.data.ids,
                            // Only update embeddings and colors if this is a complete response
                            ...(e.data.complete ? {
                                embeddings: e.data.embeddings,
                                tokenColors: e.data.tokenColors
                            } : {})
                        }
                    };
                    console.log('[App] New token results:', newResults);
                    return newResults;
                });
                
                // Only set loading to false when we get the complete response
                if (e.data.complete) {
                    setIsLoading(false);
                }
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

  const prepareScatterData = () => {
    const scatterData = [];
    
    Object.entries(tokenResults).forEach(([modelId, results]) => {
      // Skip T5
      if (modelId === 't5-small') return;
      
      if (results.tokenColors && results.tokenColors.length > 0) {
        const trace = {
          type: 'scatter3d',
          mode: 'markers+text',
          name: TOKENIZERS.find(t => t.id === modelId)?.name || modelId,
          x: results.tokenColors.map(color => color[0]),  // R values
          y: results.tokenColors.map(color => color[1]),  // G values
          z: results.tokenColors.map(color => color[2]),  // B values
          text: results.decoded,
          textposition: 'middle right',
          textfont: {
            color: '#fff',
            size: 10
          },
          marker: {
            size: 4,
            color: MODEL_COLORS[modelId],
            opacity: 0.7
          },
          hovertext: results.decoded,
          hoverinfo: 'text'
        };
        scatterData.push(trace);
      }
    });
    
    return scatterData;
  };

  return (
    <div className="container">
      <div className="header">
        <h1>TokenVisor</h1>
        <h2>A simple visual tool to explore LLM tokenization</h2>
        <p>Enter some text below and click the "Tokenize" button to visualize and compare the different ways model families tokenize input text and embed those tokens.</p>
      </div>
      
      <div className="tokenizer-controls">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          rows={5}
        />
        
        <div className="input-stats">
          <span>Words: {input.trim() ? input.trim().split(/\s+/).length : 0}</span>
          <span>Characters: {input.length}</span>
        </div>
        
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
        tokenResults[id] && (  // Only render if we have results for this tokenizer
          <div key={id} className="tokenizer-section">
            <div className="model-header">
              <h3 className="model-name">
                {name}
              </h3>
              <p className="token-count">
                ({tokenResults[id].ids.length} tokens)
            </p>
            </div>
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
                            '#e0e0e0'  // Default color when embeddings aren't ready
                    }}
                  >{token}</span>
                  <span className="token-id">{tokenResults[id].ids[index]}</span>
                </div>
              ))}
            </div>
          </div>
        )
      ))}

      {Object.keys(tokenResults).length > 0 && (
        <div className="embeddings-visualization">
          <Plot
            data={prepareScatterData()}
            layout={{
              width: 800,
              height: 800,
              title: '3D Token Embeddings',
              paper_bgcolor: '#242424',
              plot_bgcolor: '#242424',
              legend: {
                x: 0,
                xanchor: 'left',
                bgcolor: '#242424',
                font: { color: '#fff' }
              },
              scene: {
                aspectmode: 'cube',
                xaxis: { 
                  title: {
                    text: 'R',
                    font: { color: '#fff' }
                  },
                  gridcolor: '#404040',
                  zerolinecolor: '#404040',
                  tickfont: { color: '#fff' },
                  range: [0, 255],
                  dtick: 51
                },
                yaxis: { 
                  title: {
                    text: 'G',
                    font: { color: '#fff' }
                  },
                  gridcolor: '#404040',
                  zerolinecolor: '#404040',
                  tickfont: { color: '#fff' },
                  range: [0, 255],
                  dtick: 51
                },
                zaxis: { 
                  title: {
                    text: 'B',
                    font: { color: '#fff' }
                  },
                  gridcolor: '#404040',
                  zerolinecolor: '#404040',
                  tickfont: { color: '#fff' },
                  range: [0, 255],
                  dtick: 51
                }
              },
              margin: {
                l: 0,
                r: 0,
                b: 0,
                t: 30
              },
              font: {
                color: '#fff'
              }
            }}
            config={{
              displayModeBar: true,
              responsive: true
            }}
          />
        </div>
      )}
    </div>
  )
}

export default App