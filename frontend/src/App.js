import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { Button } from './components/ui/button';
import { Input } from './components/ui/input';
import { Textarea } from './components/ui/textarea';
import { Badge } from './components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Progress } from './components/ui/progress';
import { Alert, AlertDescription } from './components/ui/alert';
import { Brain, Zap, Target, Cpu, Play, Settings, BarChart3 } from 'lucide-react';
import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

function App() {
  const [modelInfo, setModelInfo] = useState(null);
  const [position, setPosition] = useState('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [trainingData, setTrainingData] = useState('');
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchModelInfo();
  }, []);

  const fetchModelInfo = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/model/info`);
      const data = await response.json();
      setModelInfo(data);
    } catch (err) {
      setError('Failed to fetch model info');
    }
  };

  const predictMoves = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${BACKEND_URL}/api/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ fen: position }),
      });
      
      if (!response.ok) {
        throw new Error('Prediction failed');
      }
      
      const data = await response.json();
      setPredictions(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const startTraining = async () => {
    if (!trainingData.trim()) {
      setError('Please provide training data (PGN games)');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const games = trainingData.split('\n\n').filter(game => game.trim());
      const response = await fetch(`${BACKEND_URL}/api/train`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          pgn_games: games,
          epochs: 10,
          batch_size: 32
        }),
      });
      
      if (!response.ok) {
        throw new Error('Training failed');
      }
      
      const data = await response.json();
      setTrainingStatus(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const resetPosition = () => {
    setPosition('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
    setPredictions(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-md border-b border-slate-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl">
                <Brain className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  Chess Transformer
                </h1>
                <p className="text-sm text-slate-600">AI-Powered Chess Engine</p>
              </div>
            </div>
            {modelInfo && (
              <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                <Cpu className="h-3 w-3 mr-1" />
                {modelInfo.config.device.toUpperCase()}
              </Badge>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 space-y-8">
        {/* Model Info Cards */}
        {modelInfo && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <Card className="bg-white/70 backdrop-blur border-slate-200 hover:shadow-lg transition-all duration-300">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-slate-600">Parameters</p>
                    <p className="text-2xl font-bold text-slate-900">
                      {(modelInfo.parameters / 1000000).toFixed(1)}M
                    </p>
                  </div>
                  <Settings className="h-8 w-8 text-blue-500" />
                </div>
              </CardContent>
            </Card>

            <Card className="bg-white/70 backdrop-blur border-slate-200 hover:shadow-lg transition-all duration-300">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-slate-600">Total Moves</p>
                    <p className="text-2xl font-bold text-slate-900">
                      {modelInfo.config.total_moves?.toLocaleString()}
                    </p>
                  </div>
                  <Target className="h-8 w-8 text-green-500" />
                </div>
              </CardContent>
            </Card>

            <Card className="bg-white/70 backdrop-blur border-slate-200 hover:shadow-lg transition-all duration-300">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-slate-600">Layers</p>
                    <p className="text-2xl font-bold text-slate-900">
                      {modelInfo.config.num_layers}
                    </p>
                  </div>
                  <Brain className="h-8 w-8 text-purple-500" />
                </div>
              </CardContent>
            </Card>

            <Card className="bg-white/70 backdrop-blur border-slate-200 hover:shadow-lg transition-all duration-300">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-slate-600">Model Size</p>
                    <p className="text-2xl font-bold text-slate-900">
                      {modelInfo.config.d_model}
                    </p>
                  </div>
                  <BarChart3 className="h-8 w-8 text-orange-500" />
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Error Alert */}
        {error && (
          <Alert className="bg-red-50 border-red-200">
            <AlertDescription className="text-red-800">{error}</AlertDescription>
          </Alert>
        )}

        {/* Main Interface */}
        <Tabs defaultValue="predict" className="w-full">
          <TabsList className="grid w-full grid-cols-2 bg-white/50 backdrop-blur">
            <TabsTrigger value="predict" className="data-[state=active]:bg-white">Move Prediction</TabsTrigger>
            <TabsTrigger value="train" className="data-[state=active]:bg-white">Model Training</TabsTrigger>
          </TabsList>

          {/* Move Prediction Tab */}
          <TabsContent value="predict" className="space-y-6">
            <Card className="bg-white/70 backdrop-blur border-slate-200">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Play className="h-5 w-5 text-blue-600" />
                  <span>Chess Position Analysis</span>
                </CardTitle>
                <CardDescription>
                  Enter a chess position in FEN notation to get AI-powered move predictions
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-slate-700">FEN Position</label>
                  <Input
                    value={position}
                    onChange={(e) => setPosition(e.target.value)}
                    placeholder="Enter FEN notation..."
                    className="font-mono text-sm bg-white/80"
                  />
                </div>
                
                <div className="flex space-x-3">
                  <Button
                    onClick={predictMoves}
                    disabled={loading}
                    className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white"
                  >
                    {loading ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent mr-2" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Zap className="h-4 w-4 mr-2" />
                        Predict Moves
                      </>
                    )}
                  </Button>
                  
                  <Button
                    onClick={resetPosition}
                    variant="outline"
                    className="border-slate-300 hover:bg-slate-50"
                  >
                    Reset to Start
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Predictions Results */}
            {predictions && (
              <Card className="bg-white/70 backdrop-blur border-slate-200">
                <CardHeader>
                  <CardTitle>Analysis Results</CardTitle>
                  <CardDescription>
                    Position Value: <Badge variant="outline" className="ml-2">
                      {predictions.position_value > 0 ? '+' : ''}{predictions.position_value.toFixed(3)}
                    </Badge>
                    <span className="ml-4">Legal Moves: {predictions.legal_moves_count}</span>
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <h4 className="font-semibold text-slate-900">Top Move Recommendations</h4>
                    <div className="grid gap-3">
                      {predictions.best_moves.slice(0, 8).map((move, index) => (
                        <div key={index} className="flex items-center justify-between p-3 bg-white/80 rounded-lg border border-slate-200">
                          <div className="flex items-center space-x-3">
                            <Badge className={`${
                              index === 0 ? 'bg-gradient-to-r from-green-500 to-emerald-500' :
                              index === 1 ? 'bg-gradient-to-r from-blue-500 to-cyan-500' :
                              index === 2 ? 'bg-gradient-to-r from-orange-500 to-yellow-500' :
                              'bg-slate-500'
                            } text-white`}>
                              #{index + 1}
                            </Badge>
                            <div>
                              <p className="font-mono font-semibold text-slate-900">{move.san}</p>
                              <p className="text-xs text-slate-500 font-mono">{move.move}</p>
                            </div>
                          </div>
                          <div className="flex items-center space-x-3">
                            <div className="w-20 bg-slate-200 rounded-full h-2">
                              <div
                                className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full"
                                style={{ width: `${(move.probability * 100)}%` }}
                              />
                            </div>
                            <span className="text-sm font-medium text-slate-700 min-w-[60px] text-right">
                              {(move.probability * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* Training Tab */}
          <TabsContent value="train" className="space-y-6">
            <Card className="bg-white/70 backdrop-blur border-slate-200">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Brain className="h-5 w-5 text-purple-600" />
                  <span>Model Training</span>
                </CardTitle>
                <CardDescription>
                  Train the transformer model on chess games in PGN format
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-slate-700">PGN Games Data</label>
                  <Textarea
                    value={trainingData}
                    onChange={(e) => setTrainingData(e.target.value)}
                    placeholder="Paste your PGN games here (one game per paragraph)..."
                    className="min-h-[200px] font-mono text-sm bg-white/80"
                  />
                </div>
                
                <Button
                  onClick={startTraining}
                  disabled={loading || !trainingData.trim()}
                  className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white"
                >
                  {loading ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent mr-2" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Brain className="h-4 w-4 mr-2" />
                      Start Training
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>

            {/* Training Status */}
            {trainingStatus && (
              <Card className="bg-white/70 backdrop-blur border-slate-200">
                <CardHeader>
                  <CardTitle>Training Status</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <p className="text-slate-700">{trainingStatus.message}</p>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-sm text-slate-600">Epochs</p>
                        <p className="font-semibold">{trainingStatus.epochs}</p>
                      </div>
                      <div>
                        <p className="text-sm text-slate-600">Batch Size</p>
                        <p className="font-semibold">{trainingStatus.batch_size}</p>
                      </div>
                    </div>
                    <Badge className="bg-blue-100 text-blue-800 border-blue-200">
                      {trainingStatus.status}
                    </Badge>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>

        {/* Architecture Overview */}
        <Card className="bg-white/70 backdrop-blur border-slate-200">
          <CardHeader>
            <CardTitle>Architecture Overview</CardTitle>
            <CardDescription>
              Chess Transformer with CUDA optimization for move prediction and position evaluation
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center p-6 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl border border-blue-200">
                <Target className="h-8 w-8 text-blue-600 mx-auto mb-3" />
                <h3 className="font-semibold text-slate-900 mb-2">Input Encoding</h3>
                <p className="text-sm text-slate-600">
                  18-channel board representation: pieces, castling rights, en passant, turn
                </p>
              </div>
              
              <div className="text-center p-6 bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl border border-purple-200">
                <Brain className="h-8 w-8 text-purple-600 mx-auto mb-3" />
                <h3 className="font-semibold text-slate-900 mb-2">Transformer Core</h3>
                <p className="text-sm text-slate-600">
                  Multi-head attention with positional encoding for chess pattern recognition
                </p>
              </div>
              
              <div className="text-center p-6 bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl border border-green-200">
                <Zap className="h-8 w-8 text-green-600 mx-auto mb-3" />
                <h3 className="font-semibold text-slate-900 mb-2">Output Heads</h3>
                <p className="text-sm text-slate-600">
                  Policy head for move probabilities with legal move validation
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </main>

      {/* Footer */}
      <footer className="bg-white/50 backdrop-blur border-t border-slate-200 mt-16">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="text-center text-slate-600">
            <p className="text-sm">
              Chess Transformer - Advanced AI for Chess Move Prediction
            </p>
            <p className="text-xs mt-2">
              Built with PyTorch, FastAPI, and React • Optimized for CUDA
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;