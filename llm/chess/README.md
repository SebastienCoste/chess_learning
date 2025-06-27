# Chess LLM Training Project

This project implements a complete pipeline for training a Large Language Model (Mistral-7B) to play chess using real PGN game data. The trained model can generate chess moves and play interactive games with move validation and board visualization.

## 🚀 Features

- **Real Chess Data Collection**: Downloads 10,000+ real chess games from public sources (TWIC, Lichess)
- **LLM Fine-tuning**: Fine-tunes Mistral-7B using LoRA for efficient training
- **Interactive Chess Playing**: Play against the trained model with full move validation
- **Board Visualization**: ASCII chess board display with game state tracking
- **Move Generation**: AI generates 10 candidate moves and selects the first valid one
- **Local Deployment**: Complete local setup, no external API dependencies

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended: 16GB+ VRAM)
- 50GB+ free disk space
- Git and internet connection for initial setup

## 🛠️ Installation

1. **Clone or download the project files**:
   ```bash
   # All files should be in the same directory:
   # - 1_download_chess_data.py
   # - 2_prepare_training_data.py  
   # - 3_train_chess_llm.py
   # - 4_play_chess.py
   # - requirements.txt
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Additional chess library setup** (if needed):
   ```bash
   pip install chess python-chess
   ```

## 📊 Usage Pipeline

### Step 1: Download Chess Data

Download 10,000 real chess games from various sources:

```bash
python download_chess_data.py
```

**What it does**:
- Downloads games from TWIC (The Week in Chess)
- Attempts to collect from Lichess API
- Generates sample games if needed
- Outputs: `chess_games_10k.pgn`

**Expected output**: A PGN file with 10,000 chess games (~50-100MB)

### Step 2: Prepare Training Data

Convert PGN games into LLM training format:

```bash
python prepare_training_data.py
```

**What it does**:
- Parses PGN games using python-chess
- Creates move prediction and position analysis examples
- Formats data for Mistral fine-tuning
- Splits into train/validation sets (80/20)
- Outputs: `chess_train.jsonl`, `chess_val.jsonl`

**Expected output**: 
- ~40,000+ training examples
- ~10,000+ validation examples

### Step 3: Train the Model

Fine-tune Mistral-7B on chess data:

```bash
python train_chess_llm.py
```

**What it does**:
- Downloads Mistral-7B-Instruct model
- Applies 4-bit quantization for memory efficiency
- Uses LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- Trains for 3 epochs with validation
- Outputs: `./chess-mistral-7b-lora/` directory with trained model

**Training time**: 2-8 hours depending on GPU

**Memory requirements**:
- Training: ~16GB GPU memory
- Inference: ~8GB GPU memory

### Step 4: Play Chess!

Start an interactive chess game against your trained model:

```bash
python play_chess.py
```

**Game features**:
- ASCII board visualization
- Standard algebraic notation input (e.g., "e4", "Nf3", "O-O")
- AI generates 10 move candidates and picks first valid one
- Full move validation using python-chess
- Game state tracking and endgame detection

**Example gameplay**:
```
Welcome to Chess vs AI!
Enter moves in standard algebraic notation (e.g., e4, Nf3, O-O)

Current Position:
==================================================
   a b c d e f g h
8  r n b q k b n r  8
7  p p p p p p p p  7
6  . . . . . . . .  6
5  . . . . . . . .  5
4  . . . . . . . .  4
3  . . . . . . . .  3
2  P P P P P P P P  2
1  R N B Q K B N R  1
   a b c d e f g h

Your turn (White). Legal moves: 20
Enter your move: e4

AI is thinking...
Generated candidates: ['e5', 'Nf6', 'd6', 'c5', 'Nc6']...
AI plays: e5
```

## 🎮 Game Commands

During gameplay, you can use:
- `moves` - Show available legal moves
- `help` - Display command help
- `quit` - Exit the game

## 📁 Project Structure

```
chess-llm-project/
├── download_chess_data.py      # Download 10K chess games
├── prepare_training_data.py    # Convert PGN to training data
├── train_chess_llm.py         # Fine-tune Mistral-7B
├── play_chess.py             # Interactive chess game
├── requirements.txt          # Python dependencies
├── chess_games_10k.pgn      # Downloaded chess games (generated)
├── chess_train.jsonl        # Training data (generated)
├── chess_val.jsonl          # Validation data (generated)
└── chess-mistral-7b-lora/   # Trained model (generated)
    ├── adapter_config.json
    ├── adapter_model.bin
    └── training_config.json
```

## ⚙️ Configuration Options

### Download Script Configuration
```python
# In download_chess_data.py
downloader = ChessPGNDownloader(target_games=10000)  # Adjust number of games
```

### Training Configuration
```python
# In train_chess_llm.py
class ChessLLMTrainer:
    def __init__(self, 
                 model_name="mistralai/Mistral-7B-Instruct-v0.2",  # Base model
                 output_dir="./chess-mistral-7b-lora"):           # Output directory
```

### LoRA Configuration
```python
lora_config = LoraConfig(
    r=16,           # Rank (higher = more parameters)
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,  # Dropout rate
)
```

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size in `train_chess_llm.py`:
```python
per_device_train_batch_size=2,  # Reduce from 4
gradient_accumulation_steps=8,  # Increase to maintain effective batch size
```

**2. Model Not Found**
```
Model not found at ./chess-mistral-7b-lora
```
**Solution**: Run the training script first, or check the model path.

**3. PGN Parsing Errors**
```
Error reading game: Invalid PGN format
```
**Solution**: The script handles this automatically and skips invalid games.

**4. Slow Download**
```
Connection timeout downloading TWIC
```
**Solution**: The script will fall back to generating sample games.

### Performance Optimization

**For Training**:
- Use mixed precision: `fp16=True` (default)
- Enable gradient checkpointing (default)
- Use DataLoader with multiple workers

**For Inference**:
- Load model in 8-bit mode for faster inference
- Use smaller sequence lengths
- Batch multiple moves if needed

## 📈 Model Performance

**Expected Training Metrics**:
- Training Loss: ~0.8-1.2 (final)
- Validation Loss: ~0.8-1.3 (final)
- Token Accuracy: ~70-80%

**Chess Playing Strength**:
- Beginner to intermediate level (~800-1200 ELO estimated)
- Knows basic chess rules and common patterns
- Makes mostly legal moves with occasional tactical errors

## 🔬 Advanced Usage

### Custom Training Data
To use your own PGN files:
```python
# In prepare_training_data.py
processor = ChessDataProcessor(pgn_file="your_games.pgn")
```

### Different Base Models
To use a different base model:
```python
# In train_chess_llm.py
trainer = ChessLLMTrainer(
    model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"  # Larger model
)
```

### Extended Training
For longer training:
```python
# In train_chess_llm.py, modify TrainingArguments:
num_train_epochs=5,     # More epochs
learning_rate=1e-4,     # Lower learning rate
```

## 📚 Technical Details

### Data Sources
- **TWIC (The Week in Chess)**: Weekly chess game collections
- **Lichess Database**: Open-source chess games
- **Generated Samples**: Fallback high-quality games

### Model Architecture
- **Base**: Mistral-7B-Instruct-v0.2 (7 billion parameters)
- **Fine-tuning**: LoRA with rank-16 adapters
- **Quantization**: 4-bit NF4 for memory efficiency
- **Training**: Causal language modeling objective

### Move Generation
1. Format current game state as text prompt
2. Generate text completion using fine-tuned model
3. Extract chess moves from generated text
4. Validate moves using python-chess library
5. Select first valid move from candidates

## 🤝 Contributing

Feel free to improve this project:
- Add stronger chess engines for comparison
- Implement different neural architectures
- Add support for chess variants
- Improve move generation algorithms
- Add GUI interface

## 📄 License

This project is for educational purposes. Please respect the licenses of:
- Mistral AI models
- Chess data sources (TWIC, Lichess)
- Python libraries used

## 🙏 Acknowledgments

- **Mistral AI** for the base language model
- **TWIC** for providing free chess game data
- **Lichess** for open chess data and API
- **python-chess** library for chess logic
- **Hugging Face** for transformers and PEFT libraries

---

**Happy Chess Playing! 🏁♟️**