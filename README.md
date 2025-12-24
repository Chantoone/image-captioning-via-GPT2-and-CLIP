# Image Captioning via CLIP and GPT2

## 📋 Project Overview

This project implements an advanced **Image Captioning** system that automatically generates natural language descriptions for images. It leverages two state-of-the-art pre-trained models:

- **CLIP** (Contrastive Language-Image Pre-training) by OpenAI: Encodes images into semantic embeddings
- **GPT2**: Generates textual descriptions from image embeddings

The system uses a **Prefix-Tuning** approach with a trainable **Mapping Network** to bridge the gap between image and text embeddings, enabling efficient fine-tuning without modifying the frozen base models.

---

## 🏗️ Architecture

### System Architecture

```
Input Image
    ↓
[CLIP Visual Encoder] → Image Embedding (512-dim)
    ↓
[Mapping Network - Trainable]
  ├─ Linear Projection: 512 → 768
  ├─ Transformer Decoder (8 layers, 8 heads)
  └─ Prefix Embedding Generation (10 tokens)
    ↓
[Concatenation] Prefix + Caption Embeddings
    ↓
[GPT2 - Frozen] → Loss Computation (Training)
                  → Text Generation (Inference)
    ↓
Generated Caption
```

### Key Components

#### 1. **FlickrDataset Class**
- Loads images from Flickr8k dataset
- Tokenizes captions with special tokens: `<|startoftext|>` and `<|endoftext|>`
- Supports random caption selection during training for data augmentation
- Image preprocessing using CLIP's preprocess function

#### 2. **MappingNetwork Class**
A learnable bridge between CLIP and GPT2:
- **Input**: CLIP image embeddings (512-dim)
- **Output**: Prefix embeddings (batch_size, 10, 768)
- **Components**:
  - Linear projection layer (512 → 768)
  - Transformer Decoder (8 layers, 8 attention heads)
  - Learnable prefix constants

#### 3. **ClipCapModel Class**
Main training architecture:
- Combines CLIP (frozen) + MappingNetwork (trainable) + GPT2 (frozen)
- Concatenates prefix embeddings with caption embeddings
- Computes language modeling loss using GPT2's language head
- Ignores prefix tokens in the loss calculation (only trains on caption tokens)

---

## 📊 Evaluation Results

### Performance Metrics

| Metric | Score | Description |
|--------|-------|-------------|
| **CIDEr** | **0.2685** | Evaluates semantic similarity between generated and reference captions |
| **SPICE** | **0.1045** | Measures syntactic and semantic correctness |

### Metrics Explanation

- **CIDEr (Consensus-based Image Description Evaluation)**: Measures n-gram overlap and TF-IDF weighting. Higher values indicate better semantic similarity to reference captions.
- **SPICE (Semantic Propositional Image Caption Evaluation)**: Uses syntactic parsing to evaluate semantic relationships. Focuses on matching objects, attributes, and relationships.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for faster training)
- 8GB+ GPU memory

### Installation

```bash
# Clone or download the project
cd "Image captioning"

# Install required packages
pip install -q --upgrade "pyarrow>=21.0.0"
pip install -q "pydantic>=2.0,<2.12"
pip install -q transformers ftfy regex tqdm
pip install -q git+https://github.com/openai/CLIP.git
pip install -q matplotlib pandas pillow
pip install -q git+https://github.com/salaniz/pycocoevalcap
```

### Dataset

The project uses the **Flickr8k** dataset:
- 8,000 images
- 40,000 captions (5 captions per image)
- Train/Val/Test split: 80% / 10% / 10%

Download from: [Flickr8k Dataset](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip)

---

## 🎯 Usage

### Training

```python
# Configuration
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5

# The training loop handles:
# - Forward pass through mapping network and GPT2
# - Backpropagation through mapping network only
# - Evaluation on validation set
# - Checkpointing after each epoch
```

### Inference - Single Image

```python
def generate_caption(image_path, model, max_length=40, num_beams=5):
    """Generate caption for a single image"""
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image_processed = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_embedding = clip_model.encode_image(image_processed).float()
        prefix_embeddings = model.mapping_network(image_embedding)
        output_ids = gpt2_model.generate(
            inputs_embeds=prefix_embeddings,
            max_length=max_length,
            num_beams=num_beams,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            early_stopping=True
        )
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return caption

# Load trained model
model.mapping_network.load_state_dict(torch.load("mapping_network_epoch_30.pth"))
caption = generate_caption("path/to/image.jpg", model)
print(f"Generated: {caption}")
```

### Inference - Batch Images

```python
def generate_caption_batch(image_batch, model, max_length=40, num_beams=5):
    """Generate captions for a batch of images"""
    model.eval()
    
    with torch.no_grad():
        image_embeddings = clip_model.encode_image(image_batch).float()
        prefix_embeddings = model.mapping_network(image_embeddings)
        output_ids = gpt2_model.generate(
            inputs_embeds=prefix_embeddings,
            max_length=max_length,
            num_beams=num_beams,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            early_stopping=True
        )
        captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    
    return captions
```

### Evaluation

```python
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

# Generate captions for validation set
gts = {}  # Ground truth captions
res = {}  # Generated captions

# Compute scores
cider_scorer = Cider()
cider_score, _ = cider_scorer.compute_score(gts, res)

spice_scorer = Spice()
spice_score, _ = spice_scorer.compute_score(gts, res)

print(f"CIDEr: {cider_score:.4f}")
print(f"SPICE: {spice_score:.4f}")
```

---

## 📁 Project Structure

```
image-captioning-via-clip-and-gpt2/
├── image-captioning-via-clip-and-gpt2.ipynb  # Main notebook
├── README.md                                  # This file
├── mapping_network_epoch_*.pth               # Saved model checkpoints
└── (dataset files - Flickr8k)
    ├── Images/                               # Image directory
    └── captions.txt                          # Caption file
```

---

## 💡 Key Features

✅ **Efficient Fine-tuning**: Only trains the small Mapping Network (~100K parameters) while keeping CLIP and GPT2 frozen

✅ **Prefix-Tuning Approach**: Adds learnable prefix embeddings to the GPT2 input, enabling fast adaptation

✅ **Transformer-based Mapping**: Uses a Transformer Decoder to intelligently process image embeddings

✅ **Beam Search Decoding**: Generates high-quality captions using beam search with 5 beams

✅ **Comprehensive Evaluation**: Evaluates using both CIDEr and SPICE metrics for semantic and syntactic quality

✅ **Data Augmentation**: Randomly selects from multiple reference captions during training

---

## 🔧 Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Epochs | 30 | Number of training epochs |
| Batch Size | 32 | Training batch size |
| Learning Rate | 1e-5 | Adam optimizer learning rate |
| Weight Decay | 1e-5 | L2 regularization |
| Max Sequence Length | 40 | Maximum caption length |
| Prefix Length | 10 | Number of learnable prefix tokens |
| Transformer Layers | 8 | Layers in Mapping Network |
| Attention Heads | 8 | Attention heads in Transformer |
| Beam Size | 5 | Beam search width for inference |

---

## 📈 Training Details

### Model Components Size

| Component | Size | Trainable |
|-----------|------|-----------|
| CLIP (ViT-B/32) | ~340M | ❌ No |
| GPT2 | ~124M | ❌ No |
| Mapping Network | ~100K | ✅ Yes |
| **Total Trainable** | **~100K** | - |

### Training Characteristics

- **Device**: GPU (CUDA) or CPU
- **Optimizer**: AdamW with weight decay
- **Loss Function**: Language Modeling Loss (Cross-Entropy)
- **Gradient Computation**: Only through Mapping Network
- **Memory Efficient**: Frozen backbone models significantly reduce memory requirements

---

## 📚 References

### Papers

- **CLIP**: [Learning Transferable Models for Unsupervised Learning](https://arxiv.org/abs/2103.00020)
- **GPT2**: [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
- **Prefix-Tuning**: [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)
- **Evaluation Metrics**: 
  - CIDEr: [CIDEr: Consensus-based Image Description Evaluation](https://arxiv.org/abs/1411.5726)
  - SPICE: [SPICE: Semantic Propositional Image Caption Evaluation](https://arxiv.org/abs/1607.08949)

### Datasets

- **Flickr8k**: [Flickr8k Dataset on GitHub](https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k)

---

## 🎓 How It Works

### Training Process

1. **Image Encoding**: CLIP encodes input images to 512-dimensional vectors
2. **Mapping**: Mapping Network projects and processes embeddings → 10 prefix tokens (768-dim)
3. **Caption Embedding**: GPT2's embedding layer converts tokens to embeddings
4. **Concatenation**: Combine prefix embeddings + caption embeddings
5. **Loss Computation**: GPT2 computes language modeling loss (ignoring prefix tokens)
6. **Backpropagation**: Only Mapping Network parameters are updated

### Inference Process

1. **Image Encoding**: Extract image embeddings using CLIP
2. **Prefix Generation**: Pass through Mapping Network to generate prefix
3. **Text Generation**: Use GPT2's generate() method with beam search
4. **Decoding**: Tokenizer decodes output token IDs to readable caption

---

## 🚨 Potential Improvements

1. **Data Augmentation**: Implement more sophisticated augmentation techniques
2. **Hyperparameter Tuning**: Experiment with different prefix lengths, learning rates
3. **Model Architecture**: Try different attention mechanisms or fusion strategies
4. **Training Strategy**: Implement learning rate scheduling or gradient accumulation
5. **Evaluation**: Use additional metrics like BLEU, METEOR, ROUGE
6. **Ensemble Methods**: Combine multiple checkpoints for better results

---

## 📝 Notes

- The model is trained only on image-caption pairs and does not require manual annotation
- Special tokens (`<|startoftext|>`, `<|endoftext|>`, `[PAD]`) are added to the tokenizer
- Prefix tokens are masked during loss calculation to focus training on caption generation
- The model uses 10 prefix tokens by default, which provides a good balance between capacity and efficiency

---

## 📧 Contact & Support

For questions, issues, or suggestions about this project, feel free to reach out.

---

**Last Updated**: December 2024

**Model Version**: Final (Epoch 30)

**Framework**: PyTorch

**License**: MIT