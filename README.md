# MURA X-Ray Classification

A deep learning project for classifing X-ray images using the MURA (MusculoSkeletal Radiographs) dataset with pretrained ResNet50 and DenseNet121 models.

## Overview

This project trains deep learning models ResNet50 and DenseNet121 to classify X-ray images as positive or negative for abnormalities. 
It includes:
- **Model training** using PyTorch as dataloader (model.ipynb)
- **Web-based inference apps** using Streamlit for a web interface.

## Features

### Core Features
- **Automated X-Ray Classification**: Binary classification 0for abnormality and 1 for normal with confidence scoring
- **Multiple Model Support**: ResNet50 and DenseNet121 architectures with transfer learning and their variation
- **Web Interface**: User-friendly Streamlit applications for real-time prediction
- **Complete ML Pipeline**: From data loading to model deployment

### Advanced Features
- **Custom Dataset Handler**: Hierarchical folder structure support for MURA dataset
- **Model Comparison**: Side-by-side evaluation of different architectures
- **Performance Monitoring**: Training/validation accuracy tracking with visualization
- **Checkpoint Management**: Automatic saving of best performing models

## Technologies & Tools Used

### Deep Learning & ML
- **PyTorch**: Deep learning framework
- **torchvision**: Pre-trained models and image transforms
- **scikit-learn**: Metrics and evaluation tools

### Web Development
- **Streamlit**: Web application framework
- **PIL (Pillow)**: Image processing

### Data Processing & Visualization
- **NumPy**: Numerical operations
- **Matplotlib**: Data visualization and plotting
- **tqdm**: Progress bars for training loops

### Development Tools
- **Jupyter Notebook**: Interactive development environment
- **Git**: Version control
  

**Basic App Interface:**
- Model selection and image upload
- Real-time prediction display
- Simple, intuitive layout

**Stylish App Interface (Recommended):**
- Professional styled header with instructions
- Model caching for faster interaction
- Enhanced visual formatting of results

## Project Structure

### Notebook Cells (model.ipynb) - Workflow:

| Cell | Function | Description |
|------|----------|-------------|
| **1** | Import Libraries | Loads PyTorch, torchvision, PIL, sklearn, matplotlib, tqdm |
| **2** | First Dataset Class | Basic MuraDataset implementation |
| **3** | Basic Transform | Image preprocessing: resize 224×224, to tensor, normalize |
| **4** | Production Dataset Class | **MURADataset** - hierarchical MURA structure handling |
| **5** | Train & Val Transforms | Data pipelines with ImageNet normalization |
| **6** | Create DataLoaders | Instantiates train_loader & val_loader (batch_size=15) |
| **7** | Initialize Models | Detects GPU/CPU, defines get_model() function |
| **8** | Training Function | Implements train_model() with forward/backward pass |
| **9** | Train Models | Trains ResNet50 & DenseNet121 for 20 epochs |
| **10**| Plot Results | Visualizes training vs validation accuracy curves |
| **11**| Evaluation Function | Defines evaluate_model() for metrics & confusion matrix |
| **12**| Final Evaluation | Loads checkpoints, evaluates on validation set |

### Streamlit Apps Structure:

**app_basic.py** - Simple Interface:
- Model selection (radio button)
- Image upload & preview
- Direct prediction output

**app_stylish.py** - Enhanced Interface (Recommended):
- Model caching (@st.cache_resource)
- Custom styled HTML header
- Professional formatted output

### File Structure:

```
MURA Training/
├── model.ipynb              # Training notebook (12 cells)
├── app_basic.py             # Basic Streamlit web app
├── app_stylish.py           # Enhanced Streamlit web app with styling
├── get_model.py             # Model architecture definitions
├── README.md                # This file
├── checkpoints/
│   ├── resnet50_best.pth    # Trained ResNet50 weights
│   └── densenet121_best.pth # Trained DenseNet121 weights
├── MURA-v1.1/
│   ├── train/               # Training images (7 body part categories)
│   ├── valid/               # Validation images
│   ├── train_image_paths.csv
│   ├── train_labeled_studies.csv
│   ├── valid_image_paths.csv
│   └── valid_labeled_studies.csv
└── __pycache__/
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, but recommended for training)

### Step 1: Clone/Navigate to Project
```bash
cd "E:\MURA Training"
```

### Step 2: Install Dependencies
```bash
# For CPU-only installation
pip install torch torchvision pillow streamlit scikit-learn numpy tqdm matplotlib

# For GPU support (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pillow streamlit scikit-learn numpy tqdm matplotlib
```

### Step 3: Download MURA Dataset
1. Download the MURA dataset from Stanford ML Group
2. Extract to `MURA-v1.1/` directory
3. Ensure the hierarchical folder structure is maintained

## Running the Project

### Training Models

1. **Open Jupyter Notebook**:
   ```bash
   jupyter notebook model.ipynb
   ```

2. **Run Training Cells Sequentially**:
   - Load and preprocess MURA dataset
   - Initialize ResNet50 and DenseNet121 models
   - Train models for 5 epochs
   - Evaluate performance and save best checkpoints

**Training Configuration**:
- Batch Size: 15
- Epochs: 5 (or more for better results)
  
#### Basic Version
```bash
streamlit run app_basic.py
```

**Web App Features**:
- Model selection (ResNet50/DenseNet121)
- Image upload interface
- Real-time prediction display
- Confidence score visualization
  

## Instructions for Testing

### Manual Testing

1. **Model Training Validation**:
   - Run `model.ipynb` completely
   - Verify checkpoint files are created in `checkpoints/`
   - Check training/validation accuracy plots

2. **Web Application Testing**:
   - Start Streamlit app
   - Test with sample X-ray images
   - Verify predictions for both models
   - Check error handling with invalid files

### Test Cases

#### Functional Tests
- **Image Upload**: Test various formats (PNG, JPG, JPEG)
- **Model Selection**: Switch between ResNet50 and DenseNet121
- **Prediction Accuracy**: Validate against known test cases
- **Error Handling**: Test with corrupted/invalid images

#### Performance Tests
- **Response Time**: Predictions should complete within 3 seconds
- **Memory Usage**: Monitor RAM usage during inference
- **Concurrent Users**: Test multiple simultaneous sessions

### Sample Test Images
Use images from the MURA validation set:
```
MURA-v1.1/valid/XR_HAND/patient11185/study1_positive/image1.png
MURA-v1.1/valid/XR_WRIST/patient11349/study1_negative/image1.png
```

## Model Performance

### Expected Results
- **Validation Accuracy**: >85%
- **Training Time**: ~3 hours on GPU
- **Inference Time**: <3 seconds per image

### Evaluation Metrics
- Binary classification accuracy
- Confusion matrix analysis
- Training vs validation loss curves

## Troubleshooting

### Common Issues

**Port Already in Use**:
```bash
streamlit run app_stylish.py --server.port 8502
```

**GPU Not Detected**:
- App directly starts training with CPU
- For manual GPU use: Set `device = "cuda"` in app files for enabling GPU usage

**Missing Checkpoint Files**:
- Ensure training completed successfully
- Check `checkpoints/` directory exists
- Verify `.pth` files are not corrupted(for best model validation and ploting graphs for comparision

### Error Messages
- **"CUDA out of memory"**: Reduce batch size.
- **"File not found"**: Check dataset path where i faced challenging in Google Collab so i used VS Code for easier approach 
- **"Invalid image format"**: Ensure image is PNG/JPG/JPEG


## Documentation

- **[Project Statement](statement.md)**: Problem definition and scope
- **[Project Report](PROJECT_REPORT.md)**: Technical data structure
- **[System Design](SYSTEM_DESIGN.md)**: Architecture and UML diagrams

## Dataset Information

### MURA Dataset
- **Total Images**: ~40000 musculoskeletal radiographs
- **Body Parts**: 7 categories (elbow, finger, forearm, hand, humerus, shoulder, wrist)
- **Classification**: Binary (normal 1 vs abnormal 0)
- **Format**: PNG images with variable resolution

### Data Preprocessing
- Resize to 224×224 pixels
- Convert to RGB tensor
- Normalize with ImageNet statictics
- Mean: [0.485, 0.456, 0.406]
- Std: [0.229, 0.224, 0.225]

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## License

This project is based on the MURA dataset. Please refer to the original dataset documentation for usage rights and restrictions.

## Contact

For questions or issues, please create an issue in the GitHub repository or contact the development team.

---

**Note**: This project is designed for educational and research purposes. For clinical applications, additional validation and regulatory approval would be required.All models use ImageNet pre-trained weights as initialization
- Batch processing is not supported in the current web app version
- Model inference runs on the selected device (GPU/CPU) automatically
