# Sperm Detection Dataset - Labelbox Uploader

This repository contains tools for uploading sperm detection datasets with YOLO annotations to Labelbox for annotation and model training workflows.

## 🚀 Features

- **Batch Upload**: Upload multiple images with YOLO annotations to Labelbox
- **Checkpoint Support**: Resume interrupted uploads with checkpoint functionality
- **Multiple Import Modes**: Support for both Model-Assisted Labeling (MAL) and Ground Truth import
- **Flexible Configuration**: Environment-based configuration for secure API key management
- **Error Handling**: Robust error handling and retry mechanisms

## 📁 Project Structure

```
├── src/
│   └── labelbox_uploader.py      # Main uploader module
├── examples/
│   ├── simple_uploader.py        # Simple single-image uploader
│   ├── batch_uploader.py         # Batch image uploader
│   ├── bulk_uploader_with_checkpoint.py  # Bulk uploader with resume capability
│   ├── advanced_uploader.py      # Advanced uploader with full features
│   ├── single_image_uploader.py  # Basic single image example
│   └── legacy_uploader.py        # Legacy implementation
├── docs/
│   └── labelbox_reference.py     # Labelbox API reference examples
├── env.example                   # Environment variables template
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## 🛠️ Setup

### Prerequisites

- Python 3.8+
- Labelbox account and API key
- YOLO-formatted annotation files

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd sperm-labelbox-uploader
   ```

2. **Install dependencies**
   ```bash
   pip install labelbox pillow tqdm
   ```

3. **Configure environment**
   ```bash
   cp env.example .env
   # Edit .env with your actual values
   ```

### Environment Configuration

Copy `env.example` to `.env` and configure the following variables:

```bash
# Required
LABELBOX_API_KEY=your_labelbox_api_key_here
PROJECT_ID=your_project_id_here
DATA_FOLDER=/path/to/your/data/folder

# Optional
DATASET_NAME=Sperm YOLO Dataset
BATCH_SIZE=100
IMPORT_MODE=MAL  # or GT for ground truth
SKIP_UNLABELED=False
CHUNK_SIZE=2000
TOOL_NAME=sperm
```

## 📊 Data Format

### Expected Directory Structure
```
data_folder/
├── image1.jpg
├── image1.txt      # YOLO annotations for image1.jpg
├── image2.jpg
├── image2.txt      # YOLO annotations for image2.jpg
└── ...
```

### YOLO Annotation Format
Each `.txt` file should contain bounding box annotations in YOLO format:
```
class_id x_center y_center width height
```
- All values are normalized (0.0 to 1.0)
- `class_id`: 0 for sperm class
- Coordinates are relative to image dimensions

## 🚀 Usage

### Basic Usage

```python
from src.labelbox_uploader import main
import os

# Set environment variables
os.environ['LABELBOX_API_KEY'] = 'your_api_key'
os.environ['PROJECT_ID'] = 'your_project_id'

# Run the uploader
main()
```

### Command Line Usage

```bash
python src/labelbox_uploader.py \
    --data_dir /path/to/data \
    --project_id your_project_id \
    --dataset_name "Sperm Dataset" \
    --import_mode ground_truth
```

### Available Examples

1. **Simple Upload** (`examples/simple_uploader.py`)
   - Basic single image upload
   - Good for testing setup

2. **Batch Upload** (`examples/batch_uploader.py`)
   - Upload multiple images at once
   - Suitable for medium datasets

3. **Bulk Upload with Checkpoint** (`examples/bulk_uploader_with_checkpoint.py`)
   - Large dataset support
   - Resume capability for interrupted uploads
   - Progress tracking

## 🔧 Configuration Options

### Import Modes

- **MAL (Model-Assisted Labeling)**: Import as pre-labels for human review
- **GT (Ground Truth)**: Import as final annotations

### Batch Processing

- Configure `CHUNK_SIZE` for optimal performance
- Larger chunks = faster upload but more memory usage
- Recommended: 1000-5000 for large datasets

### Error Handling

- Automatic retry with exponential backoff
- Checkpoint system prevents data loss
- Detailed error logging

## 📝 Labelbox Project Setup

### Required Ontology

Your Labelbox project must have:
1. **Bounding Box tool** named `"sperm"`
2. Proper project configuration for your media type

### Creating the Ontology

```python
import labelbox as lb

ontology_builder = lb.OntologyBuilder(
    tools=[
        lb.Tool(tool=lb.Tool.Type.BBOX, name="sperm")
    ]
)

ontology = client.create_ontology(
    "Sperm Detection Ontology",
    ontology_builder.asdict(),
    media_type=lb.MediaType.Image
)
```

## 🔒 Security

- **Never commit API keys** to version control
- Use environment variables or `.env` files
- Add `.env` to `.gitignore`
- Rotate API keys regularly

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**
   - Verify your API key is correct
   - Check key permissions in Labelbox

2. **Upload Failures**
   - Check internet connection
   - Verify file paths are correct
   - Ensure sufficient disk space

3. **Annotation Errors**
   - Verify YOLO format is correct
   - Check class mappings match ontology
   - Ensure coordinates are normalized

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Performance Tips

- Use appropriate batch sizes (50-200 images)
- Enable checkpointing for large uploads
- Monitor memory usage with large datasets
- Use SSD storage for better I/O performance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review Labelbox documentation
3. Open an issue in this repository

## 🔗 Related Links

- [Labelbox Documentation](https://docs.labelbox.com/)
- [YOLO Format Specification](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
- [Labelbox Python SDK](https://github.com/Labelbox/labelbox-python)
