# CarID Classification Service

A self-hosted image classification service using OpenCLIP embeddings and FAISS indexing with Google Cloud Storage for image management.

## 🚗 **Features**

- **Multi-class Classification**: Identify categories and subcategories from images
- **Negative Sample Learning**: Reject irrelevant images and unknown categories  
- **Prototype Mode**: Efficient per-class centroids or per-image indexing
- **Google Cloud Storage**: Scalable cloud storage for training images
- **Home Assistant Integration**: Native addon support
- **RESTful API**: Easy integration with existing systems

## 🏗️ **Architecture**

- **OpenCLIP**: Vision transformer for image embeddings
- **FAISS**: Fast similarity search and clustering
- **FastAPI**: Modern web framework for the API
- **Google Cloud Storage**: Cloud-native image storage
- **Docker**: Containerized deployment

## 🚀 **Quick Start**

### **Home Assistant Addon**

1. **Add Repository**: Add this repository to Home Assistant
2. **Install Addon**: Install "CarID (OpenCLIP retrieval)"
3. **Configure GCS**: Set your bucket name and service account credentials
4. **Start Service**: The API will be available at `http://homeassistant.local:8001`

See [ADDON_SETUP.md](ADDON_SETUP.md) for detailed instructions.

### **Docker Compose**

1. **Clone Repository**:
   ```bash
   git clone <repository-url>
   cd carid
   ```

2. **Set up GCS Credentials**:
   ```bash
   # Create .env file
   cp .env.example .env
   
   # Edit with your GCS settings
   GCS_BUCKET_NAME=your-bucket-name
   GCS_SERVICE_ACCOUNT_PATH=/path/to/service-account.json
   ```

3. **Start Service**:
   ```bash
   docker-compose up -d
   ```

## 📊 **API Usage**

### **Health Check**
```bash
curl http://localhost:8001/healthz
```

### **Add Training Image**
```bash
curl -X POST "http://localhost:8001/index/add" \
  -H "Content-Type: application/json" \
  -d '{
    "label": "category_a",
    "image_b64": "data:image/jpeg;base64,/9j/4AAQ...",
    "is_negative": false
  }'
```

### **Classify Image**
```bash
curl -X POST "http://localhost:8001/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "image_b64": "data:image/jpeg;base64,/9j/4AAQ..."
  }'
```

### **Rebuild Index**
```bash
curl -X POST "http://localhost:8001/index/rebuild"
```

## ⚙️ **Configuration**

| Variable | Default | Description |
|----------|---------|-------------|
| `GCS_BUCKET_NAME` | `carid-trained-images` | Google Cloud Storage bucket name |
| `GCS_CREDENTIALS_PATH` | - | Path to service account JSON file |
| `ACCEPT_THRESHOLD` | `0.80` | Minimum similarity for classification |
| `MARGIN_THRESHOLD` | `0.04` | Minimum margin over second-best match |
| `NEG_ACCEPT_CAP` | `0.80` | Maximum negative similarity threshold |
| `PROTOTYPE_MODE` | `true` | Use class centroids vs per-image index |
| `DEVICE` | `cpu` | Processing device (`cpu` or `cuda`) |

## 🗂️ **Data Organization**

Images are organized in GCS with this structure:

```
gs://your-bucket/
├── category_a/          # Positive samples
│   ├── image1.jpg
│   └── image2.jpg
├── category_b/          # Positive samples
│   ├── image1.jpg
│   └── image2.jpg
└── _negative/           # Negative samples
    ├── sample1.jpg
    └── sample2.jpg
```

## 🔧 **Advanced Usage**

### **Prototype Mode vs Per-Image Mode**

- **Prototype Mode** (`PROTOTYPE_MODE=true`): Creates a single centroid per class. Fast, memory-efficient, good for clean datasets.
- **Per-Image Mode** (`PROTOTYPE_MODE=false`): Stores every training image. More accurate but uses more memory.

### **Negative Sampling**

Add negative examples (irrelevant images, unknown categories) to folders starting with `_`:
- `_negative/` - General negative samples
- `_unknown/` - Unknown categories

### **Fine-tuning Thresholds**

- **`ACCEPT_THRESHOLD`**: Higher = stricter classification
- **`MARGIN_THRESHOLD`**: Higher = requires clearer distinction between classes
- **`NEG_ACCEPT_CAP`**: Lower = more aggressive negative rejection

## 🛠️ **Development**

### **Local Development**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r service/requirements.txt

# Set environment variables
export GCS_BUCKET_NAME=your-bucket
export GCS_CREDENTIALS_PATH=/path/to/service-account.json

# Run service
cd service
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8001
```

### **Adding New Models**

1. **Upload Training Images**: Use `/index/add` endpoint
2. **Rebuild Index**: Call `/index/rebuild` 
3. **Test Classification**: Use `/classify` endpoint
4. **Tune Thresholds**: Adjust configuration as needed

## 📈 **Performance**

- **CPU Mode**: ~100ms per classification on modern CPU
- **GPU Mode**: ~20ms per classification with CUDA GPU
- **Memory Usage**: ~500MB base + ~1MB per 1000 training images
- **Storage**: Images stored efficiently in GCS with JPEG compression

## 🔐 **Security**

- **Service Account**: Use dedicated GCS service account with minimal permissions
- **API Token**: Optional API token authentication
- **Network**: Run behind reverse proxy (nginx, Cloudflare, etc.)
- **Credentials**: Never commit service account keys to version control

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 **Contributing**

Contributions welcome! Please read the contributing guidelines and submit pull requests.

## 📞 **Support**

- **Documentation**: See [ADDON_SETUP.md](ADDON_SETUP.md)
- **Issues**: Open GitHub issues for bugs and feature requests
- **API Docs**: Visit `http://localhost:8001/docs` when running
