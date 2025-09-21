# Home Assistant Addon Setup Guide

This guide explains how to configure the CarID classification service as a Home Assistant addon. The service uses Google Cloud Storage for image storage.

## 🏠 **Home Assistant Addon Configuration**

### **Step 1: Install the Addon**

1. Add this repository as a Home Assistant addon repository
2. Install the "CarID (OpenCLIP retrieval)" addon
3. Don't start it yet - configure it first

### **Step 2: Configure GCS Settings**

In the addon configuration UI, you'll see these options:

```yaml
# Retrieval Settings
accept_threshold: 0.80
margin_threshold: 0.04
neg_accept_cap: 0.80
prototype_mode: true

# Google Cloud Storage (mandatory)
gcs_bucket_name: "carid-trained-images" 
gcs_service_account_json: ""  # See Step 3 below
```

### **Step 3: Add Service Account Credentials**

**Option A: Base64-Encoded JSON (Recommended)**

1. **Get your service account JSON file** from Google Cloud Console
2. **Encode it to base64:**
   ```bash
   # On macOS/Linux:
   base64 -i /path/to/service-account.json
   
   # On Windows:
   certutil -encode service-account.json temp.b64 && findstr /v /c:- temp.b64
   ```
3. **Copy the entire base64 string** (it will be very long)
4. **Paste it into the `gcs_service_account_json` field** in the addon config UI

**Option B: File-based (Alternative)**

1. Place your `service-account.json` file in `/config/carid/`
2. Leave `gcs_service_account_json` empty
3. The addon will automatically look for credentials in `/config/carid/service-account.json`

### **Step 4: Configure Bucket Access**

Make sure your service account has `Storage Admin` permission on your GCS bucket.

### **Step 5: Start the Addon**

1. **Save the configuration**
2. **Start the addon**
3. **Check the logs** to ensure GCS connection is working

## 📁 **File Locations in Home Assistant**

```
/config/carid/          # Optional config directory
└── service-account.json  # Alternative credential location

/data/options.json        # Addon configuration (auto-generated)

GCS Bucket Structure:     # All images stored in Google Cloud Storage
├── mazda_cx5/            # Car model folders  
├── skoda_octavia/        # Car model folders
├── _negative/            # Negative examples
└── ...
```

## 🔧 **Addon Configuration Examples**

### **Minimal Configuration**
```yaml
gcs_bucket_name: "your-bucket-name"
gcs_service_account_json: "ewogICJ0eXBlIjogInNlcnZpY2VfYWNjb3VudCIsCiAg..."
```

### **Full Configuration**
```yaml
# Fine-tune classification thresholds
accept_threshold: 0.80
margin_threshold: 0.04
neg_accept_cap: 0.80
prototype_mode: true

# GCS Configuration (mandatory)
gcs_bucket_name: "carid-trained-images"
gcs_service_account_json: "ewogICJ0eXBlIjogInNlcnZpY2VfYWNjb3VudCIsCiAg..."
```


## 📤 **Adding Images to GCS**

To populate your GCS bucket with training images:

1. **Use the API to add images:**
   ```bash
   curl -X POST "http://homeassistant.local:8001/index/add" \
     -H "Content-Type: application/json" \
     -d '{
       "label": "mazda_cx5",
       "image_b64": "data:image/jpeg;base64,/9j/4AAQ...",
       "is_negative": false
     }'
   ```

2. **Or upload directly to GCS** and rebuild the index:
   ```bash
   curl -X POST "http://homeassistant.local:8001/index/rebuild"
   ```

## 🔍 **Troubleshooting**

### **"Authentication failed" errors**
- Verify your base64-encoded JSON is complete and correct
- Check that your service account has proper permissions
- Ensure the JSON hasn't been corrupted during copy/paste

### **"Bucket not found" errors**
- Verify the bucket name is correct
- Ensure your service account has access to the bucket
- Check that the bucket exists in the correct GCP project

## 📊 **Monitoring in Home Assistant**

The addon provides these endpoints for monitoring:

- **Health Check**: `http://homeassistant.local:8001/healthz`
- **API Documentation**: `http://homeassistant.local:8001/docs`
- **Index Statistics**: `http://homeassistant.local:8001/index/stats`

You can create Home Assistant sensors to monitor these endpoints.

## 💰 **Cost Optimization**

For Home Assistant use:

1. **Use Regional Storage** in your closest region
2. **Monitor Usage** via GCP Console
3. **Consider Nearline/Coldline** for archival storage

Typical costs for personal use: **$1-5/month** depending on image volume.

## 🔐 **Security Best Practices**

1. **Least Privilege**: Only grant necessary GCS permissions
2. **Dedicated Service Account**: Create a specific service account for this addon
3. **Regular Rotation**: Rotate service account keys periodically
