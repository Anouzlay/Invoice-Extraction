# Deploying to Render

This guide will help you deploy your Streamlit Invoice Extraction app to Render.

## Prerequisites

1. A Render account (sign up at https://render.com)
2. Your code pushed to a Git repository (GitHub, GitLab, or Bitbucket)
3. Your OpenAI API key

## Deployment Steps

### Option 1: Using render.yaml (Recommended)

1. **Push your code to Git**
   - Make sure all your files are committed and pushed to your repository
   - The `render.yaml` file is already configured in this repository

2. **Create a new Web Service on Render**
   - Go to your Render dashboard: https://dashboard.render.com
   - Click "New +" → "Web Service"
   - Connect your Git repository
   - Render will automatically detect the `render.yaml` file

3. **Configure Environment Variables**
   - In the Render dashboard, go to your service settings
   - Navigate to "Environment" section
   - Add the following environment variable:
     - Key: `OPENAI_API_KEY`
     - Value: Your OpenAI API key (starts with `sk-`)

4. **Deploy**
   - Click "Create Web Service"
   - Render will automatically build and deploy your app
   - The deployment process may take 5-10 minutes (especially for EasyOCR model downloads)

### Option 2: Manual Setup (Without render.yaml)

1. **Create a new Web Service**
   - Go to https://dashboard.render.com
   - Click "New +" → "Web Service"
   - Connect your Git repository

2. **Configure the Service**
   - **Name**: `invoice-extraction-app` (or any name you prefer)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`
   - **Plan**: Choose Starter (free tier) or upgrade for better performance

3. **Add Environment Variables**
   - Go to "Environment" section
   - Add: `OPENAI_API_KEY` = `your-api-key-here`

4. **Deploy**
   - Click "Create Web Service"
   - Wait for the build to complete

## Important Notes

### First Deployment
- The first deployment may take longer (10-15 minutes) because:
  - EasyOCR needs to download detection and recognition models
  - All Python packages need to be installed
  - This is normal and only happens on the first deployment

### File Size Limits
- Render free tier has file size limits
- If you need to upload large PDFs, consider upgrading to a paid plan
- The app supports up to 15 files per batch (configurable in `streamlit_app.py`)

### Environment Variables
- Your `OPENAI_API_KEY` should be set in Render's environment variables
- Never commit API keys to your Git repository
- The app will automatically use environment variables on Render (no need for `.env` files)

### Monitoring
- Check the "Logs" tab in Render dashboard for any errors
- The "Metrics" tab shows CPU, memory, and request statistics

### Custom Domain (Optional)
- In your service settings, go to "Custom Domains"
- Add your domain and follow the DNS configuration instructions

## Troubleshooting

### App won't start
- Check the logs in Render dashboard
- Verify `OPENAI_API_KEY` is set correctly
- Ensure all dependencies in `requirements.txt` are correct

### EasyOCR model download warnings
- These are normal on first run
- Models are cached after first download
- Subsequent deployments will be faster

### Memory issues
- If you encounter memory errors, upgrade to a higher plan
- Consider reducing `MAX_UPLOAD_FILES` in `streamlit_app.py`

## Updating Your App

- Simply push changes to your Git repository
- Render will automatically redeploy if `autoDeploy: true` is set
- Or manually trigger a deploy from the Render dashboard

## Support

For Render-specific issues, check:
- Render Documentation: https://render.com/docs
- Render Community: https://community.render.com

