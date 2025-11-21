# Quick Deployment Checklist for Render

## Before You Deploy

- [ ] Push all your code to Git (GitHub, GitLab, or Bitbucket)
- [ ] Verify `requirements.txt` is up to date
- [ ] Have your OpenAI API key ready

## Deployment Steps

1. **Sign up/Login to Render**
   - Go to https://render.com
   - Sign up or log in

2. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your Git repository
   - Select the repository with this code

3. **Configure Service** (if not using render.yaml)
   - Name: `invoice-extraction-app`
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`

4. **Add Environment Variable**
   - Go to "Environment" tab
   - Add: `OPENAI_API_KEY` = `your-actual-api-key-here`

5. **Deploy**
   - Click "Create Web Service"
   - Wait 10-15 minutes for first deployment (EasyOCR models download)

6. **Verify**
   - Check logs for any errors
   - Visit your app URL
   - Test with a sample PDF

## After Deployment

- [ ] Test the app with a sample invoice PDF
- [ ] Check logs for any warnings or errors
- [ ] Verify environment variables are set correctly
- [ ] Bookmark your app URL

## Troubleshooting

- **Build fails**: Check `requirements.txt` for any missing dependencies
- **App won't start**: Check logs, verify `OPENAI_API_KEY` is set
- **Slow first load**: Normal - EasyOCR downloads models on first run
- **Memory errors**: Upgrade to a higher plan or reduce file limits

## Need Help?

- See `README_RENDER.md` for detailed instructions
- Check Render docs: https://render.com/docs
- Check Render community: https://community.render.com

