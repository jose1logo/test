# text

This is a web-ready version of your ESCS Captcha Solver application, configured for deployment on Render.

## Features

- FastAPI-based REST API for captcha solving
- Support for both regular images and GIFs
- Health check endpoint
- Optimized for cloud deployment

## Deployment Instructions

### How to Deploy on Render

1. Sign up for a [Render account](https://render.com/) if you don't have one
2. Make sure your project files are in a Git repository (GitHub, GitLab, etc.)
3. In Render dashboard, click on "New" and select "Web Service"
4. Connect your repository or provide the repository URL
5. Configure your service:
   - Name: `escs-captcha-solver` (or any name you prefer)
   - Environment: `Python`
   - Build Command: `apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Add the following environment variable:
   - Key: `PYTHON_VERSION`
   - Value: `3.9.0`
7. Select your plan (Free tier works for testing)
8. Click "Create Web Service"

### Important Notes

- Make sure to upload the data files (`data.cfg`, `data.weights`, and `data.nms`) to the same directory as `main.py`
- The application has fallback mechanisms if the data files aren't found, but for full functionality, ensure they're properly uploaded
- The free tier on Render has limitations in terms of processing power and may go to sleep after periods of inactivity
- For production use, consider upgrading to a paid plan

### Troubleshooting

If you encounter issues with the deployment:

1. Check the build logs in the Render dashboard for specific errors
2. Make sure all data files are properly uploaded to your repository
3. Ensure the build command installs all necessary system dependencies (the build command already includes the essential ones)
4. For large model files (like `data.weights`), ensure they're properly tracked in your Git repository (you may need Git LFS for large files)
5. If you get OpenCV-related errors, verify that the build command is installing the required system libraries (`libgl1-mesa-glx` and `libglib2.0-0`)

## API Usage

### Solve Captcha

```
POST /api/base64
```

Request body:
```json
{
  "base64": "your_base64_encoded_image"
}
```

Response:
```json
{
  "result": "captcha_solution",
  "counter": 1
}
```

### Health Check

```
GET /health
```

Response:
```json
{
  "status": "healthy",
  "version": "3.0"
}
```

## Local Testing

To run the application locally:

```
cd app
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then access the API at `http://localhost:8000`
