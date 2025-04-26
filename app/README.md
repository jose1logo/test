# ESCS Captcha Solver API

This is a web-ready version of your ESCS Captcha Solver application, configured for deployment on Render.

## Features

- FastAPI-based REST API for captcha solving
- Support for both regular images and GIFs
- Health check endpoint
- Optimized for cloud deployment

## Deployment Instructions

### How to Deploy on Render

1. Sign up for a [Render account](https://render.com/) if you don't have one
2. Click on "New" and select "Web Service"
3. Connect your GitHub or GitLab account, or select "Public Git Repository"
4. Enter the URL of your Git repository or upload this folder to a new repository
5. Configure your service:
   - Name: `escs-captcha-solver` (or any name you prefer)
   - Environment: `Python`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Select your plan (Free tier works for testing)
7. Click "Create Web Service"

### Important Notes

- Make sure to upload the data files (`data.cfg`, `data.weights`, and `data.nms`) to the same directory as `main.py`
- The free tier on Render has limitations in terms of processing power and may go to sleep after periods of inactivity
- For production use, consider upgrading to a paid plan

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
