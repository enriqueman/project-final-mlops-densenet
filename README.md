# DenseNet121 Fargate Inference API

## 📋 Prerequisites

- AWS CLI configured with valid credentials
- Docker installed
- Python 3.9+
- Node.js 20+ (for CDK)
- AWS CDK CLI installed (`npm install -g aws-cdk`)

## 🚀 Initial Setup

### 1. Setup Model Bucket

First, set up the S3 bucket for storing the model:

```bash
# Make script executable
chmod +x setup-models-bucket.sh

# For development environment (default)
./setup-models-bucket.sh

# For production environment
./setup-models-bucket.sh prod

# Specify custom AWS region
./setup-models-bucket.sh dev us-west-2
```

This will:
- Create a dedicated S3 bucket (`densenet-models-dev` or `densenet-models-prod`)
- Configure bucket settings (versioning, lifecycle policy, security)
- Create the `models` directory structure
- Upload models from your local `./models` directory

### 2. Setup Test Data Bucket

Set up the S3 bucket for test data:

```bash
# Make script executable
chmod +x setup-test-bucket.sh

# For development environment (default)
./setup-test-bucket.sh

# For production environment
./setup-test-bucket.sh prod

# Specify custom AWS region
./setup-test-bucket.sh dev us-west-2
```

This will:
- Create a dedicated S3 bucket (`densenet-test-data-dev` or `densenet-test-data-prod`)
- Configure bucket settings
- Create the test data directory structure
- Upload test data and labels

## 🏗️ Local Development

### 1. Environment Setup

Create a `.env` file with your AWS credentials:

```bash
AWS_REGION=us-east-1
STAGE=dev
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_ACCOUNT_ID=your_account_id
```

### 2. Run with Docker Compose

```bash
docker-compose up --build
```

The application will be available at `http://localhost:8000`

## 🚀 Deployment

### GitHub Actions Pipeline

The project uses GitHub Actions for automated deployment. The pipeline will:

1. Run tests
2. Build and test Docker container
3. Push to ECR
4. Deploy infrastructure with CDK

#### Required GitHub Secrets:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

### Manual Deployment

```bash
# Install dependencies
pip install -r requirements.txt
pip install aws-cdk-lib constructs
npm install -g aws-cdk

# Deploy with CDK
cdk deploy --app="python3 ${PWD}/app.py" \
  --require-approval=never \
  -c stage=dev \
  -c region=us-east-1 \
  densenet-fargate-dev
```

## 📊 API Endpoints

- `GET /` - Health check
- `GET /health` - Detailed health check
- `GET /health/basic` - Basic health check
- `GET /debug` - Debug information
- `POST /predict` - Image prediction endpoint
- `GET /logs/predictions` - View prediction logs
- `GET /logs/predictions/download` - Download logs
- `GET /logs/predictions/stats` - View log statistics
- `DELETE /logs/predictions` - Clear logs (dev only)

## 🔧 Infrastructure

The infrastructure is deployed using AWS CDK and includes:

- ECS Fargate Service
- Application Load Balancer
- API Gateway
- S3 Buckets for models and logs
- IAM roles and policies
- VPC and networking components

## 📝 Logging

Predictions are automatically logged to S3 with the following information:

```json
{
  "timestamp": "2024-03-14T12:01:23.456789",
  "filename": "image.jpg",
  "top_prediction": "class_name",
  "confidence": 0.8542,
  "processing_time": 0.1234,
  "all_predictions": ["class1", "class2", "class3"],
  "confidence_scores": [0.8542, 0.1234, 0.0098]
}
```

## 🔒 Security

- All S3 buckets are private with no public access
- IAM roles follow least privilege principle
- API Gateway with VPC Link for secure communication
- Health checks and monitoring enabled

## 🧪 Testing

Run tests locally:

```bash
# Install test dependencies
pip install -r tests/requirements-test.txt

# Run tests
pytest tests/test_model.py -v
```

## 📚 Project Structure

```
.
├── app/                    # Application code
├── cdk_fargate_deploy/     # CDK infrastructure code
├── scripts/               # Utility scripts
├── tests/                 # Test files
├── Dockerfile            # Container definition
├── docker-compose.yml    # Local development
├── requirements.txt      # Python dependencies
└── setup-*.sh           # Setup scripts
```