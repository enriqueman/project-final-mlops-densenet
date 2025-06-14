#!/bin/bash

# Default values
STAGE=${1:-dev}
AWS_REGION=${2:-us-east-1}

# Validate stage
if [[ "$STAGE" != "dev" && "$STAGE" != "prod" ]]; then
    echo "Error: Stage must be 'dev' or 'prod'"
    exit 1
fi

# Configuration
BUCKET_NAME="densenet-models-$STAGE"
MODELS_DIR="models"
S3_MODELS_DIR="models"

# Check AWS CLI and credentials
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo "Error: Could not get Account ID. Check your AWS credentials"
    exit 1
fi

echo "Configuring S3 bucket for models ($STAGE)..."

# Create bucket if it doesn't exist
echo "Checking/Creating S3 bucket: $BUCKET_NAME"
if ! aws s3api head-bucket --bucket "$BUCKET_NAME" 2>/dev/null; then
    echo "Creating new bucket..."
    if ! aws s3api create-bucket \
        --bucket "$BUCKET_NAME" \
        --region "$AWS_REGION"; then
        echo "Error creating bucket"
        exit 1
    fi

    # Enable versioning
    aws s3api put-bucket-versioning \
        --bucket "$BUCKET_NAME" \
        --versioning-configuration Status=Enabled

    # Configure lifecycle policy
    LIFECYCLE_CONFIG='{
        "Rules": [{
            "ID": "DeleteOldVersions",
            "Status": "Enabled",
            "NoncurrentVersionExpiration": {
                "NoncurrentDays": 30
            },
            "Filter": {
                "Prefix": ""
            }
        }]
    }'

    aws s3api put-bucket-lifecycle-configuration \
        --bucket "$BUCKET_NAME" \
        --lifecycle-configuration "$LIFECYCLE_CONFIG"

    # Block public access
    aws s3api put-public-access-block \
        --bucket "$BUCKET_NAME" \
        --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

    echo "Bucket created successfully"
else
    echo "Bucket already exists"
fi

# Create models directory in S3
echo "Creating models directory..."
aws s3api put-object --bucket "$BUCKET_NAME" --key "$S3_MODELS_DIR/"

# Upload models if they exist
if [ -d "$MODELS_DIR" ]; then
    echo "Uploading models..."
    if ! aws s3 cp "$MODELS_DIR" "s3://$BUCKET_NAME/$S3_MODELS_DIR" --recursive; then
        echo "Error: Failed to upload models"
        exit 1
    fi
    echo "Models uploaded successfully"
else
    echo "Warning: Models directory not found at ./$MODELS_DIR"
fi

echo ""
echo "✅ Setup completed successfully"
echo "📦 Bucket: $BUCKET_NAME"
echo "📍 Region: $AWS_REGION"
echo ""
echo "Bucket structure:"
aws s3 ls "s3://$BUCKET_NAME" --recursive 