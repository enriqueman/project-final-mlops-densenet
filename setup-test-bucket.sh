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
BUCKET_NAME="densenet-test-data-$STAGE"
TEST_DATA_DIR="test-data"
TEST_IMAGES_DIR="test-data/images"
TEST_LABELS_FILE="test-data/imagenet_labels.txt"

# Check AWS CLI and credentials
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo "Error: Could not get Account ID. Check your AWS credentials"
    exit 1
fi

echo "Configuring S3 bucket for test data ($STAGE)..."

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

# Create directory structure
echo "Creating directory structure..."

# Create necessary directories
aws s3api put-object --bucket "$BUCKET_NAME" --key "$TEST_DATA_DIR/"
aws s3api put-object --bucket "$BUCKET_NAME" --key "$TEST_IMAGES_DIR/"

# Upload labels file if exists
if [ -f "$TEST_LABELS_FILE" ]; then
    echo "Uploading labels file..."
    if ! aws s3 cp "$TEST_LABELS_FILE" "s3://$BUCKET_NAME/$TEST_LABELS_FILE"; then
        echo "Warning: Error uploading labels file"
    fi
fi

echo ""
echo "✅ Setup completed successfully"
echo "📦 Bucket: $BUCKET_NAME"
echo "📍 Region: $AWS_REGION"
echo ""
echo "Bucket structure:"
aws s3 ls "s3://$BUCKET_NAME" --recursive 