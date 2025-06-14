
#!/usr/bin/env python3
import os

import aws_cdk as cdk

from cdk_fargate_deploy.cdk_fargate_deploy_stack import CdkFargateDeployStack

# Obtener stage y region desde el contexto o variables de entorno
app = cdk.App()

# Obtener valores del contexto CDK
stage = app.node.try_get_context('stage') or os.environ.get('STAGE', 'dev')
region = app.node.try_get_context('region') or os.environ.get('AWS_REGION', 'us-east-1')

# Generar nombre del stack dinámicamente
stack_name = f"densenet-fargate-{stage}"

print(f"Creando stack: {stack_name} para stage: {stage} en region: {region}")

CdkFargateDeployStack(app, stack_name,
    # Set environment explicitly
    env=cdk.Environment(account='471112837636', region=region),
    
    # For more information, see https://docs.aws.amazon.com/cdk/latest/guide/environments.html
)

app.synth()