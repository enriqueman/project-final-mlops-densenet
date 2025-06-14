from aws_cdk import (
    Duration,
    Stack,
    CfnOutput,
    aws_iam as iam,
)
from constructs import Construct
from aws_cdk import (aws_ec2 as ec2, aws_ecs as ecs,
                     aws_ecs_patterns as ecs_patterns,
                     aws_apigatewayv2 as apigwv2,
                     aws_apigatewayv2_integrations as integrations,
                     aws_s3 as s3,
                     )

class CdkFargateDeployStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        # Get environment from context
        stage = self.node.try_get_context('stage') or 'dev'
        region = self.node.try_get_context('region') or 'us-east-1'
        account_id = Stack.of(self).account
        
        # Generar un deployment ID único para forzar nuevas task definitions
        import time
        deployment_id = str(int(time.time()))
        
        # Models bucket name based on stage
        models_bucket_name = f"densenet-models-{stage}"
        
        # Logs bucket name (same for all stages)
        logs_bucket_name = "predics"
        
        # Create VPC
        vpc = ec2.Vpc(self, f"VpcFargate-{stage}", max_azs=2)
        
        # Create ECS Cluster
        cluster = ecs.Cluster(self, f"ClusterFargate-{stage}", vpc=vpc)
        
        # Create task role with S3 permissions
        task_role = iam.Role(
            self,
            f"TaskRole-{stage}",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            description=f"Task role for Fargate service - {stage}"
        )
        
        # Add S3 permissions for models bucket
        task_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "s3:GetObject",
                    "s3:ListBucket",
                    "s3:HeadObject"
                ],
                resources=[
                    f"arn:aws:s3:::{models_bucket_name}",
                    f"arn:aws:s3:::{models_bucket_name}/*"
                ]
            )
        )
        
        # Add S3 permissions for logs bucket (predictions logging)
        task_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:ListBucket",
                    "s3:HeadObject",
                    "s3:CreateBucket"  # Para crear bucket si no existe
                ],
                resources=[
                    f"arn:aws:s3:::{logs_bucket_name}",
                    f"arn:aws:s3:::{logs_bucket_name}/*"
                ]
            )
        )
        
        # Create execution role
        execution_role = iam.Role(
            self,
            f"ExecutionRole-{stage}",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AmazonECSTaskExecutionRolePolicy")
            ]
        )
        
        # Create Fargate Service with ALB
        service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            f"FargateService-{stage}",
            cluster=cluster,
            cpu=1024,  # Increased for ML workload
            desired_count=1,
            task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                image=ecs.ContainerImage.from_registry(
                    f"{account_id}.dkr.ecr.{region}.amazonaws.com/app-{stage}:latest"
                ),
                container_port=80,
                task_role=task_role,
                execution_role=execution_role,
                environment={
                    "STAGE": stage,
                    "AWS_REGION": region,
                    "MODELS_BUCKET": models_bucket_name,
                    "LOGS_BUCKET": logs_bucket_name,
                    "DEPLOYMENT_ID": deployment_id  # Forzar nueva task definition
                }
            ),
            memory_limit_mib=3072,  # Increased for ML workload
            public_load_balancer=True
        )
        
        # Configure Health Check - Volver a la raíz para simplificar
        service.target_group.configure_health_check(
            port="traffic-port",
            path="/",                           # ✅ Cambiar de vuelta a "/"
            interval=Duration.seconds(60),      # Más tiempo para carga del modelo
            timeout=Duration.seconds(15),       # Timeout más generoso
            healthy_threshold_count=2,          # Menos estricto
            unhealthy_threshold_count=5,        # Más tolerante a fallos
            healthy_http_codes="200,201,202,204"
        )
        
        # VPC Link
        vpc_link = apigwv2.CfnVpcLink(
            self,
            f"HttpVpcLink-{stage}",
            name=f"V2_VPC_Link_{stage}",
            subnet_ids=[subnet.subnet_id for subnet in vpc.private_subnets],
            security_group_ids=[service.service.connections.security_groups[0].security_group_id]
        )
        
        # Create HTTP API
        api = apigwv2.HttpApi(
            self,
            f"HttpApi-{stage}",
            api_name=f"ApigwFargate-{stage}",
            description=f"Integration between API Gateway and Fargate Service - {stage}"
        )
        
        # API Integration
        integration = apigwv2.CfnIntegration(
            self,
            f"HttpApiGatewayIntegration-{stage}",
            api_id=api.http_api_id,
            connection_id=vpc_link.ref,
            connection_type="VPC_LINK",
            description=f"API Integration with Fargate Service - {stage}",
            integration_method="ANY",
            integration_type="HTTP_PROXY",
            integration_uri=service.listener.listener_arn,
            payload_format_version="1.0"
        )
        
        # API Route
        route = apigwv2.CfnRoute(
            self,
            f"Route-{stage}",
            api_id=api.http_api_id,
            route_key="ANY /{proxy+}",
            target=f"integrations/{integration.ref}"
        )
        
        # Outputs
        CfnOutput(
            self,
            "APIGatewayUrl",
            description=f"API Gateway URL for {stage} environment",
            value=api.url
        )
        
        CfnOutput(
            self,
            "LoadBalancerDNS",
            description=f"Load Balancer DNS for {stage} environment",
            value=service.load_balancer.load_balancer_dns_name
        )
        
        CfnOutput(
            self,
            "LogsBucket",
            description=f"S3 bucket for prediction logs - {stage} environment",
            value=logs_bucket_name
        )