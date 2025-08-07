# Free Tier AI/MLOps Learning Plan 🚀

## 🎯 Strategic Approach
Maximize 90+ free cloud services across AWS, Azure, and GCP to build production AI/MLOps skills WITHOUT spending your $1000 GCP credits initially.

## 🔥 Priority 1: Core MLOps Services (Start Here!)

### Week 1-2: Serverless AI APIs
**Services:** AWS Lambda (1M req/mo) + API Gateway + DynamoDB (25GB)
**Project:** Build a Multi-Model AI API Gateway
```python
# Deploy on AWS Lambda
- Text Analysis API using AWS Comprehend
- Image Analysis using Rekognition
- Sentiment Analysis pipeline
- Store results in DynamoDB
```
**Skills:** Serverless architecture, API design, NoSQL

### Week 3-4: Container Orchestration
**Services:** Azure Container Apps (180K vCPU seconds) + Azure Container Registry
**Project:** Deploy Dockerized ML Model
```yaml
# Azure Container Apps deployment
- Containerize a Hugging Face model
- Auto-scaling configuration
- Load balancing setup
- Health checks & monitoring
```
**Skills:** Docker, Kubernetes concepts, container orchestration

### Week 5-6: ML Pipeline Automation
**Services:** GCP Cloud Build (120 min/day) + Cloud Functions (2M invocations)
**Project:** Automated Model Training Pipeline
```bash
# Cloud Build pipeline
- Trigger on Git push
- Run training script
- Validate model metrics
- Deploy if threshold met
```
**Skills:** CI/CD, MLOps automation, GitOps

## 🎓 Priority 2: AI/ML Specific Services

### Month 2: Computer Vision & NLP
| Service | Free Tier | Project Idea |
|---------|-----------|--------------|
| **GCP Vision API** | 1,000 units/mo | Build OCR document processor |
| **Azure Face API** | 30,000 transactions | Create attendance system |
| **Azure Language Service** | 5,000 text records | Build customer feedback analyzer |
| **Azure Translator** | 2M characters | Multi-language chatbot |
| **GCP Speech-to-Text** | 60 min/mo | Podcast transcription tool |
| **Azure Text-to-Speech** | 500K characters | AI narrator for blogs |

### Sample Project: Document Intelligence Pipeline
```python
# Combine multiple free services
1. Azure Document Intelligence (3M chars) - Extract text
2. Azure Language Service - Analyze sentiment/entities  
3. Azure Translator - Multi-language support
4. Store in Cosmos DB (25GB free)
5. Query with Azure AI Search (50MB)
```

## 📊 Priority 3: Data & Analytics

### BigQuery Mastery (1TB queries/month FREE!)
**Week 1:** SQL for ML
```sql
-- Use public datasets
CREATE MODEL `project.dataset.model`
OPTIONS(model_type='logistic_reg') AS
SELECT * FROM `bigquery-public-data.ml_datasets.iris`
```

**Week 2:** Real-time Analytics
```python
# Pub/Sub (10GB/mo) → Cloud Functions → BigQuery
- Stream IoT data simulation
- Real-time aggregations
- Dashboard with Looker Studio (free)
```

### Azure Cosmos DB Projects (1000 RU/s free)
- Build vector database for RAG
- Real-time recommendation engine
- Global multi-region setup

## 🔧 Priority 4: Infrastructure & DevOps

### Free Compute Resources
| Provider | Service | Specs | Use Case |
|----------|---------|-------|----------|
| **AWS** | EC2 t2.micro | 750 hrs/mo | ML development server |
| **GCP** | e2-micro | 720 hrs/mo | Model serving endpoint |
| **Azure** | App Service | 1 hr/day | Demo deployments |

### Kubernetes Learning Path
```yaml
# GKE Autopilot (1 free cluster/month!)
Week 1: Deploy first ML model on K8s
Week 2: Setup Kubeflow pipelines
Week 3: Implement auto-scaling
Week 4: Multi-model serving with KServe
```

## 🚨 Priority 5: Monitoring & Observability

### Complete Observability Stack (ALL FREE)
```python
# Azure Monitor + Application Insights
- Model performance tracking
- Drift detection alerts
- Custom metrics dashboards

# GCP Cloud Logging (50GB/mo)
- Centralized log aggregation
- Error tracking
- Audit trails

# AWS CloudWatch (limited free)
- Lambda function monitoring
- API Gateway metrics
- Cost tracking
```

## 💡 Clever Combinations (Multi-Cloud Magic)

### Project 1: Multi-Cloud RAG System
```
AWS Lambda (compute) → 
Azure AI Search (vector store) → 
GCP BigQuery (analytics) →
Azure Static Web Apps (frontend)
```

### Project 2: Global ML API
```
AWS CloudFront (CDN, 50GB) →
Azure Functions (compute) →
GCP Cloud Storage (5GB models) →
Azure Cosmos DB (global database)
```

### Project 3: Real-time ML Pipeline
```
Azure Event Grid (100K ops) →
GCP Pub/Sub (10GB) →
AWS Lambda (processing) →
GCP BigQuery (storage)
```

## 📅 12-Week Learning Schedule

### Weeks 1-4: Foundations
- [ ] Setup all cloud accounts
- [ ] Deploy first serverless function (each cloud)
- [ ] Build simple ML API
- [ ] Implement basic monitoring

### Weeks 5-8: ML Operations
- [ ] Container orchestration project
- [ ] CI/CD pipeline for ML
- [ ] Model versioning system
- [ ] A/B testing framework

### Weeks 9-12: Production Systems
- [ ] Multi-cloud architecture
- [ ] Real-time ML pipeline
- [ ] Complete RAG application
- [ ] Portfolio website on Azure Static Apps

## 🎯 Skills Checkpoint

After completing this free-tier journey, you'll have:
- ✅ Deployed models on 3 major clouds
- ✅ Built 10+ production-ready projects
- ✅ Mastered 30+ cloud services
- ✅ Created multi-cloud architectures
- ✅ Implemented MLOps best practices

## 💰 Cost Management Tips

1. **Set Billing Alerts** (all free)
   - AWS Budgets
   - Azure Cost Management
   - GCP Budget Alerts

2. **Auto-shutdown Scripts**
```python
# Cloud Functions to stop resources
- Schedule: Mon-Fri 9am-6pm only
- Weekend shutdown
- Idle detection
```

3. **Resource Tagging**
```yaml
tags:
  environment: "learning"
  auto-delete: "true"
  project: "free-tier"
```

## 🚀 Quick Start Commands

```bash
# AWS CLI
aws lambda create-function --function-name my-ml-api --runtime python3.9

# Azure CLI  
az functionapp create --name my-ml-app --consumption-plan-location eastus

# GCP CLI
gcloud functions deploy my-ml-function --runtime python39 --trigger-http
```

## 📚 Learning Resources

### Free Courses
- AWS ML University (free)
- Azure AI Fundamentals (free)
- Google Cloud Skills Boost (free monthly quota)

### Documentation Deep Dives
- AWS Well-Architected ML Lens
- Azure ML Best Practices
- GCP AI Platform guides

## 🎪 Bonus: Hidden Free Gems

1. **Azure DevOps** - 5 users, unlimited repos
2. **AWS CloudFormation** - Infrastructure as Code (free)
3. **GCP Secret Manager** - 10K operations/mo
4. **Azure Key Vault** - Certificate management
5. **All SQL Database Free Tiers** - Perfect for ML metadata stores

## 📈 Success Metrics

Track your progress:
- [ ] 3 clouds configured
- [ ] 10 services deployed
- [ ] 5 ML models in production
- [ ] 1 multi-cloud project
- [ ] 100% free tier usage

---

**Remember:** These free tiers reset monthly! Use them for continuous learning and experimentation. Save your $1000 GCP credits for compute-intensive training and production deployments.

**Next Step:** Start with Priority 1, Week 1 - Deploy your first serverless ML API on AWS Lambda TODAY! 🚀
